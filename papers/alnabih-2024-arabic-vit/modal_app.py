"""Bounded real-data diagnostic; full training waits for the conflicting split counts."""
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parent
app = modal.App("repro-0285c237-arabic-vit")
image = (modal.Image.from_registry("ghcr.io/sign-language-processing/reproduction:latest")
         .pip_install("transformers==4.46.3", "pyarrow==19.0.1", "pillow==11.1.0")
         .env({"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub"}))
data = modal.Volume.from_name("datasets", version=2)
cache = modal.Volume.from_name("huggingface-cache", version=2)
outputs = modal.Volume.from_name("repro-0285c237-results", create_if_missing=True, version=1)

@app.function(image=image, gpu="A100-80GB", timeout=1800, memory=16384,
              volumes={"/datasets": data.read_only(), "/cache/huggingface": cache, "/results": outputs})
def preflight():
    import hashlib, io, json, os, subprocess, time
    from datetime import datetime, timezone
    import numpy as np
    import pyarrow.parquet as pq
    from PIL import Image
    import torch
    from transformers import ViTForImageClassification, ViTImageProcessor
    torch.manual_seed(42)
    torch.set_num_threads(8)
    folder = Path("/results/preflight-2")
    folder.mkdir(exist_ok=True, parents=True)
    if (folder / "metrics.json").exists():
        print((folder / "metrics.json").read_text())
        return (folder / "metrics.json").read_text()
    started = datetime.now(timezone.utc).isoformat()
    source = Path("/datasets/arasl-database-grayscale/data/train-00000-of-00001-aa6a48ea2f282316.parquet")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == "7c6d9b276f5960bf9fb0efc99c7df3d3854b0690101751f74ab30d68a125d3a3"
    table = pq.read_table(source)
    assert len(table) == 54049
    labels = table["label"].to_numpy()
    assert len(set(labels)) == 32
    permutation = np.random.default_rng(42).permutation(len(table))
    # Disposable diagnostic split, not the paper's unreleased split.
    train_ids, test_ids = permutation[:32], permutation[-32:]
    revision = "6074eaf2211423e928c93b93ef773d5da618aa7e"
    name = "google/vit-large-patch16-224-in21k"
    processor = ViTImageProcessor.from_pretrained(name, revision=revision)
    model = ViTForImageClassification.from_pretrained(name, revision=revision, num_labels=32).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    def batch(indices):
        rows = table.take(indices).to_pylist()
        images = [Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB") for r in rows]
        pixels = processor(images=images, return_tensors="pt")["pixel_values"].cuda()
        return pixels, torch.tensor([r["label"] for r in rows], device="cuda")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.monotonic()
    model.train()
    losses = []
    for i in range(0, 32, 8):
        x, y = batch(train_ids[i:i+8])
        optimizer.zero_grad(set_to_none=True)
        loss = model(x, labels=y).loss
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    torch.cuda.synchronize()
    training_seconds = time.monotonic() - start
    checkpoint = folder / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": 4,
                "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}, checkpoint)
    del optimizer
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["rng"])
    torch.cuda.set_rng_state(state["cuda_rng"])
    del state
    x, y = batch(train_ids[:8])
    optimizer.zero_grad(set_to_none=True)
    loss = model(x, labels=y).loss
    loss.backward()
    optimizer.step()
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(0, 32, 8):
            x, y = batch(test_ids[i:i+8])
            predictions.extend(model(x).logits.argmax(1).cpu().tolist())
    correct = int(np.sum(np.array(predictions) == labels[test_ids]))
    result = {"scope": "conditional engineering preflight, not a target result", "started_at_utc": started,
              "finished_at_utc": datetime.now(timezone.utc).isoformat(), "seed": 42,
              "train_indices": train_ids.tolist(), "eval_indices": test_ids.tolist(),
              "dataset_examples": len(table), "classes": len(set(labels)), "training_losses": losses,
              "resume_loss": loss.item(), "checkpoint_resume_verified": True,
              "training_seconds": training_seconds, "examples_per_second": 32/training_seconds,
              "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
              "correct": correct, "evaluated": 32, "accuracy_percent": 100*correct/32,
              "predictions": predictions, "labels": labels[test_ids].tolist(),
              "model": name, "model_revision": revision, "torch": torch.__version__,
              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
              "modal_task_id": os.environ.get("MODAL_TASK_ID"), "modal_image_id": os.environ.get("MODAL_IMAGE_ID")}
    for name, command in [("pip-freeze.txt", ["pip", "freeze"]), ("nvidia-smi.txt", ["nvidia-smi"])]:
        (folder/name).write_text(subprocess.check_output(command,text=True))
    (folder/"metrics.json").write_text(json.dumps(result, indent=2))
    outputs.commit()
    print(json.dumps(result))
    return json.dumps(result)

@app.local_entrypoint()
def main():
    preflight.remote()

@app.function(image=image, timeout=600, volumes={"/results": outputs.read_only(), "/cache/huggingface": cache})
def evidence():
    import hashlib, json
    manifest = []
    for path in sorted(Path('/results').glob('preflight-*/*')):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(chunk)
        manifest.append({'path': str(path.relative_to('/results')), 'sha256': h.hexdigest(), 'bytes': path.stat().st_size})
    print(json.dumps(manifest))
