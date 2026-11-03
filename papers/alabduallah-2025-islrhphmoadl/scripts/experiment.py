"""Conditional ISLRHP-HMOADL reconstruction. See reproduction.json for approvals."""
import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.signal import wiener
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from data import CLASSES, ROOT, MANIFEST, load_manifest, sha256, write_json
from search import initialize, decode, key, propose, leaders_from_history

SEED = 42
BATCH = 5  # Paper setting; do not reduce to fit memory.


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


class Images(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(ROOT / row["path"]) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            channels = [wiener(pixels[:, :, c], (3, 3), noise=None) for c in range(3)]
        pixels = np.stack(channels, axis=-1)
        # scipy wiener can form 0/0 on a locally constant image with zero noise.
        # Do not silently invent a correction or carry NaNs into training.
        if not np.isfinite(pixels).all():
            raise FloatingPointError(f"Nonfinite Wiener output: {row['path']}")
        tensor = torch.from_numpy(np.clip(pixels, 0, 255).astype(np.float32)).permute(2, 0, 1)
        tensor = TF.resize(tensor, [224, 224], interpolation=InterpolationMode.BILINEAR, antialias=True)
        tensor = TF.normalize(tensor / 255.0, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return tensor, row["label"]


class Recognizer(nn.Module):
    def __init__(self):
        super().__init__()
        res = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1)
        self.resnext = nn.Sequential(*list(res.children())[:-2])
        self.vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.heads = nn.Identity()
        self.gru = nn.GRU(3328, 256, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.head = nn.Linear(512, 29)
        self.last_shapes = None

    def backbone_parameters(self):
        for module in (self.resnext, self.vgg, self.vit):
            yield from module.parameters()

    def stage(self, unfrozen):
        self.train()
        # CNNs remain fixed feature extractors, including normalization buffers.
        for module in (self.resnext, self.vgg):
            module.eval()
            module.requires_grad_(False)
        self.vit.train(unfrozen)
        self.vit.requires_grad_(unfrozen)

    def features(self, x):
        r, v = self.resnext(x), self.vgg(x)
        patches = self.vit._process_input(x)
        token = self.vit.class_token.expand(x.shape[0], -1, -1)
        patches = self.vit.encoder(torch.cat([token, patches], dim=1))[:, 1:]
        t = patches.transpose(1, 2).reshape(x.shape[0], 768, 14, 14)
        assert r.shape[1:] == (2048, 7, 7) and v.shape[1:] == (512, 7, 7)
        # 14 -> 7 is exactly non-overlapping 2x2 average pooling.
        fused = torch.cat([r, v, nn.functional.avg_pool2d(t, 2, 2)], dim=1)
        sequence = fused.flatten(2).transpose(1, 2)
        assert sequence.shape[1:] == (49, 3328)
        self.last_shapes = {"resnext": list(r.shape), "vgg": list(v.shape),
                            "vit": list(t.shape), "sequence": list(sequence.shape)}
        return sequence

    def forward(self, x):
        sequence = self.features(x)
        _, hidden = self.gru(sequence)
        return self.head(self.dropout(torch.cat([hidden[0], hidden[1]], dim=1)))


def metric_report(cm):
    cm = np.asarray(cm, dtype=np.int64)
    tp = np.diag(cm)
    fn, fp = cm.sum(1) - tp, cm.sum(0) - tp
    total = int(cm.sum())
    tn = total - tp - fn - fp
    pairs = {"accuracy": (tp + tn, np.full(29, total)), "precision": (tp, tp + fp),
             "sensitivity": (tp, tp + fn), "specificity": (tn, tn + fp),
             "f1": (2 * tp, 2 * tp + fp + fn)}
    per_class, zero = {}, []
    for name, (num, den) in pairs.items():
        per_class[name] = (100 * np.divide(num, den, out=np.zeros(29, dtype=float), where=den != 0)).tolist()
        zero.extend({"class": CLASSES[i], "metric": name} for i in np.where(den == 0)[0])
    return {"class_order": CLASSES, "sample_count": total, "confusion_matrix": cm.tolist(),
            "misclassified": total - int(tp.sum()), "per_class": per_class,
            "macro_percent": {k: float(np.mean(v)) for k, v in per_class.items()},
            "multiclass_accuracy_percent": 100 * int(tp.sum()) / total if total else 0,
            "zero_denominators": zero}


def loader(records, epoch=None):
    generator = torch.Generator().manual_seed(SEED if epoch is None else SEED + epoch)
    return DataLoader(records if isinstance(records, Dataset) else Images(records), batch_size=BATCH, shuffle=epoch is not None,
                      num_workers=8, pin_memory=True, generator=generator)


def rows_for(manifest, split, limit_per_class=None):
    rows = [r for r in manifest["records"] if r["split"] == split]
    if limit_per_class is not None:
        rows = [r for i in range(29) for r in [r for r in rows if r["label"] == i][:limit_per_class]]
    return rows


@torch.no_grad()
def evaluate(model, records):
    model.eval()
    cm = torch.zeros(29, 29, dtype=torch.int64, device="cuda")
    started = time.monotonic()
    for x, y in loader(records):
        pred = model(x.cuda(non_blocking=True)).argmax(1)
        indices = y.cuda(non_blocking=True) * 29 + pred
        cm += torch.bincount(indices, minlength=29 * 29).reshape(29, 29)
    torch.cuda.synchronize()
    result = metric_report(cm.cpu().numpy())
    result["seconds"] = time.monotonic() - started
    return result


def optimizer_for(model, head_lr, backbone_multiplier):
    return torch.optim.SGD([
        {"params": list(model.gru.parameters()) + list(model.head.parameters()), "lr": head_lr},
        {"params": list(model.backbone_parameters()), "lr": head_lr * backbone_multiplier}],
        momentum=0.9, weight_decay=0)


def save_checkpoint(path, model, optimizer, epoch, params, best_error):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                "params": params, "best_error": best_error, "split_sha256": sha256(MANIFEST),
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}, tmp)
    tmp.replace(path)


def restore(path, model, optimizer):
    # Only our own checkpoint, in the paper-specific project output volume.
    c = torch.load(path, map_location="cpu", weights_only=False)
    if c["split_sha256"] != sha256(MANIFEST):
        raise ValueError("Checkpoint dataset identity differs")
    model.load_state_dict(c["model"])
    optimizer.load_state_dict(c["optimizer"])
    torch.set_rng_state(c["torch_rng"])
    torch.cuda.set_rng_state_all(c["cuda_rng"])
    np.random.set_state(c["numpy_rng"])
    random.setstate(c["python_rng"])
    return c


def train_epoch(model, optimizer, records, epoch, max_steps=None):
    model.stage(epoch > 5)
    losses, times = [], []
    started = time.monotonic()
    for step, (x, y) in enumerate(loader(records, epoch)):
        optimizer.zero_grad(set_to_none=True)
        output = model(x.cuda(non_blocking=True))
        loss = nn.functional.cross_entropy(output, y.cuda(non_blocking=True))
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        if not torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all():
            raise FloatingPointError("Nonfinite gradient")
        optimizer.step()
        torch.cuda.synchronize()
        now = time.monotonic()
        times.append(now - started)
        started = now
        losses.append(float(loss.detach()))
        if max_steps is not None and step + 1 >= max_steps:
            break
    return {"steps": len(losses), "mean_loss": statistics.mean(losses),
            "seconds": sum(times), "steady_step_seconds": statistics.mean(times[2:] or times),
            "step_seconds": times}


def environment(output):
    output.mkdir(parents=True, exist_ok=True)
    freeze = subprocess.check_output(["python", "-m", "pip", "freeze"], text=True)
    (output / "pip-freeze.txt").write_text(freeze)
    smi = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"], text=True)
    (output / "hardware.txt").write_text(smi)
    weights = Path(torch.hub.get_dir()) / "checkpoints"
    return {"torch": torch.__version__, "cuda": torch.version.cuda,
            "hardware": smi.strip(), "pip_freeze_sha256": sha256(output / "pip-freeze.txt"),
            "script_sha256": sha256(__file__), "split_sha256": sha256(MANIFEST),
            "precision": "float32; TF32 disabled", "batch_size": BATCH,
            "pretrained_weights": [{"file": p.name, "sha256": sha256(p)} for p in sorted(weights.glob("*.pth"))
                                   if p.name.startswith(("resnext101_32x8d-", "vgg19-", "vit_b_16-"))]}


def preflight(output):
    if output.exists():
        raise FileExistsError("Retain prior preflight output; choose a new attempt name")
    output.mkdir(parents=True)
    manifest = load_manifest()
    seed_all()
    model = Recognizer().cuda()
    optimizer = optimizer_for(model, 0.01, 0.1)
    env = environment(output)
    train_rows = rows_for(manifest, "train", 5)
    valid_rows = rows_for(manifest, "validation", 2)
    torch.cuda.reset_peak_memory_stats()
    frozen = train_epoch(model, optimizer, train_rows, 1, max_steps=20)
    unfrozen = train_epoch(model, optimizer, train_rows, 6, max_steps=20)
    first_metrics = evaluate(model, valid_rows)
    checkpoint = output / "resume.pt"
    save_checkpoint(checkpoint, model, optimizer, 6, {"head_lr": 0.01, "backbone_multiplier": 0.1}, None)
    # Execute the identical next optimizer step twice across restore to verify
    # momentum, dropout RNG and BN buffers, not merely checkpoint deserialization.
    continuation1 = train_epoch(model, optimizer, train_rows, 7, max_steps=1)
    after1 = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    restore(checkpoint, model, optimizer)
    continuation2 = train_epoch(model, optimizer, train_rows, 7, max_steps=1)
    equal = all(torch.equal(after1[k], v.detach().cpu()) for k, v in model.state_dict().items())
    if not equal or continuation1["mean_loss"] != continuation2["mean_loss"]:
        raise AssertionError("Checkpoint/resume does not reproduce the next optimizer step")
    del after1
    restore(checkpoint, model, optimizer)
    metrics = evaluate(model, valid_rows)
    assert metrics["confusion_matrix"] == first_metrics["confusion_matrix"]
    # Final run only: the separately approved cached-classifier search requires
    # its own benchmark and is deliberately excluded from this estimate.
    train_seconds = 7308 * (5 * frozen["steady_step_seconds"] + 45 * unfrozen["steady_step_seconds"])
    eval_seconds = metrics["seconds"] / 58 * (50 * 4060 + 17400)
    result = {"conditional": True, "kind": "representative_preflight", "environment": env,
              "frozen": frozen, "unfrozen": unfrozen, "shapes": model.last_shapes,
              "validation_metrics": metrics, "resume_exact_next_step": equal,
              "checkpoint_sha256": sha256(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
              "estimate": {"training_seconds": train_seconds, "evaluation_seconds": eval_seconds,
                           "gpu_hours_before_checkpoint_and_startup_overhead": (train_seconds + eval_seconds) / 3600,
                           "conservative_gpu_hours_25pct_overhead": (train_seconds + eval_seconds) / 3600 * 1.25,
                           "basis": "Final 50-epoch run ONLY: CNNs frozen, ViT frozen five epochs then fine-tuned 45 epochs; classifier always trained. Excludes cached-feature search and cache construction. Batch 5, float32; tiny-subset timing is provisional."}}
    write_json(output / "preflight.json", result)
    return result


def train_configuration(output, params, epochs, final=False, until_epoch=None):
    """Idempotent epoch-boundary resume, fixed candidate endpoint or final best."""
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.json").exists():
        result = json.loads((output / "metrics.json").read_text())
        if result["params"] != params or result["split_sha256"] != sha256(MANIFEST):
            raise ValueError("Existing completed configuration identity differs")
        return result
    manifest = load_manifest()
    seed_all()
    model = Recognizer().cuda()
    optimizer = optimizer_for(model, **params)
    environment(output)
    start, best = 1, None
    if (output / "last.pt").exists():
        checkpoint = restore(output / "last.pt", model, optimizer)
        assert checkpoint["params"] == params
        start, best = checkpoint["epoch"] + 1, checkpoint["best_error"]
    end = epochs if until_epoch is None else min(until_epoch, epochs)
    for epoch in range(start, end + 1):
        timing = train_epoch(model, optimizer, rows_for(manifest, "train"), epoch)
        report = evaluate(model, rows_for(manifest, "validation")) if final or epoch == epochs else None
        improved = final and (best is None or report["misclassified"] < best)
        if improved:
            best = report["misclassified"]
        if improved:
            save_checkpoint(output / "best.pt", model, optimizer, epoch, params, best)
        save_checkpoint(output / "last.pt", model, optimizer, epoch, params, best)
        write_json(output / f"epoch-{epoch:02d}.json", {"epoch": epoch, "training": timing, "validation": report})
        print(json.dumps({"epoch": epoch, "mean_loss": timing["mean_loss"],
                          "validation_errors": None if report is None else report["misclassified"]}), flush=True)
    if end < epochs:
        return {"complete": False, "last_epoch": end}
    if final:
        selected = restore(output / "best.pt", model, optimizer)
        result = evaluate(model, rows_for(manifest, "test"))
        result["selected_epoch"] = selected["epoch"]
        result["checkpoint_sha256"] = sha256(output / "best.pt")
    else:
        result = json.loads((output / f"epoch-{epochs:02d}.json").read_text())["validation"]
        result["checkpoint_sha256"] = sha256(output / "last.pt")
    result.update({"conditional": True, "params": params, "epochs": epochs, "split_sha256": sha256(MANIFEST)})
    write_json(output / "metrics.json", result)
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def campaign(output):
    """Execute one candidate or ten final-run epochs per bounded cloud call.

    Persist the synchronous population and search RNG before training. The CLI
    orchestrates calls, so a five-day campaign never occupies one 24-hour input.
    """
    output.mkdir(parents=True, exist_ok=True)
    from cached_search import build_cache, train_candidate
    cache_path = output / "feature-cache"
    if not (cache_path / "metadata.json").exists():
        build_cache(cache_path)
        return {"complete": False, "stage": "feature_cache_ready"}
    state_path = output / "search-state.json"
    rng = np.random.default_rng(SEED)
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if any(h["status"] != "completed" for h in state["history"]):
            raise RuntimeError("Prior candidate is incomplete: diagnose and record a budgeted resume attempt before continuing")
        if state.get("protocol") != "cached-head-search-frozen-cnn-v1":
            raise ValueError("Refusing incompatible historical search state")
        rng.bit_generator.state = state["rng"]
    else:
        state = {"protocol": "cached-head-search-frozen-cnn-v1", "generation": -1, "population": initialize(rng).tolist(), "next_agent": 0,
                 "history": [], "rng": rng.bit_generator.state, "trace": []}
        write_json(state_path, state)
    if state["next_agent"] == 5 and state["generation"] < 2:
        state["generation"] += 1
        leaders, errors = leaders_from_history(state["history"])
        cache = {key(h["position"]): h["misclassified"] / 4060 for h in state["history"] if h["status"] == "completed"}
        population, trace = propose(np.array(state["population"]), leaders, errors, state["generation"], rng, cache)
        state.update(population=population.tolist(), next_agent=0, rng=rng.bit_generator.state)
        state["trace"].append(trace)
        write_json(state_path, state)
    if state["next_agent"] < 5:
        i = state["next_agent"]
        if len(state["history"]) >= 20:
            raise RuntimeError("Evaluation ceiling reached before population completed")
        position = state["population"][i]
        index = len(state["history"])
        attempt = {"evaluation": index, "position": position, "status": "reserved", "misclassified": None}
        state["history"].append(attempt)
        write_json(state_path, state)
        try:
            result = train_candidate(output / f"candidate-{index:02d}", cache_path, decode(position), 10)
        except Exception as exc:
            attempt.update(status="failed", error_type=type(exc).__name__)
            write_json(state_path, state)
            raise
        attempt.update(status="completed", misclassified=result["misclassified"])
        state["next_agent"] = i + 1
        write_json(state_path, state)
        return {"complete": False, "completed_search_evaluations": len(state["history"])}
    best = min(state["history"], key=lambda h: (h["misclassified"], h["evaluation"]))
    write_json(output / "selected-configuration.json", best)
    end = min(state.get("final_epoch", 0) + 5, 50)
    result = train_configuration(output / "final", decode(best["position"]), 50, final=True, until_epoch=end)
    state["final_epoch"] = end
    write_json(state_path, state)
    return {"complete": end == 50, "completed_search_evaluations": len(state["history"]),
            "final_epoch": end, "result": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["preflight", "campaign"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(preflight(args.output) if args.mode == "preflight" else campaign(args.output), allow_nan=False))
