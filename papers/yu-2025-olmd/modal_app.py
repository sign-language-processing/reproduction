"""Modal app for the OLMD reproduction.

Every function runs in the repro-sign workspace, mounts the shared `datasets`
Volume read-only at /datasets and the shared `huggingface-cache` Volume
read-write at /cache/huggingface, and writes outputs to a paper-specific
Volume. Invoke only through:

    .agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run \
        papers/yu-2025-olmd/modal_app.py::smoke
"""

from __future__ import annotations

import pathlib

import modal

PAPER_ID = "fa37313bf1a4558af1462085e70b884b1418fd50"
APP_NAME = "olmd-yu-2025"

HERE = pathlib.Path(__file__).parent

image = (
    modal.Image.from_registry("ghcr.io/sign-language-processing/reproduction:latest")
    .env(
        {
            # Shared Hugging Face cache, per AGENTS.md. It is an optimization,
            # never a source of provenance.
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_CACHE": "/cache/huggingface/hub",
            # torchvision downloads the ImageNet ResNet34 weights here too.
            "TORCH_HOME": "/cache/huggingface/torch",
        }
    )
    .add_local_file(HERE / "olmd.py", "/root/olmd.py")
    .add_local_file(HERE / "data.py", "/root/data.py")
    .add_local_file(HERE / "evaluate.py", "/root/evaluate.py")
)

datasets = modal.Volume.from_name("datasets")
hf_cache = modal.Volume.from_name("huggingface-cache")
results = modal.Volume.from_name(f"{APP_NAME}-results", create_if_missing=True)

VOLUMES = {
    "/datasets": datasets,
    "/cache/huggingface": hf_cache,
    "/outputs": results,
}

app = modal.App(APP_NAME)


@app.function(image=image, volumes=VOLUMES, gpu="A10G", timeout=60 * 30)
def smoke() -> dict:
    """Cheapest check that exercises the real model on real GPU tensors.

    This is NOT the AGENTS.md preflight: it loads no dataset and runs no
    evaluation. It exists only to confirm the model constructs, that shapes
    flow end to end, that the loss is finite, and to report the true parameter
    count against the paper's Table 5 figures.
    """
    import json

    import torch

    import olmd

    device = torch.device("cuda")
    num_classes = 1296  # PHOENIX14: 1295 glosses + CTC blank
    model = olmd.OLMD(num_classes=num_classes, context=9).to(device)
    criterion = olmd.OLMDLoss().to(device)

    counts = {
        "total": sum(p.numel() for p in model.parameters()),
        "backbone": sum(p.numel() for p in model.backbone.parameters()),
        "head": sum(p.numel() for p in model.head.parameters()),
        "lma": sum(p.numel() for p in model.backbone.lma.parameters()),
        "omp": sum(p.numel() for p in model.backbone.omp.parameters()),
        "cross_stage": sum(p.numel() for p in model.backbone.cross_stage.parameters()),
    }

    # Time a few training steps so the compute estimate reflects the real model.
    # Batch size 2 as the paper specifies. T=128 is a realistic PHOENIX clip
    # length; a 32-frame probe is too short for the {K5,P2,K5,P2} head and
    # crashes in max_pool1d, which is how the branch topology was settled.
    frames = 128
    video = torch.randn(2, 3, frames, 224, 224, device=device)
    lengths = torch.full((2,), frames, dtype=torch.long)
    # ~11.5 glosses is the PHOENIX14 training mean.
    targets = torch.randint(1, num_classes, (22,), dtype=torch.long)
    target_lengths = torch.tensor([11, 11], dtype=torch.long)

    model.train()
    outputs = model(video, lengths)
    losses = criterion(outputs, targets, target_lengths)
    losses["loss"].backward()

    # Median wall time of a full training step at T=128, for the compute estimate.
    import time as _time

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    step_times = []
    for _ in range(6):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = _time.time()
        step_losses = criterion(model(video, lengths), targets, target_lengths)
        step_losses["loss"].backward()
        optimizer.step()
        torch.cuda.synchronize()
        step_times.append(_time.time() - started)
    step_times.sort()
    median_step = step_times[len(step_times) // 2]

    grad_norm = sum(
        p.grad.pow(2).sum().item() for p in model.parameters() if p.grad is not None
    ) ** 0.5

    result = {
        "paper_id": PAPER_ID,
        "params": {key: int(value) for key, value in counts.items()},
        "params_total_M": round(counts["total"] / 1e6, 3),
        "paper_table5_ctx9_M": 80.74,
        "logit_shapes": {
            key: [int(dim) for dim in value.shape]
            for key, value in outputs.items()
            if key.endswith("logits")
        },
        "output_lengths": {
            key: [int(item) for item in value.tolist()]
            for key, value in outputs.items()
            if key.endswith("lengths")
        },
        "input_frames": int(frames),
        "loss": float(losses["loss"]),
        "ctc": float(losses["ctc"]),
        "distill": float(losses["distill"]),
        "grad_norm": float(grad_norm),
        "seconds_per_train_step_t128": round(float(median_step), 4),
        "lift": float(losses.get("lift", 0.0)),
        "peak_memory_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "gpu": str(torch.cuda.get_device_name(0)),
        "torch": str(torch.__version__),
    }

    # Persist the evidence on the paper-specific Volume, not only to stdout.
    output = pathlib.Path("/outputs/smoke")
    output.mkdir(parents=True, exist_ok=True)
    (output / "smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    results.commit()
    return result


@app.function(image=image, volumes=VOLUMES, timeout=60 * 20)
def inspect_data() -> dict:
    """Confirm the mounted corpora match what the data gate recorded.

    CPU-only: reads annotation files and counts frame directories. No training.
    """
    import data

    return data.inspect_all("/datasets")


@app.function(image=image, volumes=VOLUMES, timeout=60 * 20)
def diagnose_csl() -> dict:
    """Explain the one CSL-Daily training sample the loader does not yield."""
    from pathlib import Path

    import data

    return data.diagnose_csl_daily(Path("/datasets"))


@app.function(image=image, volumes=VOLUMES, gpu="A100-40GB", timeout=60 * 90)
def preflight(dataset: str = "phoenix14", train_steps: int = 12, eval_clips: int = 8) -> dict:
    """The AGENTS.md representative preflight.

    Exercises the real path at the cheapest representative scale:
      * loads the exact data format through the real loader and preprocessing;
      * loads the intended ImageNet-pretrained weights;
      * runs real optimizer steps with the paper's optimizer and schedule;
      * saves a checkpoint and reloads it, verifying parameters match;
      * runs the real evaluation code on a small subset and parses a WER;
      * measures throughput and peak memory for the compute estimate.

    Deliberately NOT a score: `eval_clips` utterances from an almost-untrained
    model cannot approximate the paper's numbers. The point is that the
    pipeline reaches a parsed metric with a terminal exit status.
    """
    import json
    import time

    import numpy as np
    import torch

    import data
    import evaluate
    import olmd

    device = torch.device("cuda")
    root = pathlib.Path("/datasets")
    started = time.time()

    # --- real data through the real loader -------------------------------
    vocabulary = data.build_vocabulary(root, dataset)
    num_classes = len(vocabulary) + 1  # index 0 is the CTC blank
    train_samples = data.LOADERS[dataset](root, "train")
    dev_samples = data.LOADERS[dataset](root, "dev")

    rng = np.random.default_rng(0)  # the paper states no seed; fixed for repeatability

    def batch_of(samples, indices, train):
        clips, lengths, targets, target_lengths, references = [], [], [], [], []
        for index in indices:
            sample = samples[index]
            frames = data.load_frames(root, sample)
            if train:
                keep = data.temporal_rescale(len(frames), rng)
                frames = frames[keep]
            clips.append(torch.from_numpy(data.preprocess(frames, train, rng)))
            lengths.append(len(frames))
            ids = [vocabulary[gloss] for gloss in sample.glosses if gloss in vocabulary]
            targets.extend(ids)
            target_lengths.append(len(ids))
            references.append(sample.glosses)

        # Replicate padding, matching the lineage: never zeros.
        longest = max(clip.shape[0] for clip in clips)
        padded = []
        for clip in clips:
            if clip.shape[0] < longest:
                tail = clip[-1:].expand(longest - clip.shape[0], -1, -1, -1)
                clip = torch.cat([clip, tail], dim=0)
            padded.append(clip)
        video = torch.stack(padded).permute(0, 2, 1, 3, 4)  # (N, C, T, H, W)
        return (
            video.to(device),
            torch.tensor([longest] * len(clips), dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
            torch.tensor(target_lengths, dtype=torch.long),
            references,
        )

    # --- intended weights ------------------------------------------------
    model = olmd.OLMD(num_classes=num_classes, context=9).to(device)
    criterion = olmd.OLMDLoss().to(device)
    # The paper: Adam, weight decay 0.001, initial LR 0.001, x0.3 at epochs 25/40.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 40], gamma=0.3)

    # --- real optimizer steps -------------------------------------------
    model.train()
    losses, step_times, load_times = [], [], []
    components = []
    order = rng.permutation(len(train_samples))
    for step in range(train_steps):
        picks = order[step * 2 : step * 2 + 2]  # batch size 2, per the paper
        # Time data loading and the GPU step separately: conflating them is what
        # made the first preflight's estimate unusable.
        load_started = time.time()
        video, lengths, targets, target_lengths, _ = batch_of(train_samples, picks, True)
        load_times.append(time.time() - load_started)

        torch.cuda.synchronize()
        step_started = time.time()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(video, lengths)
        parts = criterion(outputs, targets, target_lengths)
        parts["loss"].backward()
        optimizer.step()
        torch.cuda.synchronize()
        step_times.append(time.time() - step_started)
        losses.append(float(parts["loss"]))
        components.append(
            {
                "ctc": round(float(parts["ctc"]), 4),
                "distill": round(float(parts["distill"]), 6),
                "lift": round(float(parts.get("lift", 0.0)), 4),
            }
        )
    scheduler.step()

    # --- checkpoint save and reload -------------------------------------
    output = pathlib.Path("/outputs/preflight")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / f"{dataset}-preflight.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "num_classes": num_classes,
            "context": 9,
        },
        checkpoint_path,
    )
    reloaded = olmd.OLMD(num_classes=num_classes, context=9).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    reloaded.load_state_dict(state["model"])
    mismatched = [
        name
        for (name, before), (_, after) in zip(
            model.state_dict().items(), reloaded.state_dict().items()
        )
        if not torch.equal(before.float(), after.float())
    ]

    # --- real evaluation on a small subset ------------------------------
    reloaded.eval()
    references, hypotheses = [], []
    inverse = {index: gloss for gloss, index in vocabulary.items()}
    eval_started = time.time()
    with torch.no_grad():
        for index in range(0, min(eval_clips, len(dev_samples)), 2):
            picks = list(range(index, min(index + 2, eval_clips, len(dev_samples))))
            video, lengths, _, _, batch_references = batch_of(dev_samples, picks, False)
            outputs = reloaded(video, lengths)
            decoded = evaluate.greedy_ctc_decode(
                outputs["global_logits"], outputs["global_lengths"]
            )
            for reference, ids in zip(batch_references, decoded):
                references.append(evaluate.strip_phoenix_markers(reference))
                hypotheses.append([inverse[i] for i in ids if i in inverse])
    eval_seconds = time.time() - eval_started

    scored = evaluate.corpus_wer(references, hypotheses).as_dict()

    # --- compute estimate for the full run ------------------------------
    seconds_per_step = float(np.median(step_times))
    seconds_per_load = float(np.median(load_times))
    steps_per_epoch = len(train_samples) // 2  # batch size 2
    # A step needs both its GPU work and its two clips loaded. Serial today, so
    # additive; a prefetching DataLoader would overlap them.
    epoch_hours = (seconds_per_step + seconds_per_load) * steps_per_epoch / 3600
    dev_hours = len(dev_samples) * (eval_seconds / max(len(references), 1)) / 3600
    result = {
        "paper_id": PAPER_ID,
        "dataset": dataset,
        "num_classes": num_classes,
        "vocabulary": len(vocabulary),
        "train_utterances": len(train_samples),
        "dev_utterances": len(dev_samples),
        "train_steps": train_steps,
        "losses": [round(value, 4) for value in losses],
        "loss_first": round(losses[0], 4),
        "loss_last": round(losses[-1], 4),
        "seconds_per_gpu_step_median": round(seconds_per_step, 3),
        "seconds_per_batch_load_median": round(seconds_per_load, 3),
        "loss_components": components,
        "checkpoint": {
            "path": str(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "reload_verified": not mismatched,
            "mismatched_tensors": mismatched[:5],
        },
        "evaluation": {
            "clips": len(references),
            "seconds": round(eval_seconds, 2),
            **scored,
        },
        "estimate": {
            "steps_per_epoch": steps_per_epoch,
            "train_hours_per_epoch": round(epoch_hours, 3),
            "dev_hours_per_epoch": round(dev_hours, 3),
            "hours_per_epoch_total": round(epoch_hours + dev_hours, 3),
            "gpu_hours_70_epochs": round((epoch_hours + dev_hours) * 70, 1),
            "note": (
                "Single GPU, batch size 2, 70 epochs per the paper, INCLUDING a "
                "dev pass every epoch for best-dev checkpoint selection. Data "
                "loading is timed separately and added, since loading is serial "
                "here; a prefetching DataLoader would overlap it with GPU work. "
                "Measured with TLP pooling active."
            ),
        },
        "peak_memory_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "gpu": str(torch.cuda.get_device_name(0)),
        "torch": str(torch.__version__),
        "wall_seconds": round(time.time() - started, 1),
    }

    (output / f"{dataset}-preflight.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    results.commit()
    return result


@app.function(image=image, volumes=VOLUMES, gpu="A100-40GB", timeout=60 * 60)
def throughput(dataset: str = "phoenix14", clips: int = 24, workers: int = 8) -> dict:
    """Separate data-loading cost from GPU cost, which the preflight conflated.

    preflight-phoenix14-001 timed only the optimizer step, so its
    seconds_per_step excluded frame loading; its evaluation loop then showed
    50.4 s per clip, roughly 260x the per-clip training compute. Inference
    cannot cost more than training, so that figure is serial PNG I/O from a
    network Volume, not computation. This measures the two separately and tests
    whether parallel loading removes the bottleneck.
    """
    import json
    import time
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import torch

    import data
    import olmd

    root = pathlib.Path("/datasets")
    samples = data.LOADERS[dataset](root, "dev")[:clips]
    rng = np.random.default_rng(0)

    def fetch(sample):
        frames = data.load_frames(root, sample)
        return data.preprocess(frames, False), len(frames)

    # Serial loading, as the preflight did.
    serial_started = time.time()
    serial = [fetch(sample) for sample in samples[: min(4, clips)]]
    serial_seconds = (time.time() - serial_started) / len(serial)
    frame_counts = [count for _, count in serial]

    # Parallel loading with a thread pool: PNG decode releases the GIL and the
    # bottleneck is network I/O, so threads should help substantially.
    parallel_started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        loaded = list(pool.map(fetch, samples))
    parallel_seconds = (time.time() - parallel_started) / len(loaded)

    # Pure GPU cost, data already resident.
    vocabulary = data.build_vocabulary(root, dataset)
    model = olmd.OLMD(num_classes=len(vocabulary) + 1, context=9).to("cuda")
    model.eval()
    gpu_times = []
    with torch.no_grad():
        for clip, count in loaded[:8]:
            video = torch.from_numpy(clip).unsqueeze(0).permute(0, 2, 1, 3, 4).to("cuda")
            lengths = torch.tensor([clip.shape[0]], dtype=torch.long)
            torch.cuda.synchronize()
            started = time.time()
            model(video, lengths)
            torch.cuda.synchronize()
            gpu_times.append(time.time() - started)

    result = {
        "paper_id": PAPER_ID,
        "dataset": dataset,
        "clips_measured": len(loaded),
        "frames_per_clip": {
            "min": int(min(frame_counts)),
            "max": int(max(frame_counts)),
            "mean": round(float(np.mean(frame_counts)), 1),
        },
        "load_seconds_per_clip_serial": round(serial_seconds, 2),
        "load_seconds_per_clip_parallel": round(parallel_seconds, 2),
        "parallel_workers": workers,
        "speedup": round(serial_seconds / parallel_seconds, 2) if parallel_seconds else None,
        "gpu_inference_seconds_per_clip": round(float(np.median(gpu_times)), 4),
        "gpu": str(torch.cuda.get_device_name(0)),
    }
    output = pathlib.Path("/outputs/throughput")
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{dataset}-throughput.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    results.commit()
    return result


@app.function(image=image, volumes=VOLUMES, timeout=60 * 20)
def hash_output(relative_path: str) -> dict:
    """SHA-256 a file on the results Volume without downloading it."""
    import hashlib

    target = pathlib.Path("/outputs") / relative_path
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return {
        "path": str(target),
        "sha256": digest.hexdigest(),
        "size_bytes": target.stat().st_size,
    }


@app.function(image=image, volumes=VOLUMES, gpu=None, cpu=8.0, timeout=60 * 40)
def test_consolidation(dataset: str = "phoenix14", clips: int = 6) -> dict:
    """Test the hypothesis that per-file latency, not bandwidth, causes the I/O cost.

    throughput-phoenix14-001 measured 15.48 s per clip with 8 threads and only a
    1.79x thread speedup, which points at per-request latency across the ~179
    PNG files per utterance rather than at bytes moved. If that is right,
    consolidating each clip into ONE compressed file should recover most of the
    cost even on the same Volume. Writes to the results Volume, never to
    `datasets`, which stays read-only.
    """
    import io
    import json
    import time
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    from PIL import Image

    import data

    root = pathlib.Path("/datasets")
    cache = pathlib.Path("/outputs/cache-probe") / dataset
    cache.mkdir(parents=True, exist_ok=True)
    samples = data.LOADERS[dataset](root, "dev")[:clips]

    # Baseline: read the PNG directory as the training loop currently does.
    read_started = time.time()
    originals = []
    for sample in samples:
        frames = data.load_frames(root, sample)
        originals.append(frames)
    png_seconds = (time.time() - read_started) / len(samples)

    # Consolidate: resize to 256 once, JPEG-encode each frame, one .npz per clip.
    def write_one(pair):
        sample, frames = pair
        encoded = []
        for frame in frames:
            buffer = io.BytesIO()
            Image.fromarray(frame).resize((256, 256), Image.BILINEAR).save(
                buffer, format="JPEG", quality=90
            )
            encoded.append(np.frombuffer(buffer.getvalue(), dtype=np.uint8))
        path = cache / f"{sample.name}.npz"
        np.savez(path, **{str(i): item for i, item in enumerate(encoded)})
        return path.stat().st_size, len(frames)

    write_started = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        written = list(pool.map(write_one, zip(samples, originals)))
    write_seconds = (time.time() - write_started) / len(samples)
    results.commit()

    # Read back the consolidated form, which is what training would do each epoch.
    def read_one(sample):
        with np.load(cache / f"{sample.name}.npz") as bundle:
            keys = sorted(bundle.files, key=int)
            return np.stack(
                [np.asarray(Image.open(io.BytesIO(bundle[key].tobytes()))) for key in keys]
            )

    cached_started = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        restored = list(pool.map(read_one, samples))
    cached_seconds = (time.time() - cached_started) / len(samples)

    # The cache must be faithful: same shape, and JPEG q90 close to the original.
    shapes_match = all(
        original.shape[0] == item.shape[0] for original, item in zip(originals, restored)
    )
    deltas = []
    for original, item in zip(originals, restored):
        reference = np.stack(
            [
                np.asarray(Image.fromarray(frame).resize((256, 256), Image.BILINEAR))
                for frame in original
            ]
        )
        deltas.append(float(np.abs(reference.astype(np.int16) - item.astype(np.int16)).mean()))

    result = {
        "paper_id": PAPER_ID,
        "dataset": dataset,
        "clips": len(samples),
        "png_read_seconds_per_clip": round(png_seconds, 2),
        "cache_write_seconds_per_clip": round(write_seconds, 2),
        "cache_read_seconds_per_clip": round(cached_seconds, 3),
        "speedup_vs_png": round(png_seconds / cached_seconds, 1) if cached_seconds else None,
        "cache_bytes_per_clip_mean": int(np.mean([size for size, _ in written])),
        "frames_per_clip_mean": round(float(np.mean([count for _, count in written])), 1),
        "fidelity": {
            "frame_counts_match": shapes_match,
            "mean_abs_pixel_delta": round(float(np.mean(deltas)), 3),
            "note": "Delta is JPEG q90 loss against the same bilinear 256x256 resize.",
        },
    }
    output = pathlib.Path("/outputs/cache-probe")
    (output / f"{dataset}-probe.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    results.commit()
    return result


# Cache and outputs both live on the paper-specific Volume; `datasets` stays
# read-only, per AGENTS.md.
CACHE_ROOT = "/outputs/cache"


@app.function(image=image, volumes=VOLUMES, cpu=16.0, timeout=60 * 60 * 6)
def build_cache(
    dataset: str = "phoenix14", split: str = "all", workers: int = 16, limit: int = 0
) -> dict:
    """Build the JPEG frame cache. CPU-only, idempotent, resumable.

    One-time cost, reusable across every epoch, seed and both experiments. Safe
    to re-run: existing entries are skipped and each entry is written to a
    .partial name first, so an interrupted build cannot leave a truncated file
    that later reads would accept.
    """
    import json
    import time
    from concurrent.futures import ThreadPoolExecutor

    import data

    root = pathlib.Path("/datasets")
    cache_root = pathlib.Path(CACHE_ROOT)
    splits = ("train", "dev", "test") if split == "all" else (split,)

    started = time.time()
    report = {}
    for name in splits:
        samples = data.LOADERS[dataset](root, name)
        if limit:
            samples = samples[:limit]
        split_started = time.time()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            sizes = list(
                pool.map(
                    lambda sample: data.write_cache_entry(root, cache_root, dataset, sample),
                    samples,
                )
            )
        results.commit()
        report[name] = {
            "utterances": len(samples),
            "bytes": int(sum(sizes)),
            "gib": round(sum(sizes) / 2**30, 2),
            "seconds": round(time.time() - split_started, 1),
        }

    # Verify a cached entry reproduces the PNG path through the real
    # preprocessing, rather than trusting that files merely appeared.
    import numpy as np

    check_split = splits[0]
    check_samples = data.LOADERS[dataset](root, check_split)
    if limit:
        check_samples = check_samples[:limit]
    fidelity = {
        "checked": 0,
        "frame_counts_match": True,
        "max_abs_pixel_delta": 0.0,
        "mean_abs_pixel_delta": 0.0,
    }
    mean_deltas = []
    for sample in check_samples[:3]:
        from_png = data.preprocess(
            data.load_frames(root, sample), train=False, already_resized=False
        )
        from_cache = data.preprocess(
            data.read_cache_entry(cache_root, dataset, sample),
            train=False,
            already_resized=True,
        )
        fidelity["checked"] += 1
        if from_png.shape != from_cache.shape:
            fidelity["frame_counts_match"] = False
            continue
        difference = np.abs(from_png - from_cache)
        fidelity["max_abs_pixel_delta"] = max(
            fidelity["max_abs_pixel_delta"], float(difference.max())
        )
        mean_deltas.append(float(difference.mean()))
    fidelity["max_abs_pixel_delta"] = round(fidelity["max_abs_pixel_delta"], 4)
    fidelity["mean_abs_pixel_delta"] = round(
        float(np.mean(mean_deltas)) if mean_deltas else 0.0, 6
    )
    fidelity["pixel_levels"] = {
        "max": round(fidelity["max_abs_pixel_delta"] / (2 / 255), 1),
        "mean": round(fidelity["mean_abs_pixel_delta"] / (2 / 255), 3),
    }
    fidelity["note"] = (
        "Deterministic centre-crop path on both sides, so any difference is JPEG q90 "
        "loss alone. Inputs are scaled to [-1, 1], so 1 pixel level is about 0.0078. "
        "A max of roughly 20-30 levels with a mean near 1 is normal q90 behaviour: "
        "ringing concentrates at high-contrast edges while most pixels are near-exact."
    )

    total_bytes = sum(entry["bytes"] for entry in report.values())
    result = {
        "paper_id": PAPER_ID,
        "dataset": dataset,
        "splits": report,
        "total_gib": round(total_bytes / 2**30, 2),
        "wall_seconds": round(time.time() - started, 1),
        "workers": workers,
        "fidelity": fidelity,
    }
    output = pathlib.Path("/outputs/cache-build")
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{dataset}-cache.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    results.commit()
    return result


@app.function(image=image, volumes=VOLUMES, gpu="H100!", cpu=16.0, timeout=60 * 60)
def benchmark_loader_h100(
    dataset: str = "phoenix14", steps: int = 40, workers: int = 8, limit: int = 0
) -> dict:
    """Same benchmark on an H100, to measure rather than assume its speedup.

    The "H100!" form pins a real H100: a bare "H100" lets Modal substitute an
    H200, which would silently measure different hardware than the one being
    priced.
    """
    return _benchmark(dataset, steps, workers, limit, "H100")


@app.function(image=image, volumes=VOLUMES, gpu="A100-40GB", cpu=16.0, timeout=60 * 60)
def benchmark_loader(
    dataset: str = "phoenix14", steps: int = 40, workers: int = 8, limit: int = 0
) -> dict:
    """Measure whether the prefetching DataLoader actually hides loading latency.

    preflight-phoenix14-002 measured a 0.422 s GPU step against 5.326 s of
    serial batch loading, so the pipeline was 12.6x I/O-bound. If prefetching
    works, wall time per step should approach the GPU step time rather than the
    sum, and the estimate drops from about 57.8 to about 25.5 GPU-hours.
    """
    return _benchmark(dataset, steps, workers, limit, "A100-40GB")


def _benchmark(
    dataset: str, steps: int, workers: int, limit: int, label: str
) -> dict:
    """Shared benchmark body, so both GPU variants measure identically."""
    import json
    import time

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset

    import data
    import olmd

    root = pathlib.Path("/datasets")
    cache_root = pathlib.Path(CACHE_ROOT)
    vocabulary = data.build_vocabulary(root, dataset)

    class TorchDataset(Dataset):
        def __init__(self, inner):
            self.inner = inner

        def __len__(self):
            return len(self.inner)

        def __getitem__(self, index):
            return self.inner[index]

    inner = data.OLMDDataset(
        root, dataset, "train", vocabulary, cache_root=cache_root, train=True, seed=0
    )
    if limit:
        # Restrict to a cached prefix so the benchmark measures the cached path
        # rather than silently falling back to PNG reads for uncached clips.
        inner.samples = inner.samples[:limit]
    cached = sum(
        1
        for sample in inner.samples[:200]
        if data.cache_path(cache_root, dataset, sample).exists()
    )

    loader = DataLoader(
        TorchDataset(inner),
        batch_size=2,  # the paper's batch size
        shuffle=True,
        num_workers=workers,
        collate_fn=data.collate,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
        drop_last=True,
    )

    model = olmd.OLMD(num_classes=len(vocabulary) + 1, context=9).to("cuda")
    criterion = olmd.OLMDLoss().to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    model.train()

    wall_times, gpu_times = [], []
    iterator = iter(loader)
    epochs_consumed = 0
    for index in range(steps):
        step_started = time.time()
        try:
            batch = next(iterator)
        except StopIteration:
            # A short cached subset holds fewer batches than the requested step
            # count, so cycle epochs rather than crashing. Note this reuses the
            # same clips; on the full dataset each step sees fresh ones.
            epochs_consumed += 1
            iterator = iter(loader)
            batch = next(iterator)
        video, lengths, targets, target_lengths, _ = batch
        video = video.to("cuda", non_blocking=True)

        torch.cuda.synchronize()
        gpu_started = time.time()
        optimizer.zero_grad(set_to_none=True)
        losses = criterion(model(video, lengths), targets, target_lengths)
        losses["loss"].backward()
        optimizer.step()
        torch.cuda.synchronize()
        gpu_times.append(time.time() - gpu_started)
        wall_times.append(time.time() - step_started)

    # Discard the first few steps: worker startup and cuDNN autotuning inflate them.
    warm = max(5, steps // 4)
    wall = float(np.median(wall_times[warm:]))
    gpu = float(np.median(gpu_times[warm:]))

    steps_per_epoch = len(inner) // 2
    dev_count = len(data.LOADERS[dataset](root, "dev"))
    epoch_hours = wall * steps_per_epoch / 3600
    dev_hours = dev_count * (gpu / 2) / 3600  # inference only, prefetched

    result = {
        "paper_id": PAPER_ID,
        "dataset": dataset,
        "requested_gpu": label,
        "workers": workers,
        "steps": steps,
        "epochs_consumed": epochs_consumed,
        "dataset_batches": len(inner) // 2,
        "cache_hits_in_first_200": cached,
        "seconds_per_step_wall_median": round(wall, 4),
        "seconds_per_step_gpu_median": round(gpu, 4),
        "loading_hidden_fraction": round(max(0.0, 1.0 - (wall - gpu) / wall), 4),
        "overhead_seconds": round(wall - gpu, 4),
        "first_steps_wall": [round(value, 3) for value in wall_times[:5]],
        "estimate": {
            "steps_per_epoch": steps_per_epoch,
            "train_hours_per_epoch": round(epoch_hours, 4),
            "dev_hours_per_epoch": round(dev_hours, 4),
            "gpu_hours_70_epochs": round((epoch_hours + dev_hours) * 70, 1),
        },
        "peak_memory_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "gpu": str(torch.cuda.get_device_name(0)),
    }
    output = pathlib.Path("/outputs/loader-benchmark")
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{dataset}-loader-{label}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    results.commit()
    return result


@app.function(
    image=image,
    volumes=VOLUMES,
    gpu="A100-40GB",
    cpu=16.0,
    timeout=60 * 60 * 24,  # per-attempt ceiling; resume continues across attempts
)
def train(
    dataset: str = "phoenix14",
    epochs: int = 70,
    workers: int = 8,
    resume: bool = True,
    seed: int = 0,
) -> dict:
    """Full training run: the paper's protocol, checkpointed and resumable.

    Protocol, all stated by the paper: 70 epochs, batch size 2, Adam with weight
    decay 0.001, initial LR 0.001 reduced to 30% at epochs 25 and 40, random
    224x224 crop from 256x256 with 50% horizontal flip and a 20% probability of
    temporal rescaling, and a central 224x224 crop at inference.

    Checkpoint selection follows the lineage convention recorded as guess
    checkpoint-rule: the epoch with the best dev WER is kept, and the test split
    is scored once at the end from that checkpoint.

    Resumable by design. A 24-hour Modal timeout is shorter than the expected
    24.5 GPU-hours on A100, so this writes a checkpoint every epoch and reloads
    the latest on restart. Re-invoking the function continues rather than
    restarting.
    """
    import json
    import time

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset

    import data
    import evaluate
    import olmd

    device = torch.device("cuda")
    root = pathlib.Path("/datasets")
    cache_root = pathlib.Path(CACHE_ROOT)
    output = pathlib.Path("/outputs/train") / dataset
    output.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    vocabulary = data.build_vocabulary(root, dataset)
    inverse = {index: gloss for gloss, index in vocabulary.items()}
    num_classes = len(vocabulary) + 1

    class TorchDataset(Dataset):
        def __init__(self, inner):
            self.inner = inner

        def __len__(self):
            return len(self.inner)

        def __getitem__(self, index):
            return self.inner[index]

    def make_loader(split: str, train_mode: bool, epoch_seed: int) -> DataLoader:
        inner = data.OLMDDataset(
            root, dataset, split, vocabulary,
            cache_root=cache_root, train=train_mode, seed=epoch_seed,
        )
        return DataLoader(
            TorchDataset(inner),
            batch_size=2,  # the paper's batch size
            shuffle=train_mode,
            num_workers=workers,
            collate_fn=data.collate,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=4,
            drop_last=train_mode,
        )

    model = olmd.OLMD(num_classes=num_classes, context=9).to(device)
    criterion = olmd.OLMDLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[25, 40], gamma=0.3
    )

    latest = output / "latest.pt"
    best = output / "best.pt"
    history_path = output / "history.json"
    start_epoch = 0
    best_dev = float("inf")
    best_epoch = -1
    history = []

    if resume and latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        best_dev = state["best_dev"]
        best_epoch = state["best_epoch"]
        history = state.get("history", [])

    # Reference glosses by utterance name, built once per split rather than
    # per batch: this runs every epoch for 70 epochs.
    references_by_split = {
        split: {
            sample.name: sample.glosses
            for sample in data.LOADERS[dataset](root, split)
        }
        for split in ("dev", "test")
    }

    def score(split: str) -> dict:
        model.eval()
        by_name = references_by_split[split]
        references, hypotheses = [], []
        with torch.no_grad():
            for video, lengths, _, _, names in make_loader(split, False, seed):
                video = video.to(device, non_blocking=True)
                outputs = model(video, lengths)
                decoded = evaluate.greedy_ctc_decode(
                    outputs["global_logits"], outputs["global_lengths"]
                )
                for name, ids in zip(names, decoded):
                    references.append(evaluate.strip_phoenix_markers(by_name[name]))
                    hypotheses.append([inverse[i] for i in ids if i in inverse])
        return evaluate.corpus_wer(references, hypotheses).as_dict()

    started = time.time()
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_started = time.time()
        losses = []
        # Re-seed per epoch so augmentation varies while staying reproducible.
        for video, lengths, targets, target_lengths, _ in make_loader(
            "train", True, seed * 1000 + epoch
        ):
            video = video.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            parts = criterion(model(video, lengths), targets, target_lengths)
            parts["loss"].backward()
            optimizer.step()
            losses.append(float(parts["loss"]))
        scheduler.step()

        dev = score("dev")
        entry = {
            "epoch": epoch,
            "loss_mean": round(float(np.mean(losses)), 4),
            "dev_wer": dev["wer"],
            "dev_del": dev["del"],
            "dev_ins": dev["ins"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - epoch_started, 1),
        }
        history.append(entry)

        # Best-dev checkpoint selection (guess checkpoint-rule).
        if dev["wer"] < best_dev:
            best_dev = dev["wer"]
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "dev": dev}, best)

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_dev": best_dev,
                "best_epoch": best_epoch,
                "history": history,
            },
            latest,
        )
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        results.commit()

    # Final scoring from the best-dev checkpoint, per the checkpoint rule.
    state = torch.load(best, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    final_dev = score("dev")
    final_test = score("test")

    result = {
        "paper_id": PAPER_ID,
        "dataset": dataset,
        "epochs": epochs,
        "seed": seed,
        "num_classes": num_classes,
        "best_epoch": best_epoch,
        "best_dev_wer": best_dev,
        "final": {"dev": final_dev, "test": final_test},
        "published": {"dev": 17.1, "test": 17.2} if dataset == "phoenix14" else None,
        "history": history,
        "wall_seconds": round(time.time() - started, 1),
        "gpu": str(torch.cuda.get_device_name(0)),
        "peak_memory_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 3),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    results.commit()
    return result


@app.function(image=image, volumes=VOLUMES, cpu=4.0, timeout=60 * 60 * 2)
def build_cache_shard(spec: tuple) -> dict:
    """Cache one shard of one split. Runs as one container among many.

    A single container with 16 threads measured only 160 clips/hour, because
    the cost is per-request latency across the ~179 PNG files per utterance and
    threads inside one container cannot hide it (cache-probe measured just
    1.79x from 8 threads). Fanning out across containers multiplies the
    concurrent network paths to the Volume, which threads did not.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    import data

    dataset, split, shard_index, shard_count = spec
    root = pathlib.Path("/datasets")
    cache_root = pathlib.Path(CACHE_ROOT)
    samples = data.LOADERS[dataset](root, split)[shard_index::shard_count]

    started = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        sizes = list(
            pool.map(
                lambda sample: data.write_cache_entry(root, cache_root, dataset, sample),
                samples,
            )
        )
    results.commit()
    elapsed = time.time() - started
    return {
        "split": split,
        "shard": shard_index,
        "clips": len(samples),
        "bytes": int(sum(sizes)),
        "seconds": round(elapsed, 1),
        "seconds_per_clip": round(elapsed / max(len(samples), 1), 2),
    }


@app.function(image=image, volumes=VOLUMES, timeout=60 * 60 * 6)
def build_cache_sharded(
    dataset: str = "phoenix14", shards: int = 40, split: str = "all"
) -> dict:
    """Fan the cache build across containers and report measured scaling."""
    import json
    import time

    splits = ("train", "dev", "test") if split == "all" else (split,)
    specs = [
        (dataset, name, index, shards) for name in splits for index in range(shards)
    ]

    started = time.time()
    shard_results = list(build_cache_shard.map(specs))
    wall = time.time() - started

    total_clips = sum(entry["clips"] for entry in shard_results)
    total_bytes = sum(entry["bytes"] for entry in shard_results)
    per_clip = [entry["seconds_per_clip"] for entry in shard_results if entry["clips"]]
    result = {
        "paper_id": PAPER_ID,
        "dataset": dataset,
        "shards_per_split": shards,
        "containers_launched": len(specs),
        "total_clips": total_clips,
        "total_gib": round(total_bytes / 2**30, 2),
        "wall_seconds": round(wall, 1),
        "wall_hours": round(wall / 3600, 3),
        "effective_clips_per_hour": round(total_clips / (wall / 3600), 1),
        "per_shard_seconds_per_clip": {
            "min": min(per_clip) if per_clip else None,
            "median": sorted(per_clip)[len(per_clip) // 2] if per_clip else None,
            "max": max(per_clip) if per_clip else None,
        },
        "single_container_baseline_clips_per_hour": 160,
        "by_split": {
            name: sum(e["clips"] for e in shard_results if e["split"] == name)
            for name in splits
        },
    }
    output = pathlib.Path("/outputs/cache-build")
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{dataset}-cache-sharded.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    results.commit()
    return result


@app.local_entrypoint()
def main(action: str = "inspect", path: str = "", limit: int = 0) -> None:
    """Print a function's result locally so runs leave readable evidence.

        modal_repro_sign.sh run papers/yu-2025-olmd/modal_app.py --action inspect
        modal_repro_sign.sh run papers/yu-2025-olmd/modal_app.py --action smoke
    """
    import json

    if action == "inspect":
        result = inspect_data.remote()
    elif action == "smoke":
        result = smoke.remote()
    elif action == "diagnose-csl":
        result = diagnose_csl.remote()
    elif action == "preflight":
        result = preflight.remote()
    elif action == "throughput":
        result = throughput.remote()
    elif action == "hash":
        result = hash_output.remote(path)
    elif action == "cache-probe":
        result = test_consolidation.remote()
    elif action == "build-cache":
        result = build_cache.remote(limit=limit)
    elif action == "benchmark-loader":
        result = benchmark_loader.remote(limit=limit)
    elif action == "benchmark-loader-h100":
        result = benchmark_loader_h100.remote(limit=limit)
    elif action == "train":
        result = train.remote()
    elif action == "build-cache-sharded":
        result = build_cache_sharded.remote(shards=limit or 40)
    else:
        raise SystemExit(f"unknown action {action!r}; use 'inspect' or 'smoke'")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
