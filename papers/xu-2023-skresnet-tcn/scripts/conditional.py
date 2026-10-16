"""Declared conditional reconstruction; never presented as the missing author model."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def prepare_video(job):
    import cv2
    from simple_video_utils.frames import read_frames_exact

    source, destination, relative = job
    path = Path(destination)
    if path.exists():
        a = np.load(path)
        return dict(
            source=relative,
            source_sha256=sha(source),
            file=path.name,
            sha256=sha(path),
            indices=a["indices"].tolist(),
            frames=int(a["frame_count"]),
        )
    resized, differences = [], []
    previous = None
    for frame in read_frames_exact(source):
        small = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
        differences.append(
            0.0 if previous is None else float(np.abs(gray - previous).mean())
        )
        resized.append(small)
        previous = gray
    assert resized, source
    smooth = np.convolve(
        np.pad(differences, (1, 1), mode="edge"), np.ones(3) / 3, mode="valid"
    )
    peaks = (
        np.flatnonzero((smooth[1:-1] > smooth[:-2]) & (smooth[1:-1] >= smooth[2:])) + 1
    )
    # Rank local maxima by smoothed motion, preserve temporal order; fill shortages uniformly.
    selected = set(peaks[np.argsort(-smooth[peaks], kind="stable")[:32]].tolist())
    for index in (
        np.linspace(0, len(resized) - 1, min(32, len(resized))).round().astype(int)
    ):
        if len(selected) < 32:
            selected.add(int(index))
    for index in range(len(resized)):
        if len(selected) < 32:
            selected.add(index)
    indices = sorted(selected)
    indices += [indices[-1]] * (32 - len(indices))
    with open(str(path) + ".tmp", "wb") as f:
        np.savez_compressed(
            f,
            frames=np.stack([resized[i] for i in indices]),
            indices=np.array(indices),
            frame_count=len(resized),
        )
    os.replace(str(path) + ".tmp", path)
    return dict(
        source=relative,
        source_sha256=sha(source),
        file=path.name,
        sha256=sha(path),
        indices=indices,
        frames=len(resized),
    )


def prepare(root, out):
    from concurrent.futures import ProcessPoolExecutor
    import tarfile

    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists() and (out / "frames.tar").exists():
        m = json.loads((out / "manifest.json").read_text())
        assert sha(out / "frames.tar") == m["archive_sha256"]
        print("Already verified prepared data", flush=True)
        return
    files = sorted(root.rglob("*.mp4"))
    assert len(files) == 3200
    jobs = [
        (str(p), str(out / (p.stem + ".npz")), str(p.relative_to(root))) for p in files
    ]
    with ProcessPoolExecutor(max_workers=12) as pool:
        records = []
        for index, record in enumerate(pool.map(prepare_video, jobs)):
            records.append(record)
            if index % 100 == 0:
                print("Prepared", index + 1, "/3200", flush=True)
    rng = np.random.default_rng(42)
    splits = {"train": [], "validation": [], "test": []}
    for label in range(1, 65):
        ids = [
            i
            for i, r in enumerate(records)
            if int(Path(r["file"]).stem.split("_")[0]) == label
        ]
        assert len(ids) == 50
        ids = rng.permutation(ids).tolist()
        splits["train"] += ids[:30]
        splits["validation"] += ids[30:40]
        splits["test"] += ids[40:]
    with tarfile.open(out / "frames.tar", "w") as tar:
        for r in records:
            tar.add(out / r["file"], arcname=r["file"])
    m = dict(
        protocol="conditional-v1",
        source_manifest_sha256=sha(root / "manifest.json"),
        seed=42,
        frames_per_video=32,
        spatial_size=224,
        records=records,
        splits=splits,
        archive_sha256=sha(out / "frames.tar"),
    )
    (out / "manifest.json").write_text(json.dumps(m, indent=2))
    print("Prepared all3200", {k: len(v) for k, v in splits.items()}, flush=True)


def train(root, out, preflight, paper_sized=False):
    import shutil
    import subprocess
    import tarfile
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    import timm
    import torch_optimizer
    from fvcore.nn import FlopCountAnalysis
    import sys

    sys.path.insert(0, "/opt/TCN/TCN")
    from tcn import TemporalBlock

    torch.set_num_threads(8)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists():
        print((out / "metrics.json").read_text())
        return
    started = time.time()
    manifest = json.loads((root / "manifest.json").read_text())
    if paper_sized:
        assert sha(root / "manifest.json") == "d04e5c9b682f00eaa21cbfaa9df6aa62f7a7f45fcda788aca820acb3ad5b6e92"
    local = Path("/tmp/lsa64-conditional")
    local.mkdir(exist_ok=True)
    archive = local / "frames.tar"
    shutil.copyfile(root / "frames.tar", archive)
    assert sha(archive) == manifest["archive_sha256"]
    with tarfile.open(archive) as tar:
        tar.extractall(local, filter="data")
    for r in manifest["records"]:
        assert sha(local / r["file"]) == r["sha256"]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    class Videos(Dataset):
        def __init__(self, split, augment=False):
            self.ids = manifest["splits"][split]
            self.augment = augment
            if preflight:
                per_class = 30 if split == "train" else 10
                self.ids = [
                    self.ids[c * per_class + j]
                    for c in range(64)
                    for j in range(2 if split == "train" else 1)
                ]

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, i):
            index = self.ids[i]
            r = manifest["records"][index]
            with np.load(local / r["file"]) as a:
                x = (
                    torch.from_numpy(a["frames"].copy()).permute(0, 3, 1, 2).float()
                    / 255
                )
            if self.augment and torch.rand(()) < 0.5:
                x = x.flip(-1)
            x = (
                x - torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
            ) / torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
            return x, int(Path(r["file"]).stem.split("_")[0]) - 1, index

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.spatial = timm.create_model(
                "skresnext50_32x4d",
                pretrained=False,
                num_classes=0,
                global_pool="max",
                act_layer=nn.Mish,
                **({"layers": [1, 2, 2, 1]} if paper_sized else {}),
            )
            blocks = []
            temporal_width = 224 if paper_sized else 256
            width = 2048
            for dilation in [1, 2, 5]:
                block = TemporalBlock(
                    width, temporal_width, 3, 1, dilation, 2 * dilation, dropout=0.2
                )
                # The paper explicitly replaces ReLU by Mish; use the published block otherwise.
                block.relu1 = nn.Mish()
                block.relu2 = nn.Mish()
                block.relu = nn.Mish()
                block.net = nn.Sequential(
                    block.conv1,
                    block.chomp1,
                    block.relu1,
                    block.dropout1,
                    block.conv2,
                    block.chomp2,
                    block.relu2,
                    block.dropout2,
                )
                blocks.append(block)
                width = temporal_width
            self.temporal = nn.Sequential(*blocks)
            self.classifier = nn.Linear(temporal_width, 64)

        def forward(self, x):
            b, t, c, h, w = x.shape
            z = (
                self.spatial(x.reshape(b * t, c, h, w))
                .reshape(b, t, -1)
                .transpose(1, 2)
            )
            return self.classifier(self.temporal(z).amax(-1))

    model = Model().cuda()
    optimizer = torch_optimizer.Ranger(model.parameters(), lr=0.0001)
    parameters = sum(p.numel() for p in model.parameters())
    model.eval()
    with torch.no_grad():
        flops = FlopCountAnalysis(model, torch.zeros(1, 32, 3, 224, 224, device="cuda"))
        flop_count = flops.total()
        unsupported = dict(flops.unsupported_ops())
    (out / "model.txt").write_text(str(model))
    (out / "freeze.txt").write_text(
        subprocess.check_output(["python", "-m", "pip", "freeze"], text=True)
    )
    (out / "gpu.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    generator = torch.Generator().manual_seed(42)
    # Four videos per microbatch =>128 frames for BatchNorm. Effective optimizer batch128.
    loader = DataLoader(
        Videos("train", True),
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        generator=generator,
    )

    def evaluate(split, augmented=False):
        model.eval()
        truth = []
        prediction = []
        indices = []
        for x, y, index in DataLoader(
            Videos(split, augmented), batch_size=4, num_workers=4, pin_memory=True
        ):
            with torch.no_grad():
                logits = model(x.cuda(non_blocking=True))
            truth.extend(y.tolist())
            prediction.extend(logits.argmax(1).cpu().tolist())
            indices.extend(index.tolist())
        return dict(
            accuracy=float(np.mean(np.array(truth) == prediction)),
            truth=truth,
            prediction=prediction,
            indices=indices,
        )

    step = 0
    best = -1.0
    best_state = None
    history = []
    microbatch_times = []
    checkpoint = out / "last.pt"
    if checkpoint.exists():
        state = torch.load(checkpoint, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]
        best = state["best"]
        history = state["history"]
        best_state = state["best_model"]
        torch.save(best_state, out / "best.tmp")
        os.replace(out / "best.tmp", out / "best.pt")
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        generator.set_state(state["loader_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])

    def save():
        torch.save(
            dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                step=step,
                best=best,
                best_model=best_state,
                history=history,
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(),
                loader_rng=generator.get_state(),
                python_rng=random.getstate(),
                numpy_rng=np.random.get_state(),
            ),
            out / "last.tmp",
        )
        os.replace(out / "last.tmp", checkpoint)

    iterator = iter(loader)
    maximum = 7 if preflight else 1000
    training_started = time.time()
    while step < maximum:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0
        for _ in range(32):
            try:
                x, y, _ = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                x, y, _ = next(iterator)
            tick = time.time()
            logits = model(x.cuda(non_blocking=True))
            loss = nn.functional.cross_entropy(logits, y.cuda())
            (loss / 32).backward()
            loss_total += loss.item()
            torch.cuda.synchronize()
            microbatch_times.append(time.time() - tick)
        optimizer.step()
        step += 1
        if step % 10 == 0 or preflight:
            print(
                json.dumps(
                    dict(
                        step=step,
                        loss=loss_total / 32,
                        elapsed=time.time() - training_started,
                    )
                ),
                flush=True,
            )
        if step % 50 == 0 or step == maximum:
            val = evaluate("validation")
            history.append(dict(step=step, validation_accuracy=val["accuracy"]))
            if val["accuracy"] > best:
                best = val["accuracy"]
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                torch.save(best_state, out / "best.tmp")
                os.replace(out / "best.tmp", out / "best.pt")
            print("validation", step, val["accuracy"], flush=True)
        # 60steps is four complete1920-example epochs, so resume starts at the exact next permutation.
        if step % 60 == 0 or step == maximum:
            save()
            print("checkpoint", step, flush=True)
    # Verify full state reload and perform one actual resumed step only in disposable preflight.
    state = torch.load(checkpoint, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if preflight:
        # A fresh instance proves restoration independently of the objects just trained.
        del model, optimizer
        torch.cuda.empty_cache()
        model = Model().cuda()
        optimizer = torch_optimizer.Ranger(model.parameters(), lr=0.0001)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        # Verify every model, Ranger and Lookahead tensor before a real resumed update.
        def equal(a, b):
            if isinstance(a, torch.Tensor):
                return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
            if isinstance(a, dict):
                return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
            if isinstance(a, (list, tuple)):
                return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
            return a == b
        assert equal(model.state_dict(), state["model"])
        assert equal(optimizer.state_dict(), state["optimizer"])
        model.train()
        x, y, _ = next(iter(loader))
        optimizer.zero_grad()
        nn.functional.cross_entropy(model(x.cuda()), y.cuda()).backward()
        optimizer.step()
    model.load_state_dict(torch.load(out / "best.pt", weights_only=True))
    torch.manual_seed(314159)
    primary = evaluate("test", True)
    torch.manual_seed(314159)
    secondary = evaluate("test", False)
    np.savez_compressed(
        out / "predictions.npz",
        truth=primary["truth"],
        prediction=primary["prediction"],
        indices=primary["indices"],
    )
    metrics = dict(
        scope=("parameter-sized reconstruction authorized before accuracy; parameter calibration is not independent reproduction evidence" if paper_sized else "conditional reconstruction, not comparable to the unspecified author architecture"),
        paper_sized=paper_sized,
        accuracy=primary["accuracy"],
        test_count=len(primary["truth"]),
        deterministic_test_accuracy=secondary["accuracy"],
        parameters=parameters,
        flops_one_fused_multiply_add=flop_count,
        unsupported_flop_ops=unsupported,
        steps=step,
        best_validation_accuracy=best,
        history=history,
        checkpoint_reload_verified=True,
        optimizer_resume_verified=preflight,
        fresh_instance_checkpoint_restore_verified=preflight,
        wall_seconds=time.time() - started,
        training_seconds=time.time() - training_started,
        mean_warm_microbatch_seconds=float(np.mean(microbatch_times[1:])),
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        script_sha256=sha(__file__),
        manifest_sha256=sha(root / "manifest.json"),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
    )
    metrics["files"] = {
        p.name: sha(p)
        for p in out.iterdir()
        if p.is_file() and p.name != "metrics.json"
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["prepare", "preflight", "full"])
    p.add_argument("--output", required=True)
    p.add_argument("--paper-sized", action="store_true")
    a = p.parse_args()
    if a.mode == "prepare":
        prepare(Path("/datasets/lsa64"), Path(a.output))
    else:
        train(
            Path("/datasets/lsa64-skresnet-conditional-v1"),
            Path(a.output),
            a.mode == "preflight",
            a.paper_sized,
        )
