"""Conditional SNN+STBP reconstruction for Chen et al. 2023 (Electronics 12, 786), Table 2.

The paper released no code. The LIF neuron and surrogate gradient are imported
unchanged from the pinned thiswinex/STBP-simple checkout, whose `state_update`
is the paper's Algorithm 1. Everything defined in this file (event encoding,
network, loss scaling, optimizer details) is a reconstruction decision that is
documented in README.md; none of it is a paper fact unless a comment cites it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

CLASSES = 15  # Table 1
SIZE = 128  # Section 4.3: network input is 128 x 128


def load_split(root: Path, split: str, step: int, dt: float, limit: int) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Bin `timestamp_s,x,y,polarity` rows into binary frames [N, 2, H, W, step]."""
    frames, labels, names = [], [], []
    for label in range(CLASSES):
        files = sorted((root / split / str(label)).glob("*.csv"), key=lambda path: int(path.stem))
        for path in files[: limit or None]:
            events = np.loadtxt(path, delimiter=",", ndmin=2)
            bins = (events[:, 0] * 1000 / dt).astype(int)
            keep = bins < step  # events after the step * dt simulation window are dropped
            frame = np.zeros((2, SIZE, SIZE, step), dtype=np.uint8)
            frame[events[keep, 3].astype(int), events[keep, 2].astype(int), events[keep, 1].astype(int), bins[keep]] = 1
            frames.append(frame)
            labels.append(label)
            names.append(f"{split}/{label}/{path.name}")
    return torch.from_numpy(np.stack(frames)), torch.tensor(labels), names


def build_model(layers, input_pool: str) -> nn.Module:
    td, spike = layers.tdLayer, layers.LIFSpike()

    class SignNet(nn.Module):
        """Figure 3: pooling32-conv32-conv32-pooling16-conv16-pooling8-fc1(256)-fc2(15)."""

        def __init__(self):
            super().__init__()
            self.features = nn.ModuleList([
                td(nn.MaxPool2d(4) if input_pool == "max" else nn.AvgPool2d(4)),  # 128 -> 32
                td(nn.Conv2d(2, 34, 3, padding=1)),
                td(nn.Conv2d(34, 64, 3, padding=1)),
                td(nn.AvgPool2d(2)),  # 32 -> 16
                td(nn.Conv2d(64, 128, 3, padding=1)),
                td(nn.AvgPool2d(2)),  # 16 -> 8
            ])
            self.fc1 = td(nn.Linear(128 * 8 * 8, 256))
            self.fc2 = td(nn.Linear(256, CLASSES))

        def forward(self, x):
            for layer in self.features:
                x = layer(x)
                if isinstance(layer.layer, nn.Conv2d):
                    x = spike(x)
            x = x.view(x.shape[0], -1, x.shape[4])
            x = spike(self.fc2(spike(self.fc1(x))))
            return x.sum(dim=2) / layers.steps  # firing rate per class

    return SignNet()


def predict(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return torch.cat([model(x[i : i + 50].float()).argmax(1) for i in range(0, len(x), 50)])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None, on_epoch=lambda: None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stbp-dir", type=Path, default=Path("/opt/STBP-simple"))
    parser.add_argument("--optimizer", choices=["sgd", "sgd-momentum", "adam"], default="sgd")
    parser.add_argument("--input-pool", choices=["avg", "max"], default="avg")
    # Table 3, DVS_v2e 77.00% row: step 80, dt 40, Vth 0.3, lr 1e-3.
    parser.add_argument("--step", type=int, default=80)
    parser.add_argument("--dt", type=float, default=40.0, help="milliseconds per step")
    parser.add_argument("--vth", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    # Section 4.3: batch size 20, 200 epochs.
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="files per class and split; 0 = all (preflight only)")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(args.stbp_dir))
    import layers  # pinned upstream STBP-simple

    layers.steps, layers.Vth = args.step, args.vth  # upstream reads these module globals at call time

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_file = args.output_dir / "run.json"
    if run_file.exists() and json.loads(run_file.read_text())["final_epoch"] >= args.epochs:
        return json.loads(run_file.read_text())

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    started = time.time()
    train_x, train_y, _ = load_split(args.data_root, "train", args.step, args.dt, args.limit)
    test_x, test_y, test_names = load_split(args.data_root, "test", args.step, args.dt, args.limit)
    train_x, train_y, test_x, test_y = (t.to(device) for t in (train_x, train_y, test_x, test_y))
    print(f"loaded train={len(train_x)} test={len(test_x)} in {time.time() - started:.0f}s", flush=True)

    model = build_model(layers, args.input_pool).to(device)
    optimizer = {
        "sgd": lambda p: torch.optim.SGD(p, lr=args.lr),
        "sgd-momentum": lambda p: torch.optim.SGD(p, lr=args.lr, momentum=0.9),
        "adam": lambda p: torch.optim.Adam(p, lr=args.lr),
    }[args.optimizer](model.parameters())
    generator = torch.Generator().manual_seed(args.seed)
    state = {"epoch": 0, "best_acc": -1.0, "best_epoch": 0, "seconds": 0.0}

    checkpoint = args.output_dir / "checkpoint.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["generator"].cpu())  # map_location moved it to the GPU
        state = saved["state"]
        print(f"resumed from epoch {state['epoch']}", flush=True)

    target = torch.eye(CLASSES, device=device)
    predictions = predict(model, test_x)
    for epoch in range(state["epoch"], args.epochs):
        epoch_started = time.time()
        lr = args.lr * 0.1 ** (epoch // 60)  # Eq. 8
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        order = torch.randperm(len(train_x), generator=generator).to(device)
        total_loss = 0.0
        for i in range(0, len(order), args.batch_size):
            index = order[i : i + args.batch_size]
            output = model(train_x[index].float())
            loss = 0.5 * ((output - target[train_y[index]]) ** 2).sum(1).mean()  # Eq. 6
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(index)
        predictions = predict(model, test_x)
        accuracy = 100 * (predictions == test_y).float().mean().item()
        state["epoch"] = epoch + 1
        state["seconds"] += time.time() - epoch_started
        if accuracy > state["best_acc"]:
            state["best_acc"], state["best_epoch"] = accuracy, epoch + 1
        record = {"epoch": epoch + 1, "lr": lr, "train_loss": total_loss / len(train_x), "test_acc": accuracy}
        with (args.output_dir / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "generator": generator.get_state(), "state": state},
            checkpoint,
        )
        on_epoch()

    correct = (predictions == test_y).cpu()
    # The paper never defines Acc1/Acc2 ("first/second part of the test set").
    # File-index halves are one reading, reported as conditional evidence only.
    first_half = torch.tensor([int(Path(name).stem) < 5 for name in test_names])
    (args.output_dir / "freeze.txt").write_text(subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout)
    run = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "train_count": len(train_x),
        "test_count": len(test_x),
        "final_epoch": state["epoch"],
        "final_test_acc": 100 * correct.float().mean().item(),
        "final_test_acc_files_0_4": 100 * correct[first_half].float().mean().item(),
        "final_test_acc_files_5_9": 100 * correct[~first_half].float().mean().item(),
        "best_test_acc_diagnostic": state["best_acc"],
        "best_epoch_diagnostic": state["best_epoch"],
        "predictions": dict(zip(test_names, predictions.tolist())),
        "train_seconds": state["seconds"],
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "stbp_simple_commit": subprocess.run(["git", "-C", str(args.stbp_dir), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "train_py_sha256": sha256(Path(__file__)),
        "dataset_manifest_sha256": sha256(args.data_root / "MANIFEST.sha256"),
    }
    run_file.write_text(json.dumps(run, indent=2) + "\n")
    return run


if __name__ == "__main__":
    main()
