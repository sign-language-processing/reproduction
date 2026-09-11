"""OLMD full training outside Modal, for CINECA (or any SLURM cluster).

Same protocol as modal_app.py's train(): 70 epochs, batch size 2, Adam with
weight decay 0.001, initial LR 0.001 cut to 30% at epochs 25 and 40, the
paper's augmentation, best-dev checkpoint selection, and a checkpoint every
epoch so a SLURM walltime limit only pauses the run -- resubmitting resumes
from latest.pt. Scoring strips PHOENIX markers from BOTH sides (symmetric).

This branch's model: LMA on a side branch, the Figure-2 head (trunk and X_CS
through ONE weight-shared 1D-CNN branch, outputs summed -> BiLSTM), and the
parameter-budget MP-Block (dense (3x1)+(1x3) pair).

Offline by design: compute nodes need no internet. Before submitting, stage
(1) the annotation CSVs, (2) the complete JPEG frame cache, and (3) the
torchvision ResNet34 weights into TORCH_HOME (see README). The script fails
fast if any of the three is missing.

Usage:
    python train_cineca.py --data-root DATA --cache-root CACHE --output OUT \
        [--epochs 70] [--workers 8] [--seed 0] [--smoke]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import data
import evaluate
import olmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=pathlib.Path,
                        help="Directory containing rwth-phoenix-2014/... annotations")
    parser.add_argument("--cache-root", required=True, type=pathlib.Path,
                        help="Directory containing phoenix14/*.npz JPEG frame cache")
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--dataset", default="phoenix14")
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh even if latest.pt exists")
    parser.add_argument("--smoke", action="store_true",
                        help="2 training steps + 4-clip dev eval, then exit 0")
    return parser.parse_args()


class TorchDataset(Dataset):
    def __init__(self, inner):
        self.inner = inner

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, index):
        return self.inner[index]


def main() -> int:
    args = parse_args()
    device = torch.device("cuda")
    args.output.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- fail fast on missing offline prerequisites -------------------------
    samples_by_split = {
        split: data.LOADERS[args.dataset](args.data_root, split)
        for split in ("train", "dev", "test")
    }
    missing = [
        sample.name
        for split, samples in samples_by_split.items()
        for sample in samples
        if not data.cache_path(args.cache_root, args.dataset, sample).exists()
    ]
    if missing:
        print(f"FATAL: {len(missing)} utterances missing from the cache, "
              f"e.g. {missing[:3]}; stage the full cache first.", file=sys.stderr)
        return 2

    vocabulary = data.build_vocabulary(args.data_root, args.dataset)
    inverse = {index: gloss for gloss, index in vocabulary.items()}
    num_classes = len(vocabulary) + 1

    model = olmd.OLMD(num_classes=num_classes, context=9).to(device)
    criterion = olmd.OLMDLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[25, 40], gamma=0.3
    )

    def make_loader(split: str, train_mode: bool, epoch_seed: int) -> DataLoader:
        inner = data.OLMDDataset(
            args.data_root, args.dataset, split, vocabulary,
            cache_root=args.cache_root, train=train_mode, seed=epoch_seed,
        )
        return DataLoader(
            TorchDataset(inner),
            batch_size=2,  # the paper's batch size
            shuffle=train_mode,
            num_workers=args.workers,
            collate_fn=data.collate,
            pin_memory=True,
            prefetch_factor=4 if args.workers else None,
            drop_last=train_mode,
        )

    references_by_split = {
        split: {sample.name: sample.glosses for sample in samples}
        for split, samples in samples_by_split.items()
    }

    def score(split: str, limit: "int | None" = None) -> dict:
        model.eval()
        by_name = references_by_split[split]
        references, hypotheses = [], []
        with torch.no_grad():
            for batch_index, (video, lengths, _, _, names) in enumerate(
                make_loader(split, False, args.seed)
            ):
                if limit is not None and batch_index * 2 >= limit:
                    break
                video = video.to(device, non_blocking=True)
                outputs = model(video, lengths)
                decoded = evaluate.greedy_ctc_decode(
                    outputs["global_logits"], outputs["global_lengths"]
                )
                for name, ids in zip(names, decoded):
                    references.append(evaluate.strip_phoenix_markers(by_name[name]))
                    hypotheses.append(
                        evaluate.strip_phoenix_markers(
                            [inverse[i] for i in ids if i in inverse]
                        )
                    )
        return evaluate.corpus_wer(references, hypotheses).as_dict()

    if args.smoke:
        model.train()
        loader = make_loader("train", True, args.seed)
        for step, (video, lengths, targets, target_lengths, _) in enumerate(loader):
            video = video.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            parts = criterion(model(video, lengths), targets, target_lengths)
            parts["loss"].backward()
            optimizer.step()
            print(f"smoke step {step}: loss {float(parts['loss']):.3f} "
                  f"ctc {float(parts['ctc']):.3f}")
            if step >= 1:
                break
        print("smoke dev:", json.dumps(score("dev", limit=4)))
        print("smoke OK")
        return 0

    latest = args.output / "latest.pt"
    best = args.output / "best.pt"
    history_path = args.output / "history.json"
    start_epoch = 0
    best_dev = float("inf")
    best_epoch = -1
    history = []

    if not args.no_resume and latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        best_dev = state["best_dev"]
        best_epoch = state["best_epoch"]
        history = state.get("history", [])
        print(f"resumed from epoch {state['epoch']} (best dev {best_dev})")

    started = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_started = time.time()
        losses, parts_ctc, parts_distill, parts_lift = [], [], [], []
        for video, lengths, targets, target_lengths, _ in make_loader(
            "train", True, args.seed * 1000 + epoch
        ):
            video = video.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            parts = criterion(model(video, lengths), targets, target_lengths)
            parts["loss"].backward()
            optimizer.step()
            losses.append(float(parts["loss"]))
            parts_ctc.append(float(parts["ctc"]))
            parts_distill.append(float(parts["distill"]))
            parts_lift.append(float(parts.get("lift", 0.0)))
        scheduler.step()

        dev = score("dev")
        entry = {
            "epoch": epoch,
            "loss_mean": round(float(np.mean(losses)), 4),
            "ctc_mean": round(float(np.mean(parts_ctc)), 4),
            "distill_mean": round(float(np.mean(parts_distill)), 6),
            "distill_weighted": round(25.0 * float(np.mean(parts_distill)), 4),
            "lift_mean": round(float(np.mean(parts_lift)), 2),
            "dev_wer": dev["wer"],
            "dev_del": dev["del"],
            "dev_ins": dev["ins"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - epoch_started, 1),
        }
        history.append(entry)
        print(json.dumps(entry), flush=True)

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

    # Final scoring from the best-dev checkpoint, per the checkpoint rule.
    state = torch.load(best, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    final_dev = score("dev")
    final_test = score("test")

    result = {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "seed": args.seed,
        "num_classes": num_classes,
        "best_epoch": best_epoch,
        "best_dev_wer": best_dev,
        "final": {"dev": final_dev, "test": final_test},
        "published": {"dev": 17.1, "test": 17.2} if args.dataset == "phoenix14" else None,
        "history": history,
        "wall_seconds": round(time.time() - started, 1),
        "gpu": str(torch.cuda.get_device_name(0)),
        "peak_memory_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "wiring": "params-conformant (figure2 wiring + weight-shared temporal branch + dense factorized MP-Block per the Table 5 parameter budget)",
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["final"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
