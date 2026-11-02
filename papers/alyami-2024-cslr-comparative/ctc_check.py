"""Preflight diagnostic: does nn.CTCLoss on cuDNN give non-finite gradients here?

Hypothesis carried over from papers/yu-2025-cvsign: on the current PyTorch and
cuDNN, the repos' CTC call (CPU int32 targets) is routed to cuDNN's kernel,
which returns non-finite gradients, so AMP's GradScaler skips most updates.

Run with cwd = a prepared per-run copy of one upstream repo. It builds that
repo's own Processor from configs/run.yaml exactly as main.py's __main__ does
and runs the repo's own seq_train over the first training batches. The arms
differ only by a cudnn.flags(enabled=False) context around
criterion_calculation. Nothing here is used by a training run.
"""

import argparse
import inspect
import json
import sys

import numpy as np
import torch
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["as-is", "ctc-without-cudnn"], required=True)
    parser.add_argument("--batches", type=int, default=60)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output", required=True)
    ours = parser.parse_args()

    sys.path.insert(0, ".")
    sys.argv = ["main.py", "--config", "./configs/run.yaml", "--device", "0", "--work-dir", ours.work_dir]
    import main as upstream  # module body only defines Processor and helpers
    import utils
    from seq_scripts import seq_train

    sparser = utils.get_parser()
    with open(sparser.parse_args().config) as f:
        sparser.set_defaults(**yaml.load(f, Loader=yaml.FullLoader))
    args = sparser.parse_args()
    with open(f"./configs/{args.dataset}.yaml") as f:
        args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)
    processor = upstream.Processor(args)
    model, optimizer = processor.model, processor.optimizer

    stats = {"finite_loss_batches": 0, "nonfinite_loss_batches": 0, "applied_steps": 0, "nonfinite_grad_applied_steps": 0}
    criterion = model.criterion_calculation

    def counted_criterion(*a, **k):
        if ours.arm == "ctc-without-cudnn":
            with torch.backends.cudnn.flags(enabled=False):
                out = criterion(*a, **k)
        else:
            out = criterion(*a, **k)
        loss = out[0] if isinstance(out, tuple) else out
        stats["finite_loss_batches" if bool(torch.isfinite(loss).all()) else "nonfinite_loss_batches"] += 1
        return out

    model.criterion_calculation = counted_criterion
    inner = optimizer.optimizer
    step = inner.step

    def counted_step(*a, **k):  # under AMP, GradScaler calls this only when the grads were finite
        stats["applied_steps"] += 1
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
               for group in inner.param_groups for p in group["params"]):
            stats["nonfinite_grad_applied_steps"] += 1
        return step(*a, **k)

    inner.step = counted_step
    dataset = processor.data_loader["train"].dataset
    subset = torch.utils.data.Subset(dataset, range(min(len(dataset), 2 * ours.batches)))
    loader = processor.build_dataloader(subset, "train", True)
    kwargs = {"loss_weights": args.loss_weights} if "loss_weights" in inspect.signature(seq_train).parameters else {}
    losses = seq_train(loader, model, optimizer, processor.device, 0, processor.recoder, **kwargs)
    stats.update(arm=ours.arm, batches=len(loader),
                 skipped_by_grad_scaler=stats["finite_loss_batches"] - stats["applied_steps"],
                 mean_loss=float(np.mean(losses)) if len(losses) else None,
                 torch=torch.__version__, cudnn=torch.backends.cudnn.version())
    with open(ours.output, "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
