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
    parser.add_argument("--arm", choices=["as-is", "ctc-without-cudnn", "compare-grads"], required=True)
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
    if ours.arm == "compare-grads":
        with open(ours.output, "w") as f:
            json.dump(compare_grads(processor), f, indent=2)
        return

    stats = {"finite_loss_batches": 0, "nonfinite_loss_batches": 0, "applied_steps": 0, "nonfinite_grad_applied_steps": 0}
    losses = []
    criterion = model.criterion_calculation

    def counted_criterion(*a, **k):
        if ours.arm == "ctc-without-cudnn":
            with torch.backends.cudnn.flags(enabled=False):
                out = criterion(*a, **k)
        else:
            out = criterion(*a, **k)
        loss = out[0] if isinstance(out, tuple) else out
        finite = bool(torch.isfinite(loss).all())
        stats["finite_loss_batches" if finite else "nonfinite_loss_batches"] += 1
        if finite:
            losses.append(float(loss))
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
    seq_train(loader, model, optimizer, processor.device, 0, processor.recoder, **kwargs)  # CorrNet's returns None
    stats.update(arm=ours.arm, batches=len(loader),
                 skipped_by_grad_scaler=stats["finite_loss_batches"] - stats["applied_steps"],
                 mean_loss=float(np.mean(losses)) if len(losses) else None,
                 torch=torch.__version__, cudnn=torch.backends.cudnn.version())
    with open(ours.output, "w") as f:
        json.dump(stats, f, indent=2)


def compare_grads(processor, batches=5):
    """CTC loss and its gradient w.r.t. the sequence logits on fixed real batches:
    the repo's own call with cuDNN allowed, the same call with cuDNN disabled,
    and a float64 CPU reference. Logits come from one forward pass per batch."""
    model, device = processor.model, processor.device
    ctc = torch.nn.CTCLoss(reduction="none", zero_infinity=False)
    model.eval()
    results = []
    for index, data in enumerate(processor.data_loader["train"]):
        if index == batches:
            break
        vid, vid_lgt = device.data_to_device(data[0]), device.data_to_device(data[1])
        label, label_lgt = device.data_to_device(data[2]), device.data_to_device(data[3])
        with torch.no_grad():
            ret = model(vid, vid_lgt, label=label, label_lgt=label_lgt)
        logits, feat_len = ret["sequence_logits"].detach().float(), ret["feat_len"]
        out = {"feat_len": [int(x) for x in feat_len], "label_len": [int(x) for x in label_lgt]}
        grads = {}
        for mode in ("cudnn_allowed", "cudnn_disabled", "cpu_float64"):
            x = (logits.double().cpu() if mode == "cpu_float64" else logits.clone()).requires_grad_(True)
            args = (x.log_softmax(-1), label.cpu().int(), feat_len.cpu().int(), label_lgt.cpu().int())
            with torch.backends.cudnn.flags(enabled=mode == "cudnn_allowed"):
                loss = ctc(*args).mean()
                loss.backward()
            out[f"loss_{mode}"] = float(loss)
            grads[mode] = x.grad.double().cpu().flatten()
        reference = grads["cpu_float64"]
        for mode in ("cudnn_allowed", "cudnn_disabled"):
            diff = (grads[mode] - reference).abs().max()
            cosine = torch.nn.functional.cosine_similarity(grads[mode], reference, dim=0)
            out[f"{mode}_vs_cpu"] = {"max_abs_diff": float(diff), "cosine": float(cosine),
                                     "reference_max_abs": float(reference.abs().max())}
        results.append(out)
    return {"arm": "compare-grads", "batches": results, "torch": torch.__version__,
            "cudnn": torch.backends.cudnn.version()}


if __name__ == "__main__":
    main()
