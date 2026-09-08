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


@app.local_entrypoint()
def main(action: str = "inspect") -> None:
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
    else:
        raise SystemExit(f"unknown action {action!r}; use 'inspect' or 'smoke'")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
