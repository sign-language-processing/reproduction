"""Modal entry point for the conditional CNN + Bi-LSTM learning-rate sweep (Sangeetha & Divya Gowri 2025)."""

from __future__ import annotations

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parent.parent
LEARNING_RATES = (1e-2, 1e-3, 1e-4, 1e-5)  # Sec. IV.G, Figures 4-7

app = modal.App("sangeetha-2025-tamil-sign-bilstm")
image = (
    modal.Image.from_dockerfile(REPOSITORY_ROOT / "Dockerfile", context_dir=REPOSITORY_ROOT)
    .pip_install("keras==3.11.3", "opencv-python-headless==4.12.0.88", "scikit-learn==1.7.2")
    .env({"KERAS_BACKEND": "torch"})
    .add_local_file(ROOT / "train.py", "/app/train.py")
)
datasets = modal.Volume.from_name("datasets", create_if_missing=False).read_only()
cache = modal.Volume.from_name("huggingface-cache", create_if_missing=False)
results = modal.Volume.from_name("sangeetha-2025-tamil-sign-bilstm-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16384,
    timeout=2 * 60 * 60,
    volumes={"/datasets": datasets, "/cache/huggingface": cache, "/results": results},
    env={"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub"},
)
def train(lr: float, per_class: int = 300, epochs: int = 30, name: str = "") -> dict:
    import sys

    sys.path.insert(0, "/app")
    import train as reconstruction

    run = reconstruction.main([
        "--data-root", "/datasets/tlfs23",
        "--output-dir", f"/results/{name or f'lr-{lr:g}'}",
        "--lr", str(lr),
        "--per-class", str(per_class),
        "--epochs", str(epochs),
    ])
    results.commit()
    return {key: value for key, value in run.items() if key not in ("predictions", "history")}


@app.local_entrypoint()
def main(per_class: int = 300, epochs: int = 30, prefix: str = ""):
    """Run all four learning rates in parallel and print each function-call ID and summary."""
    calls = {lr: train.spawn(lr, per_class, epochs, f"{prefix}lr-{lr:g}") for lr in LEARNING_RATES}
    for lr, call in calls.items():
        print(f"lr={lr:g} function_call_id={call.object_id}", flush=True)
    for lr, call in calls.items():
        print(json.dumps({"lr": lr, **call.get()}), flush=True)
