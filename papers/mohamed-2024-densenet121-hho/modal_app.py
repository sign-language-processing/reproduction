"""Modal entry points for the clean-room ArASL2018 Table II reproduction."""

from __future__ import annotations

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
RESULTS_VOLUME = "mohamed-2024-densenet121-hho-results"

app = modal.App("mohamed-2024-densenet121-hho")
# The paper's stack is Keras/TensorFlow, unlike the root study image's
# PyTorch/NGC stack, so this paper builds its own image (see Dockerfile).
image = modal.Image.from_dockerfile(ROOT / "Dockerfile", context_dir=ROOT).add_local_file(ROOT / "train.py", "/app/train.py")
datasets = modal.Volume.from_name("datasets", create_if_missing=False)
cache = modal.Volume.from_name("huggingface-cache", create_if_missing=False)
results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)

CACHE_ENV = {"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub", "KERAS_HOME": "/cache/huggingface/keras"}


@app.function(
    image=image,
    gpu="A10G",
    cpu=4,
    timeout=60 * 60,
    volumes={"/datasets": datasets, "/cache/huggingface": cache, "/results": results},
    env=CACHE_ENV,
)
def preflight(model: str, run_name: str) -> str:
    """Representative preflight: real data/weights, a few steps, checkpoint
    save/reload, tiny-subset evaluation. Disposable; not a retained run."""
    import subprocess

    output_dir = Path(f"/results/preflight/{run_name}")
    subprocess.run(
        [
            "python", "/app/train.py",
            "--model", model,
            "--data-root", "/datasets/arasl-database-grayscale",
            "--output-dir", str(output_dir),
            "--preflight",
        ],
        check=True,
    )
    results.commit()
    return (output_dir / "run.json").read_text(encoding="utf-8")


@app.function(
    image=image,
    gpu="A10G",
    cpu=8,
    timeout=24 * 60 * 60,
    volumes={"/datasets": datasets, "/cache/huggingface": cache, "/results": results},
    env=CACHE_ENV,
)
def train(model: str, run_name: str) -> str:
    """Run the paper's Table II recipe for one model once, into /results/<run_name>."""
    import subprocess

    output_dir = Path(f"/results/{run_name}")
    if (output_dir / "run.json").exists():
        return (output_dir / "run.json").read_text(encoding="utf-8")
    subprocess.run(
        [
            "python", "/app/train.py",
            "--model", model,
            "--data-root", "/datasets/arasl-database-grayscale",
            "--output-dir", str(output_dir),
        ],
        check=True,
    )
    results.commit()
    return (output_dir / "run.json").read_text(encoding="utf-8")


@app.local_entrypoint()
def main(model: str, preflight_only: bool = False, run_name: str = "") -> None:
    # Each run keeps its own directory: train() returns an existing run.json
    # instead of retraining, and train.py resumes from an existing state.json,
    # so a new attempt needs a new run_name.
    if preflight_only:
        print(preflight.remote(model, run_name or model))
    else:
        print(train.remote(model, run_name or model))
