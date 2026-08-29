"""Modal entry points for the conditional 2023 ASL CNN reproduction."""

from __future__ import annotations

import json
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parent
app = modal.App("03785db2-asl-cnn-reproduction")
image = modal.Image.from_registry("tensorflow/tensorflow:2.15.0-gpu").pip_install("pillow==10.2.0", "scipy==1.11.4", "scikit-learn==1.4.2").add_local_file(ROOT / "train.py", "/app/train.py").add_local_file(ROOT / "svm.py", "/app/svm.py")
datasets = modal.Volume.from_name("datasets", create_if_missing=False)
cache = modal.Volume.from_name("huggingface-cache", create_if_missing=False)
results = modal.Volume.from_name("03785db2-asl-cnn-results", create_if_missing=True)
ENV = {"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub", "PYTHONHASHSEED": "2026"}


@app.function(image=image, gpu="L4", cpu=8, timeout=2 * 60 * 60, volumes={"/datasets": datasets, "/cache/huggingface": cache, "/results": results}, env=ENV)
def preflight() -> dict:
    output = Path("/results/preflight-threads")
    if output.exists():
        raise FileExistsError("preflight output exists; retain it rather than overwrite evidence")
    import subprocess

    subprocess.run(["python", "/app/train.py", "--data-root", "/datasets/asl-alphabet", "--output-dir", str(output), "--limit-per-class", "10", "--validation-augmentations", "2"], check=True)
    results.commit()
    return json.loads((output / "run.json").read_text())


@app.function(image=image, gpu="L4", cpu=8, timeout=12 * 60 * 60, volumes={"/datasets": datasets, "/cache/huggingface": cache, "/results": results}, env=ENV)
def train() -> dict:
    output = Path("/results/full-threads")
    if output.exists():
        raise FileExistsError("full output exists; retain it rather than overwrite evidence")
    import subprocess

    subprocess.run(["python", "/app/train.py", "--data-root", "/datasets/asl-alphabet", "--output-dir", str(output)], check=True)
    results.commit()
    return json.loads((output / "run.json").read_text())


@app.function(image=image, gpu="L4", cpu=8, timeout=12 * 60 * 60, volumes={"/datasets": datasets, "/cache/huggingface": cache, "/results": results}, env=ENV)
def svm() -> dict:
    output = Path("/results/svm-full")
    if output.exists():
        raise FileExistsError("full SVM output exists; retain it rather than overwrite evidence")
    import subprocess

    subprocess.run(["python", "/app/svm.py", "--data-root", "/datasets/asl-alphabet", "--weights-root", "/results/full-threads", "--output", str(output), "--limit-per-class", "3000", "--validation-augmentations", "10"], check=True)
    results.commit()
    return json.loads((output / "run.json").read_text())
