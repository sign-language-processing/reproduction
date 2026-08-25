"""Approved ISL-HS dataset population for the landmark reproduction."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parent.parent
app = modal.App("8526aecd1407305d-isl-hs-data")
base_image = modal.Image.from_dockerfile(
    REPOSITORY_ROOT / "Dockerfile", context_dir=REPOSITORY_ROOT
)
image = base_image.add_local_file(ROOT / "data.sh", "/app/data.sh")
asl_image = base_image.add_local_file(ROOT / "asl_data.sh", "/app/asl_data.sh")
evaluation_image = (
    base_image.pip_install("mediapipe==0.10.18", "scikit-learn==1.6.1")
    .add_local_file(ROOT / "evaluate.py", "/app/evaluate.py")
    .add_local_file(ROOT / "evaluate_asl.py", "/app/evaluate_asl.py")
)
datasets = modal.Volume.from_name("datasets", create_if_missing=False)
cache = modal.Volume.from_name("huggingface-cache", create_if_missing=False)
results = modal.Volume.from_name("8526aecd-landmark-results", create_if_missing=True)


@app.function(
    image=image,
    cpu=2,
    timeout=60 * 60,
    volumes={"/datasets": datasets, "/cache/huggingface": cache},
    env={"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub"},
)
def populate_isl_hs() -> dict[str, object]:
    """Populate and commit the user-authorized, pinned ISL-HS source once."""
    completed = subprocess.run(
        ["bash", "/app/data.sh"], check=True, capture_output=True, text=True
    )
    datasets.commit()
    return json.loads(completed.stdout)


@app.function(
    image=asl_image,
    cpu=2,
    timeout=2 * 60 * 60,
    volumes={"/datasets": datasets, "/cache/huggingface": cache},
    env={"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub"},
)
def populate_asl_alphabet() -> dict[str, object]:
    """Populate the user-selected A-Z, SPACE, DELETE ASL subset once."""
    completed = subprocess.run(
        ["bash", "/app/asl_data.sh"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if completed.returncode:
        raise RuntimeError(completed.stdout)
    datasets.commit()
    return json.loads(Path("/datasets/asl-alphabet/manifest.json").read_text(encoding="utf-8"))


@app.function(
    image=evaluation_image,
    cpu=8,
    timeout=2 * 60 * 60,
    volumes={"/datasets": datasets, "/cache/huggingface": cache, "/results": results},
    env={"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub"},
)
def preflight() -> dict[str, object]:
    """Exercise real decoding, landmark extraction, reduction, and both CV paths."""
    output = Path("/results/preflight-frame-shuffled")
    if output.exists():
        raise FileExistsError("preflight output exists; retain it as evidence rather than overwrite it")
    subprocess.run(["python", "/app/evaluate.py", "--data-root", "/datasets/isl-hs", "--output-dir", str(output), "--videos-per-class", "2", "--folds", "2"], check=True)
    results.commit()
    return json.loads((output / "run.json").read_text(encoding="utf-8"))


@app.function(
    image=evaluation_image,
    cpu=8,
    timeout=8 * 60 * 60,
    volumes={"/datasets": datasets, "/cache/huggingface": cache, "/results": results},
    env={"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub"},
)
def evaluate_isl_hs() -> dict[str, object]:
    """Run the documented conditional ISL-HS protocols once, with no selection."""
    output = Path("/results/isl-hs-conditional")
    if output.exists():
        raise FileExistsError("full evaluation output exists; retain it as evidence rather than overwrite it")
    subprocess.run(["python", "/app/evaluate.py", "--data-root", "/datasets/isl-hs", "--output-dir", str(output)], check=True)
    results.commit()
    return json.loads((output / "run.json").read_text(encoding="utf-8"))


@app.function(
    image=evaluation_image,
    cpu=8,
    timeout=2 * 60 * 60,
    volumes={"/datasets": datasets, "/cache/huggingface": cache, "/results": results},
    env={"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub"},
)
def preflight_asl_alphabet() -> dict[str, object]:
    """Exercise image loading, landmarks, reduction, and classifier fitting."""
    output = Path("/results/asl-alphabet-preflight")
    if output.exists():
        raise FileExistsError("ASL preflight output exists; retain it as evidence rather than overwrite it")
    subprocess.run(["python", "/app/evaluate_asl.py", "--data-root", "/datasets/asl-alphabet", "--output-dir", str(output), "--images-per-class", "10", "--folds", "2"], check=True)
    results.commit()
    return json.loads((output / "run.json").read_text(encoding="utf-8"))


@app.function(
    image=evaluation_image,
    cpu=8,
    timeout=8 * 60 * 60,
    volumes={"/datasets": datasets, "/cache/huggingface": cache, "/results": results},
    env={"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub"},
)
def evaluate_asl_alphabet() -> dict[str, object]:
    """Run the documented 28-class conditional ASL evaluation once."""
    output = Path("/results/asl-alphabet-conditional")
    if output.exists():
        raise FileExistsError("ASL output exists; retain it as evidence rather than overwrite it")
    subprocess.run(["python", "/app/evaluate_asl.py", "--data-root", "/datasets/asl-alphabet", "--output-dir", str(output)], check=True)
    results.commit()
    return json.loads((output / "run.json").read_text(encoding="utf-8"))
