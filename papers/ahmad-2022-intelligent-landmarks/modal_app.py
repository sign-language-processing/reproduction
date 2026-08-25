"""Approved ISL-HS dataset population for the landmark reproduction."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parent.parent
app = modal.App("8526aecd1407305d-isl-hs-data")
image = modal.Image.from_dockerfile(
    REPOSITORY_ROOT / "Dockerfile", context_dir=REPOSITORY_ROOT
).add_local_file(ROOT / "data.sh", "/app/data.sh")
datasets = modal.Volume.from_name("datasets", create_if_missing=False)
cache = modal.Volume.from_name("huggingface-cache", create_if_missing=False)


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
