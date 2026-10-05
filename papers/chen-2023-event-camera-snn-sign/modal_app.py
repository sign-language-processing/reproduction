"""Modal entry point for the conditional SNN+STBP reconstruction (Chen et al. 2023, Table 2)."""

from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parent.parent
# Last thiswinex/STBP-simple commit before the paper's 2022-11-30 submission.
STBP_COMMIT = "dca9590b465c5cb37b3828bf6888c6857ad4238b"

app = modal.App("chen-2023-event-camera-snn-sign")
image = (
    modal.Image.from_dockerfile(REPOSITORY_ROOT / "Dockerfile", context_dir=REPOSITORY_ROOT)
    .run_commands(
        "git clone https://github.com/thiswinex/STBP-simple.git /opt/STBP-simple"
        f" && git -C /opt/STBP-simple checkout {STBP_COMMIT}"
    )
    .add_local_file(ROOT / "train.py", "/app/train.py")
)
datasets = modal.Volume.from_name("datasets", create_if_missing=False).read_only()
cache = modal.Volume.from_name("huggingface-cache", create_if_missing=False)
results = modal.Volume.from_name("chen-2023-event-camera-snn-sign-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    cpu=4,
    memory=16384,
    timeout=12 * 60 * 60,
    volumes={"/datasets": datasets, "/cache/huggingface": cache, "/results": results},
    env={"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub"},
)
def train(optimizer: str = "sgd", input_pool: str = "avg", name: str = "", epochs: int = 200, limit: int = 0) -> str:
    """Train and evaluate one variant; re-invoking resumes from the last epoch checkpoint."""
    import json
    import sys

    sys.path.insert(0, "/app")
    import train as reconstruction

    run = reconstruction.main(
        [
            "--data-root", "/datasets/dvs-sign-v2e",
            "--output-dir", f"/results/{name or f'{optimizer}-{input_pool}'}",
            "--optimizer", optimizer,
            "--input-pool", input_pool,
            "--epochs", str(epochs),
            "--limit", str(limit),
        ],
        on_epoch=results.commit,
    )
    results.commit()
    summary = {key: value for key, value in run.items() if key != "predictions"}
    print(json.dumps(summary, indent=2))
    return json.dumps(summary)
