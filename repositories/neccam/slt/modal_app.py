"""Run the pinned neccam/slt recipe on the shared Modal data."""

import datetime as dt
import hashlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import modal


PAPER_DIR = Path(__file__).resolve().parent
PYTHON = "/root/miniconda3/bin/python"
MODEL_DIR = Path("/outputs/neccam-slt/pami0-seed-42")
DATA_DIR = Path("/datasets/rwth-phoenix-2014-t/features/author")
DATA_FILES = {
    "train": (
        "PHOENIX2014T/phoenix14t.pami0.train",
        1_935_289_362,
        "196842893dd43c98a5574132dceccf50f3e8f95af853042377cf05519757a773",
    ),
    "dev": (
        "PHOENIX2014T/phoenix14t.pami0.dev",
        130_831_227,
        "5be78d8488eaa4400e3bfcac1cf9096f7932595e4b5116eab53c19024693c6e6",
    ),
    "test": (
        "PHOENIX2014T/phoenix14t.pami0.test",
        151_327_204,
        "068c85c2f675e21ce0e6e4e9d419bc63fac7f43678783a7e5eb452ecb38c566a",
    ),
}

IMAGE = modal.Image.from_dockerfile(
    PAPER_DIR / "Dockerfile",
    context_dir=PAPER_DIR,
    add_python="3.11",
)
DATASETS = modal.Volume.from_name("datasets", create_if_missing=False)
HF_CACHE = modal.Volume.from_name("huggingface-cache", create_if_missing=False)
RESULTS = modal.Volume.from_name(
    "neccam-slt-results", create_if_missing=True, version=2
)
APP = modal.App("repro-neccam-slt")


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["HF_HOME"] = "/cache/huggingface"
    environment["HF_HUB_CACHE"] = "/cache/huggingface/hub"
    return environment


def _config() -> Path:
    config = Path("/slt/configs/sign.yaml").read_text(encoding="utf-8")
    replacements = {
        "data_path: ./data/": f"data_path: {DATA_DIR}/",
        'model_dir: "./sign_sample_model"': f"model_dir: {MODEL_DIR}",
    }
    for old, new in replacements.items():
        if config.count(old) != 1:
            raise RuntimeError(f"expected one upstream config field: {old}")
        config = config.replace(old, new)
    path = Path("/tmp/sign.yaml")
    path.write_text(config, encoding="utf-8")
    return path


def _commit_periodically(stop: threading.Event) -> None:
    while not stop.wait(600):
        RESULTS.commit()


def _verify_data() -> None:
    for name, expected_size, expected_hash in DATA_FILES.values():
        path = DATA_DIR / name
        if not path.is_file() or path.stat().st_size != expected_size:
            raise FileNotFoundError(
                f"missing or invalid required pami0 archive: {path}"
            )
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_hash:
            raise RuntimeError(f"checksum mismatch for {path}")


@APP.function(
    image=IMAGE,
    gpu="T4",
    volumes={
        "/datasets": DATASETS.read_only(),
        "/cache/huggingface": HF_CACHE,
        "/outputs": RESULTS,
    },
    timeout=86_400,
)
def train() -> dict:
    _verify_data()

    started_at = dt.datetime.now(dt.timezone.utc)
    started = time.monotonic()
    stop = threading.Event()
    committer = threading.Thread(target=_commit_periodically, args=(stop,), daemon=True)
    committer.start()
    try:
        subprocess.run(
            [PYTHON, "-m", "signjoey", "train", str(_config())],
            cwd="/slt",
            env=_environment(),
            check=True,
        )
    finally:
        stop.set()
        committer.join()
        RESULTS.commit()

    result = {
        "function_call_id": modal.current_function_call_id(),
        "started_at": started_at.isoformat(),
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "duration_seconds": time.monotonic() - started,
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
        "upstream_revision": subprocess.check_output(
            ["git", "-C", "/slt", "rev-parse", "HEAD"], text=True
        ).strip(),
        "model_dir": str(MODEL_DIR),
    }
    (MODEL_DIR / "modal-run.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    RESULTS.commit()
    return result
