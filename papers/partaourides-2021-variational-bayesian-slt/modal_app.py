"""Modal entry points for the Partaourides et al. 2021 Gloss2Text reproduction.

Workspace invariant: run only via
    .agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run papers/partaourides-2021-variational-bayesian-slt/modal_app.py::<fn>

Functions:
    check_env                import + CUDA smoke test
    preflight                tiny 4-mode end-to-end check on the real data
    train  (mode=...)        full training of one model, writes /outputs/<mode>/
    evaluate (mode=..., quantize_bits=N)  re-score a saved checkpoint
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import threading
import time
from pathlib import Path

import modal

PAPER_DIR = Path(__file__).resolve().parent
RESULTS_VOLUME = "partaourides-2021-results"
DATA_DIR = "/datasets/rwth-phoenix-2014-t/annotations"

# sha256 of the three corpus CSVs actually used (verified before every run).
DATA_FILES = {
    "PHOENIX-2014-T.train.corpus.csv": "cc3dc2461f0a222b92f3927c24ac21c1467f3e5428b406ee7fe40bca1b0b8d44",
    "PHOENIX-2014-T.dev.corpus.csv": "1085141d0ed6f28c6de6196a271b72c07366ed3fe5470c9717bd44640737f00b",
    "PHOENIX-2014-T.test.corpus.csv": "632b19c9a87fb9c98b0821e04861750565348bce20a069c6dea1bba5bda27879",
}

image = (
    modal.Image.from_dockerfile(
        PAPER_DIR.parent.parent / "Dockerfile",
        context_dir=PAPER_DIR.parent.parent,
    )
    .pip_install("sacrebleu==2.4.3", "rouge-score==0.1.2")
    .add_local_file(PAPER_DIR / "sbgru_slt.py", "/app/sbgru_slt.py")
)

datasets = modal.Volume.from_name("datasets", create_if_missing=False)
cache = modal.Volume.from_name("huggingface-cache", create_if_missing=False)
results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)

ENV = {"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub"}
VOLUMES = {
    "/datasets": datasets.read_only(),
    "/cache/huggingface": cache,
    "/outputs": results,
}
app = modal.App("partaourides-2021-variational-bayesian-slt")


def _verify_data() -> None:
    for name, want in DATA_FILES.items():
        path = Path(DATA_DIR) / name
        if not path.is_file():
            raise FileNotFoundError(f"missing required corpus file: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(8 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != want:
            raise RuntimeError(f"checksum mismatch for {path}: {digest.hexdigest()} != {want}")


def _commit_periodically(stop: threading.Event) -> None:
    while not stop.wait(300):
        results.commit()


def _run(cmd: list[str]) -> dict:
    _verify_data()
    started_at = dt.datetime.now(dt.timezone.utc)
    t0 = time.monotonic()
    stop = threading.Event()
    committer = threading.Thread(target=_commit_periodically, args=(stop,), daemon=True)
    committer.start()
    try:
        proc = subprocess.run(["python", "-u", "/app/sbgru_slt.py", *cmd], cwd="/app", check=False)
    finally:
        stop.set()
        committer.join()
        results.commit()
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
    return {
        "function_call_id": modal.current_function_call_id(),
        "exit_code": proc.returncode,
        "started_at": started_at.isoformat(),
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "duration_seconds": time.monotonic() - t0,
        "gpu": gpu,
        "cmd": cmd,
    }


@app.function(image=image, gpu="A10G", timeout=1200, volumes=VOLUMES, environment_name="main")
def check_env() -> str:
    out = subprocess.run(
        ["python", "-c", "import torch,sacrebleu,rouge_score;print(torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0))"],
        capture_output=True, text=True,
    )
    msg = f"exit={out.returncode}\n{out.stdout}\n{out.stderr}"
    print(msg)
    return msg


@app.function(image=image, gpu="A10G", timeout=3600, volumes=VOLUMES, environment_name="main")
def preflight() -> dict:
    return _run(["preflight", "--data-dir", DATA_DIR, "--scratch", "/outputs/_preflight"])


@app.function(image=image, gpu="A10G", timeout=24 * 3600, volumes=VOLUMES, environment_name="main")
def train(mode: str, max_epochs: int = 0) -> dict:
    out_dir = f"/outputs/{mode}"
    meta = _run(["train", "--data-dir", DATA_DIR, "--out-dir", out_dir,
                 "--mode", mode, "--max-epochs", str(max_epochs)])
    res_path = Path(out_dir) / "results.json"
    if res_path.is_file():
        meta["results"] = json.loads(res_path.read_text())
    (Path(out_dir) / "run_meta.json").write_text(json.dumps(meta, indent=2))
    results.commit()
    return meta


@app.function(image=image, gpu="A10G", timeout=3600, volumes=VOLUMES, environment_name="main")
def evaluate(mode: str, quantize_bits: int = 0) -> dict:
    out_dir = f"/outputs/{mode}/eval_q{quantize_bits}"
    meta = _run(["evaluate", "--data-dir", DATA_DIR,
                 "--checkpoint", f"/outputs/{mode}/best.pt",
                 "--out-dir", out_dir, "--quantize-bits", str(quantize_bits)])
    ev_path = Path(out_dir) / "eval.json"
    if ev_path.is_file():
        meta["eval"] = json.loads(ev_path.read_text())
    (Path(out_dir) / "run_meta.json").write_text(json.dumps(meta, indent=2))
    results.commit()
    return meta
