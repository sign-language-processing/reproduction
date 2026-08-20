"""Modal entry points for the neccam/slt reproduction."""

import datetime as dt
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import modal


PAPER_DIR = Path(__file__).resolve().parent
TRAIN_PYTHON = "/root/miniconda3/bin/python"
IMAGE = modal.Image.from_dockerfile(
    PAPER_DIR / "Dockerfile",
    context_dir=PAPER_DIR,
    add_python="3.11",
).add_local_dir(PAPER_DIR, "/repro", copy=False)
DATASETS = modal.Volume.from_name("datasets", create_if_missing=False)
HF_CACHE = modal.Volume.from_name("huggingface-cache", create_if_missing=False)
RESULTS = modal.Volume.from_name("neccam-slt-results", create_if_missing=False)
APP = modal.App("repro-neccam-slt")
ENVIRONMENT = {
    "HF_HOME": "/cache/huggingface",
    "HF_HUB_CACHE": "/cache/huggingface/hub",
}


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _training_env() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _run_streamed(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$ " + " ".join(command), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd="/slt",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=_training_env(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def _gpu_details() -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


def _modal_identity() -> dict:
    return {
        "function_call_id": modal.current_function_call_id(),
        "input_id": modal.current_input_id(),
    }


def _load_scores(model_dir: Path, result_stem: str = "best.IT_*") -> dict:
    output = subprocess.check_output(
        [
            TRAIN_PYTHON,
            "/repro/scripts/extract_scores.py",
            str(model_dir),
            "--result-stem",
            result_stem,
        ],
        text=True,
        env=_training_env(),
    )
    return json.loads(output)


def _commit_periodically(stop_event: threading.Event) -> None:
    while not stop_event.wait(600):
        RESULTS.commit()
        print("committed periodic checkpoint snapshot", flush=True)


@APP.function(
    image=IMAGE,
    gpu="A100",
    volumes={"/cache/huggingface": HF_CACHE},
    env=ENVIRONMENT,
    timeout=1_800,
)
def environment() -> dict:
    details = {
        "python": subprocess.check_output(
            [TRAIN_PYTHON, "--version"], text=True, env=_training_env()
        ).strip(),
        "torch": subprocess.check_output(
            [TRAIN_PYTHON, "-c", "import torch; print(torch.__version__)"],
            text=True,
            env=_training_env(),
        ).strip(),
        "upstream_revision": subprocess.check_output(
            ["git", "-C", "/slt", "rev-parse", "HEAD"], text=True
        ).strip(),
        "gpu": _gpu_details(),
        "torch_cuda": subprocess.check_output(
            [
                TRAIN_PYTHON,
                "-c",
                "import torch; print(torch.cuda.is_available(), torch.version.cuda)",
            ],
            text=True,
            env=_training_env(),
        ).strip(),
        "cuda_compute": subprocess.check_output(
            [TRAIN_PYTHON, "/repro/scripts/check_cuda.py"],
            text=True,
            env=_training_env(),
        ).strip(),
    }
    print(json.dumps(details, indent=2), flush=True)
    return details


@APP.function(
    image=IMAGE,
    volumes={
        "/datasets": DATASETS,
        "/cache/huggingface": HF_CACHE,
    },
    env=ENVIRONMENT,
    timeout=7_200,
)
def populate_features() -> None:
    DATASETS.reload()
    subprocess.run(
        [TRAIN_PYTHON, "/repro/scripts/populate_features.py"],
        cwd="/slt",
        check=True,
        env=_training_env(),
    )
    DATASETS.commit()


@APP.function(
    image=IMAGE,
    gpu="A100",
    volumes={
        "/datasets": DATASETS.read_only(),
        "/cache/huggingface": HF_CACHE,
        "/outputs": RESULTS,
    },
    env=ENVIRONMENT,
    timeout=3_600,
)
def dry_run() -> dict:
    started_at = _utc_now()
    started = time.monotonic()
    subprocess.run(
        [TRAIN_PYTHON, "/repro/scripts/make_dry_data.py"],
        check=True,
        env=_training_env(),
    )
    model_dir = Path("/outputs/neccam-slt/dry-seed-42")
    _run_streamed(
        [TRAIN_PYTHON, "-m", "signjoey", "train", "/repro/configs/dry.yaml"],
        Path("/outputs/neccam-slt/dry-run.log"),
    )
    checkpoint = model_dir / "best.ckpt"
    if not checkpoint.exists() or not checkpoint.resolve().is_file():
        raise FileNotFoundError(checkpoint)
    result = {
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_seconds": time.monotonic() - started,
        "gpu": _gpu_details(),
        **_modal_identity(),
        "checkpoint": str(checkpoint.resolve()),
        **_load_scores(model_dir),
    }
    result_path = Path("/outputs/neccam-slt/dry-run.json")
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    RESULTS.commit()
    return result


@APP.function(
    image=IMAGE,
    gpu="A100",
    volumes={
        "/datasets": DATASETS.read_only(),
        "/cache/huggingface": HF_CACHE,
        "/outputs": RESULTS,
    },
    env=ENVIRONMENT,
    timeout=86_400,
)
def train() -> dict:
    started_at = _utc_now()
    started = time.monotonic()
    stop_event = threading.Event()
    commit_thread = threading.Thread(
        target=_commit_periodically, args=(stop_event,), daemon=True
    )
    commit_thread.start()
    try:
        _run_streamed(
            [TRAIN_PYTHON, "-m", "signjoey", "train", "/repro/configs/sign.yaml"],
            Path("/outputs/neccam-slt/full-seed-42/modal-run.log"),
        )
    finally:
        stop_event.set()
        commit_thread.join()
        RESULTS.commit()

    model_dir = Path("/outputs/neccam-slt/full-seed-42")
    result = {
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_seconds": time.monotonic() - started,
        "gpu": _gpu_details(),
        **_modal_identity(),
        **_load_scores(model_dir),
    }
    result_path = model_dir / "modal-result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    RESULTS.commit()
    return result


@APP.function(
    image=IMAGE,
    volumes={
        "/cache/huggingface": HF_CACHE,
        "/outputs": RESULTS,
    },
    env=ENVIRONMENT,
    timeout=600,
)
def collect_results() -> dict:
    model_dir = Path("/outputs/neccam-slt/full-seed-42")
    result = {
        "collected_at": _utc_now(),
        **_load_scores(model_dir, result_stem="re-eval"),
    }
    (model_dir / "modal-result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    RESULTS.commit()
    print(json.dumps(result, sort_keys=True))
    return result


@APP.function(
    image=IMAGE,
    gpu="A100",
    volumes={
        "/datasets": DATASETS.read_only(),
        "/cache/huggingface": HF_CACHE,
        "/outputs": RESULTS,
    },
    env=ENVIRONMENT,
    timeout=43_200,
)
def evaluate() -> dict:
    model_dir = Path("/outputs/neccam-slt/full-seed-42")
    checkpoint = model_dir / "best.ckpt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    _run_streamed(
        [
            TRAIN_PYTHON,
            "-m",
            "signjoey",
            "test",
            "/repro/configs/sign.yaml",
            "--ckpt",
            str(checkpoint),
            "--output_path",
            str(model_dir / "re-eval"),
        ],
        model_dir / "modal-eval.log",
    )
    RESULTS.commit()
    return _load_scores(model_dir, result_stem="re-eval")
