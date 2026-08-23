"""Run the pinned camgoz/nslt recipe on mounted PHOENIX-2014T videos."""

import datetime as dt
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import modal


PAPER_DIR = Path(__file__).resolve().parent.parent
PYTHON = "/usr/bin/python"
VIDEO_ROOT = Path("/datasets/rwth-phoenix-2014-t/videos")
OUTPUT_DIR = Path("/outputs/camgoz-nslt/luong-seed-285")

IMAGE = modal.Image.from_dockerfile(
    PAPER_DIR / "Dockerfile",
    context_dir=PAPER_DIR,
    add_python="3.11",
)
DATASETS = modal.Volume.from_name("datasets", create_if_missing=False)
HF_CACHE = modal.Volume.from_name("huggingface-cache", create_if_missing=False)
RESULTS = modal.Volume.from_name(
    "camgoz-nslt-results", create_if_missing=True, version=2
)
APP = modal.App("repro-camgoz-nslt")


def _environment():
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _commit_periodically(stop):
    while not stop.wait(600):
        RESULTS.commit()


def _rewrite_manifests():
    data_dir = Path("/tmp/nslt-data")
    data_dir.mkdir(exist_ok=True)
    placeholder = "<PATH_TO_EXTRACTED_AND_RESIZED_FRAMES>/features/fullFrame-227x227px/"
    for source in Path("/nslt/Data").iterdir():
        target = data_dir / source.name
        if source.suffix != ".sign":
            target.write_bytes(source.read_bytes())
            continue
        paths = []
        for line in source.read_text().splitlines():
            if not line.startswith(placeholder) or not line.endswith("/"):
                raise RuntimeError("unexpected upstream sign manifest entry")
            relative = line[len(placeholder):].rstrip("/")
            paths.append(str(VIDEO_ROOT / (relative + ".mp4")))
        target.write_text("\n".join(paths) + "\n")
    return data_dir


def _verify_videos(data_dir):
    expected = {"train": 7096, "dev": 519, "test": 642}
    for split, count in expected.items():
        files = list((VIDEO_ROOT / split).glob("*.mp4"))
        if len(files) != count:
            raise RuntimeError("%s has %d videos, expected %d" % (split, len(files), count))
        manifest = data_dir / ("phoenix2014T.%s.sign" % split)
        paths = [Path(line) for line in manifest.read_text().splitlines()]
        if len(paths) != count or any(not path.is_file() for path in paths):
            raise RuntimeError("%s video manifest does not match the mounted dataset" % split)


def _verify_resume():
    hparams_path = OUTPUT_DIR / "hparams"
    checkpoint_path = OUTPUT_DIR / "checkpoint"
    if not hparams_path.exists() and not checkpoint_path.exists():
        return
    if not hparams_path.is_file() or not checkpoint_path.is_file():
        raise RuntimeError("refusing to resume from an incomplete output directory")
    hparams = json.loads(hparams_path.read_text())
    expected = {
        "attention": "luong",
        "batch_size": 1,
        "learning_rate": 0.00001,
        "num_layers": 4,
        "num_units": 1000,
        "random_seed": 285,
        "residual": True,
        "source_reverse": True,
        "unit_type": "gru",
    }
    mismatches = {
        key: hparams.get(key)
        for key, value in expected.items()
        if hparams.get(key) != value
    }
    if mismatches:
        raise RuntimeError("refusing to resume mismatched hparams: %r" % mismatches)


def _training_command(data_dir):
    return [
        PYTHON,
        "-m",
        "nmt",
        "--src=sign",
        "--tgt=de",
        "--train_prefix=%s" % (data_dir / "phoenix2014T.train"),
        "--dev_prefix=%s" % (data_dir / "phoenix2014T.dev"),
        "--test_prefix=%s" % (data_dir / "phoenix2014T.test"),
        "--out_dir=%s" % OUTPUT_DIR,
        "--vocab_prefix=%s" % (data_dir / "phoenix2014T.vocab"),
        "--source_reverse=True",
        "--num_units=1000",
        "--num_layers=4",
        "--num_train_steps=150000",
        "--residual=True",
        "--attention=luong",
        "--base_gpu=0",
        "--unit_type=gru",
    ]


@APP.function(
    image=IMAGE,
    gpu="A100",
    cpu=8,
    memory=32768,
    volumes={
        "/datasets": DATASETS.read_only(),
        "/cache/huggingface": HF_CACHE,
        "/outputs": RESULTS,
    },
    timeout=86_400,
)
def train():
    data_dir = _rewrite_manifests()
    _verify_videos(data_dir)
    _verify_resume()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    command = _training_command(data_dir)

    started_at = dt.datetime.now(dt.timezone.utc)
    started = time.monotonic()
    stop = threading.Event()
    committer = threading.Thread(target=_commit_periodically, args=(stop,), daemon=True)
    committer.start()
    try:
        subprocess.check_call(command, cwd="/nslt/nslt", env=_environment())
    finally:
        stop.set()
        committer.join()
        RESULTS.commit()

    return {
        "function_call_id": modal.current_function_call_id(),
        "started_at": started_at.isoformat(),
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "duration_seconds": time.monotonic() - started,
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            universal_newlines=True,
        ).strip(),
        "upstream_revision": subprocess.check_output(
            ["git", "-C", "/nslt", "rev-parse", "HEAD"], universal_newlines=True
        ).strip(),
        "model_dir": str(OUTPUT_DIR),
        "command": command,
    }


@APP.local_entrypoint()
def launch_train():
    call = train.spawn()
    print(call.object_id)
