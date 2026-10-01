"""Modal entry points for the CVSign reproduction on PHOENIX14-T.

The recipe is upstream code run as published: CorrNet's main.py, pinned in the
Dockerfile, with TLP's temporal decoder and the paper's CCA/CVA applied as the
patches in patches/. This file only stages data into the container, runs
upstream commands, and keeps their outputs on the paper's results Volume.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

PAPER_DIR = Path(__file__).resolve().parent
IMAGE = modal.Image.from_dockerfile(PAPER_DIR / "Dockerfile", context_dir=PAPER_DIR)
DATASETS = modal.Volume.from_name("datasets")
HF_CACHE = modal.Volume.from_name("huggingface-cache")
RESULTS = modal.Volume.from_name("yu-2025-cvsign-results", version=2)
APP = modal.App("repro-yu-2025-cvsign")
VOLUMES = {"/datasets": DATASETS.read_only(), "/cache/huggingface": HF_CACHE, "/results": RESULTS}
GPU = "L40S"  # 48 GB, the memory of the paper's single NVIDIA A6000

SOURCE = Path("/datasets/rwth-phoenix-2014-t/raw/PHOENIX-2014-T-release-v3/PHOENIX-2014-T")
FRAMES_TAR = Path("/results/data/phoenix2014-T-fullFrame-256x256px.tar")
FRAMES_MANIFEST = Path(f"{FRAMES_TAR}.json")
STAGED = Path("/data/phoenix2014-T")  # ./dataset/phoenix2014-T symlinks here (Dockerfile)
CORRNET = "/opt/CorrNet"
CVSIGN = "/workspace/CVSign"
# CorrNet README, PHOENIX2014-T ResNet18 checkpoint (reported dev 18.9 / test 20.5).
CORRNET_CHECKPOINT_ID = "1c_wNHYMqCbqRE5KqrQL1P6chOw5VBS6Q"
CORRNET_CHECKPOINT = Path("/results/weights/corrnet-phoenix2014-T-resnet18.pt")
SPLITS = ("train", "dev", "test")


def _sha256(path, chunk=1 << 24):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _environment(repo):
    def out(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()

    return {
        "gpu": out("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"),
        "torch": out("python", "-c", "import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version())"),
        "repo_head": out("git", "-C", repo, "rev-parse", "HEAD"),
        "repo_dirty_diff_sha256": hashlib.sha256(out("git", "-C", repo, "diff", "HEAD").encode()).hexdigest(),
        "image_id": os.environ.get("MODAL_IMAGE_ID"),
        "task_id": os.environ.get("MODAL_TASK_ID"),
    }


def _stage_frames():
    """Unpacks the preprocessed frames onto local disk for upstream's loader."""
    manifest = json.loads(FRAMES_MANIFEST.read_text())
    if not manifest["checks_passed"]:
        raise RuntimeError(f"frame archive failed its checks: {manifest['checks']}")
    started = time.monotonic()
    if not STAGED.exists():
        STAGED.mkdir(parents=True)
        subprocess.run(["tar", "-xf", str(FRAMES_TAR), "-C", str(STAGED)], check=True)
    for split in SPLITS:
        root = STAGED / "features/fullFrame-256x256px" / split
        count = sum(len(files) for _, _, files in os.walk(root))
        if count != manifest["frames"][split]:
            raise RuntimeError(f"{split}: {count} staged frames, manifest says {manifest['frames'][split]}")
    return {"stage_seconds": round(time.monotonic() - started, 1), "frames_tar_sha256": manifest["sha256"]}


def _run_upstream(command, cwd, work_dir):
    """Runs one upstream command, keeping its log and committing outputs as it goes.

    Upstream main.py asks on stdin whether to wipe an existing work_dir; the
    answer is always no, so a resumed run keeps its checkpoints and logs.
    """
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    peak = {"mib": 0}

    def commit_periodically():
        while not stop.wait(300):
            RESULTS.commit()

    def sample_memory():
        while not stop.wait(5):
            used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True).stdout.split()
            if used:
                peak["mib"] = max(peak["mib"], int(used[0]))

    threads = [threading.Thread(target=f, daemon=True) for f in (commit_periodically, sample_memory)]
    for thread in threads:
        thread.start()
    started_at, started = _now(), time.monotonic()
    with open(Path(work_dir) / "stdout.log", "a") as log:
        log.write(f"\n### {started_at} {' '.join(command)}\n")
        log.flush()
        proc = subprocess.run(command, cwd=cwd, input="n\n", text=True, stdout=log, stderr=subprocess.STDOUT)
    stop.set()
    for thread in threads:
        thread.join()
    RESULTS.commit()
    # seq_eval() swallows scoring errors and logs a WER of 100 instead.
    scoring_failed = "Unexpected error" in (Path(work_dir) / "stdout.log").read_text()
    return {
        "scoring_failed": scoring_failed,
        "command": command,
        "cwd": cwd,
        "exit_code": proc.returncode,
        "started_at_utc": started_at,
        "finished_at_utc": _now(),
        "duration_seconds": round(time.monotonic() - started, 1),
        "peak_gpu_memory_mib_nvidia_smi": peak["mib"],
    }


def _wer(work_dir, split):
    """Last WER upstream logged for a split: '[ time ] Epoch N, dev 18.90%'."""
    lines = (Path(work_dir) / f"{split}.txt").read_text().splitlines()
    epoch, wer = re.search(r"Epoch (\d+), \w+ +([\d.]+)%", lines[-1]).groups()
    return int(epoch), float(wer)


def _dev_history(work_dir):
    history = {}
    for line in (Path(work_dir) / "dev.txt").read_text().splitlines():
        epoch, wer = re.search(r"Epoch (\d+), dev +([\d.]+)%", line).groups()
        history[int(epoch)] = float(wer)  # a resumed epoch overwrites its earlier entry
    return history


def _epoch_checkpoints(work_dir):
    found = {}
    for path in Path(work_dir).glob("dev_*_epoch*_model.pt"):
        found[int(re.search(r"_epoch(\d+)_model\.pt$", path.name).group(1))] = path
    return found


@APP.function(image=IMAGE, cpu=32, memory=65536, ephemeral_disk=512 * 1024, volumes=VOLUMES, timeout=12 * 3600)
def prepare_frames():
    """Resizes the official 210x260 PNG frames to 256x256 with CorrNet's own code, and archives them.

    CorrNet's preprocess/dataset_preprocess-T.py cannot be run end to end on
    this release: its CSVs list frames as <id>/1/*.png, but the frames sit at
    <id>/*.png, so the script's globs match nothing. CorrNet's committed
    <split>_info.npy files already use <id>/*.png and otherwise equal the CSVs
    (checked first, below), so the script's own resize_dataset() is driven with
    them, exactly as its __main__ does. It writes next to the originals, so it
    runs on a local copy; the datasets Volume stays read-only.
    """
    if FRAMES_MANIFEST.exists():
        return json.loads(FRAMES_MANIFEST.read_text())
    import importlib.util
    from functools import partial

    import numpy as np

    started_at, started = _now(), time.monotonic()
    committed = {split: np.load(f"{CORRNET}/preprocess/phoenix2014-T/{split}_info.npy", allow_pickle=True).item()
                 for split in SPLITS}
    for split, info in committed.items():
        rows = (SOURCE / f"annotations/manual/PHOENIX-2014-T.{split}.corpus.csv").read_text(encoding="utf-8").splitlines()[1:]
        expected = [row.replace("/1/*.png", "/*.png") for row in rows]
        if [info[k]["original_info"] for k in range(len(rows))] != expected or len(info) - 1 != len(rows):
            raise RuntimeError(f"CorrNet's {split}_info.npy does not match the official CSV up to the frame path")

    local = Path("/tmp/PHOENIX-2014-T")
    sources = []
    for split in SPLITS:
        for clip in os.scandir(SOURCE / "features/fullFrame-210x260px" / split):
            sources += [Path(entry.path) for entry in os.scandir(clip.path)]

    def copy(path):
        target = local / path.relative_to(SOURCE)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)

    with ThreadPoolExecutor(256) as pool:
        list(pool.map(copy, sources))
    copied = time.monotonic()

    spec = importlib.util.spec_from_file_location("preprocess_t", f"{CORRNET}/preprocess/dataset_preprocess-T.py")
    upstream = importlib.util.module_from_spec(spec)
    sys.modules["preprocess_t"] = upstream  # its Pool pickles resize_dataset by module name
    spec.loader.exec_module(upstream)
    for split, info in committed.items():
        info["prefix"] = str(local / "features/fullFrame-210x260px")
        upstream.run_mp_cmd(10, partial(upstream.resize_dataset, dsize="256x256px", info_dict=info), np.arange(len(info) - 1))
    resized = time.monotonic()

    def pngs(root):
        return sum(name.endswith(".png") for _, _, names in os.walk(root) for name in names)

    frames, checks = {}, {}
    for split, info in committed.items():
        frames[split] = pngs(local / "features/fullFrame-256x256px" / split)
        checks[split] = {
            "resized_png": frames[split],
            "original_png": pngs(local / "features/fullFrame-210x260px" / split),
            "corrnet_num_frames": sum(info[k]["num_frames"] for k in info if isinstance(k, int)),
            "original_non_png": sorted(str(Path(d, n).relative_to(local)) for d, _, names in
                                       os.walk(local / "features/fullFrame-210x260px" / split)
                                       for n in names if not n.endswith(".png")),
        }
    checks_passed = all(c["resized_png"] == c["original_png"] == c["corrnet_num_frames"] for c in checks.values())

    FRAMES_TAR.parent.mkdir(parents=True, exist_ok=True)
    partial_tar = Path(f"{FRAMES_TAR}.partial")
    subprocess.run(["tar", "--sort=name", "--owner=0", "--group=0", "--numeric-owner", "--mtime=@0",
                    "-cf", str(partial_tar), "-C", str(local), "features/fullFrame-256x256px"], check=True)
    manifest = {
        "source": str(SOURCE),
        "method": f"{CORRNET}/preprocess/dataset_preprocess-T.py: run_mp_cmd(10, resize_dataset(dsize='256x256px')) over CorrNet's committed <split>_info.npy",
        "corrnet_commit": subprocess.run(["git", "-C", CORRNET, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "opencv": subprocess.run(["python", "-c", "import cv2; print(cv2.__version__)"], capture_output=True, text=True).stdout.strip(),
        "frames": frames,
        "checks": checks,
        "checks_passed": checks_passed,
        "bytes": partial_tar.stat().st_size,
        "sha256": _sha256(partial_tar),
        "started_at_utc": started_at,
        "finished_at_utc": _now(),
        "seconds": {"copy": round(copied - started), "resize": round(resized - copied), "total": round(time.monotonic() - started)},
    }
    partial_tar.rename(FRAMES_TAR)
    FRAMES_MANIFEST.write_text(json.dumps(manifest, indent=2))
    RESULTS.commit()
    return manifest


@APP.function(image=IMAGE, gpu=GPU, cpu=16, memory=65536, ephemeral_disk=512 * 1024, volumes=VOLUMES, timeout=4 * 3600)
def check_corrnet_checkpoint(run_id: str):
    """Scores CorrNet's released PHOENIX14-T checkpoint with pristine CorrNet.

    This exercises everything CVSign reuses (frames, loader, ctcdecode beam
    search, sclite scoring) against a published number, without training.
    """
    import gdown

    work_dir = f"/results/runs/{run_id}/"
    if not CORRNET_CHECKPOINT.exists():
        CORRNET_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(id=CORRNET_CHECKPOINT_ID, output=str(CORRNET_CHECKPOINT), quiet=True)
        RESULTS.commit()
    record = {"checkpoint_sha256": _sha256(CORRNET_CHECKPOINT), **_stage_frames(), **_environment(CORRNET)}
    command = ["python", "main.py", "--config", "./configs/baseline.yaml", "--dataset", "phoenix2014-T",
               "--device", "0", "--num-worker", "12", "--phase", "test",
               "--load-weights", str(CORRNET_CHECKPOINT), "--work-dir", work_dir]
    record.update(_run_upstream(command, CORRNET, work_dir))
    record["wer"] = {split: _wer(work_dir, split)[1] for split in ("dev", "test")}
    return record


def _write_preflight_subset(n_train=200, n_long=20, n_eval=50):
    """A 'phoenix2014-T-preflight' dataset for upstream main.py: real clips, subset lists.

    The longest training clips are included so peak memory is representative.
    """
    import numpy as np

    base = Path(CVSIGN)
    target = base / "preprocess/phoenix2014-T-preflight"
    target.mkdir(parents=True, exist_ok=True)
    evaluation = base / "evaluation/preflight"
    evaluation.mkdir(parents=True, exist_ok=True)
    for name in ("preprocess.sh", "mergectmstm.py"):
        shutil.copy(base / "evaluation/slr_eval" / name, evaluation / name)
    chosen = {}
    for split in SPLITS:
        info = np.load(base / f"preprocess/phoenix2014-T/{split}_info.npy", allow_pickle=True).item()
        clips = [info[k] for k in sorted(k for k in info if isinstance(k, int))]
        if split == "train":
            longest = sorted(clips, key=lambda c: -c["num_frames"])[:n_long]
            rest = [c for c in clips if c not in longest]
            picked = longest + [rest[i] for i in np.random.RandomState(0).choice(len(rest), n_train - n_long, replace=False)]
        else:
            picked = clips[:n_eval]
        subset = {i: clip for i, clip in enumerate(picked)}
        subset["prefix"] = info["prefix"]
        np.save(target / f"{split}_info.npy", subset)
        ids = {clip["fileid"] for clip in picked}
        stm = (base / f"evaluation/slr_eval/phoenix2014-T-groundtruth-{split}.stm").read_text().splitlines(keepends=True)
        (evaluation / f"phoenix2014-T-groundtruth-{split}.stm").write_text("".join(l for l in stm if l.split(" ")[0] in ids))
        chosen[split] = {"clips": len(picked), "frames": sum(c["num_frames"] for c in picked)}
    (base / "configs/phoenix2014-T-preflight.yaml").write_text(
        "dataset_root: ./dataset/phoenix2014-T\n"
        "dict_path: ./preprocess/phoenix2014-T/gloss_dict.npy\n"
        "evaluation_dir: ./evaluation/preflight\n"
        "evaluation_prefix: phoenix2014-T-groundtruth\n")
    return chosen


def _cvsign_command(work_dir, *extra, dataset="phoenix2014-T", workers=12):
    return ["python", "main.py", "--config", "./configs/cvsign.yaml", "--dataset", dataset, "--device", "0",
            "--num-worker", str(workers), "--save-interval", "1", "--work-dir", work_dir, *extra]


@APP.function(image=IMAGE, gpu=GPU, cpu=16, memory=65536, ephemeral_disk=512 * 1024, volumes=VOLUMES, timeout=4 * 3600)
def preflight(run_id: str):
    """Real clips through upstream main.py: train an epoch, resume for a second, then test."""
    work_dir = f"/results/runs/{run_id}/"
    record = {"subset": _write_preflight_subset(), **_stage_frames(), **_environment(CVSIGN), "steps": []}
    steps = (
        lambda: ["--num-epoch", "1"],
        lambda: ["--num-epoch", "2", "--load-checkpoints", str(_epoch_checkpoints(work_dir)[0])],
        lambda: ["--phase", "test", "--load-weights", str(_epoch_checkpoints(work_dir)[1])],
    )
    for extra in steps:
        step = _run_upstream(_cvsign_command(work_dir, *extra(), dataset="phoenix2014-T-preflight"), CVSIGN, work_dir)
        record["steps"].append(step)
        if step["exit_code"]:
            return record
    log = (Path(work_dir) / "log.txt").read_text()
    record["epoch_seconds"] = [int(m) * 60 + int(s) for m, s in re.findall(r"Epoch \d+ costs (\d+) mins (\d+) seconds", log)]
    record["dev_wer_history"] = _dev_history(work_dir)
    record["test_phase_wer"] = {split: _wer(work_dir, split)[1] for split in ("dev", "test")}
    record["checkpoint_bytes"] = _epoch_checkpoints(work_dir)[1].stat().st_size
    return record


# Upstream evaluates in fp32, where CCA's layer-2 affinity for the longest dev
# clips is one 10.4 GiB block; with the default allocator that failed twice on
# fragmentation (11.5 GiB reserved but unusable). Expandable segments only
# change how freed memory is reused, not what is computed.
ALLOCATOR_ENV = {"PYTORCH_ALLOC_CONF": "expandable_segments:True"}


@APP.function(image=IMAGE, gpu=GPU, cpu=16, memory=65536, ephemeral_disk=512 * 1024, volumes=VOLUMES, env=ALLOCATOR_ENV,
              timeout=24 * 3600, retries=modal.Retries(max_retries=3, initial_delay=60.0))
def train(run_id: str, epochs: int = 70):
    """Trains CVSign to `epochs`, resuming from the newest epoch checkpoint when one exists.

    A call that hits Modal's 24 h limit is retried and resumes from there.
    """
    work_dir = f"/results/runs/{run_id}/"
    RESULTS.reload()
    record = {**_stage_frames(), **_environment(CVSIGN)}
    checkpoints = _epoch_checkpoints(work_dir)
    extra = ["--num-epoch", str(epochs)]
    if checkpoints:
        if max(checkpoints) + 1 >= epochs:
            return {**record, "already_complete": True, "dev_wer_history": _dev_history(work_dir)}
        extra += ["--load-checkpoints", str(checkpoints[max(checkpoints)])]
    record.update(_run_upstream(_cvsign_command(work_dir, *extra), CVSIGN, work_dir))
    record["dev_wer_history"] = _dev_history(work_dir)
    if record["exit_code"]:
        raise RuntimeError(f"upstream training exited {record['exit_code']}")
    return record


@APP.function(image=IMAGE, gpu=GPU, cpu=16, memory=65536, ephemeral_disk=512 * 1024, volumes=VOLUMES, env=ALLOCATOR_ENV,
              timeout=4 * 3600)
def evaluate(train_run_id: str, eval_run_id: str):
    """Selects the lowest-dev-WER epoch (earliest on ties) and scores dev and test once."""
    train_dir = f"/results/runs/{train_run_id}/"
    history = _dev_history(train_dir)
    best = min(history, key=lambda e: (history[e], e))
    checkpoint = _epoch_checkpoints(train_dir)[best]
    work_dir = f"/results/runs/{eval_run_id}/"
    record = {"selected_epoch": best, "selected_dev_wer_during_training": history[best],
              "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
              **_stage_frames(), **_environment(CVSIGN)}
    record.update(_run_upstream(_cvsign_command(work_dir, "--phase", "test", "--load-weights", str(checkpoint)), CVSIGN, work_dir))
    record["wer"] = {split: _wer(work_dir, split)[1] for split in ("dev", "test")}
    return record


def _print(record):
    print(json.dumps(record, indent=2, default=str))


@APP.local_entrypoint()
def launch_prepare_frames():
    _print(prepare_frames.remote())


@APP.local_entrypoint()
def launch_check_corrnet(run_id: str = "corrnet-checkpoint-check-001"):
    _print(check_corrnet_checkpoint.remote(run_id))


@APP.local_entrypoint()
def launch_preflight(run_id: str = "preflight-phoenix14t-003"):
    _print(preflight.remote(run_id))


@APP.local_entrypoint()
def launch_train(run_id: str = "full-phoenix14t-002", epochs: int = 70):
    call = train.spawn(run_id, epochs)
    print(call.object_id)


@APP.local_entrypoint()
def launch_evaluate(train_run_id: str = "full-phoenix14t-002", eval_run_id: str = "eval-phoenix14t-002"):
    _print(evaluate.remote(train_run_id, eval_run_id))
