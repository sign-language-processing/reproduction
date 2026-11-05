"""Modal entry points for the SC2SLR reproduction on PHOENIX-2014 and PHOENIX-2014T (Table 3).

The recipe is the authors' released code (pinned in the Dockerfile) plus the
patches in patches/. This file stages data into containers, extracts the
AlphaPose keypoints once, runs upstream main.py, and keeps outputs on the
paper's results Volume.
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
POSE_IMAGE = modal.Image.from_dockerfile(PAPER_DIR / "alphapose.Dockerfile", context_dir=PAPER_DIR)
DATASETS = modal.Volume.from_name("datasets")
HF_CACHE = modal.Volume.from_name("huggingface-cache")
RESULTS = modal.Volume.from_name("lyu-2024-sc2slr-results", version=2, create_if_missing=True)
APP = modal.App("repro-lyu-2024-sc2slr")
VOLUMES = {"/datasets": DATASETS.read_only(), "/cache/huggingface": HF_CACHE, "/results": RESULTS}
GPU = "A100-80GB"

SC2SLR = "/workspace/SC2SLR"
ALPHAPOSE = "/opt/AlphaPose"
SPLITS = ("train", "dev", "test")
# Per dataset: the RWTH release root on the datasets Volume, the directory
# upstream's config reads as dataset_root (./dataset/... symlinks to /data),
# the upstream preprocess script, and the annotation CSV name.
DATASETS_INFO = {
    "phoenix2014": {
        "source": "/datasets/rwth-phoenix-2014/raw/phoenix2014-release/phoenix-2014-multisigner",
        "staged": "/data/phoenix2014/phoenix-2014-multisigner",
        "preprocess": "preprocess/dataset_preprocess.py",
        "csv": "annotations/manual/{}.corpus.csv",
    },
    "phoenix2014-T": {
        "source": "/datasets/rwth-phoenix-2014-t/raw/PHOENIX-2014-T-release-v3/PHOENIX-2014-T",
        "staged": "/data/phoenix2014-T",
        "preprocess": "preprocess/dataset_preprocess-T.py",
        "csv": "annotations/manual/PHOENIX-2014-T.{}.corpus.csv",
    },
}
# AlphaPose README's Halpe-136 model (FastPose ResNet50, symmetric integral
# regression) and the YOLOv3-SPP detector from docs/INSTALL.md.
POSE_CFG = "configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml"
POSE_WEIGHTS = {
    "halpe136_fast50_regression_256x192.pth": "1_10JYI3O-VbrAiONfL36UxLf9UXMoUYA",
    "yolov3-spp.weights": "1D47msNOOiJKvPOXlnpyzdKA3k6E97NTC",
}
ORIGINAL_SIZE = (210, 260)  # width, height of the RWTH full frames
RESIZED_SIZE = (256, 256)


def _sha256(path, chunk=1 << 24):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _out(*cmd, cwd=None):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd).stdout.strip()


def _environment(repo):
    return {
        "gpu": _out("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"),
        "torch": _out("python", "-c", "import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version())"),
        "repo_head": _out("git", "-C", repo, "rev-parse", "HEAD"),
        "repo_dirty_diff_sha256": hashlib.sha256(_out("git", "-C", repo, "diff", "HEAD").encode()).hexdigest(),
        "image_id": os.environ.get("MODAL_IMAGE_ID"),
        "task_id": os.environ.get("MODAL_TASK_ID"),
    }


def _data_path(dataset, kind):
    return Path(f"/results/data/{dataset}-{kind}.tar")


def _clip_dirs(root):
    """Relative directories (e.g. train/<id>/1 or train/<id>) that hold PNG frames, sorted."""
    found = []
    for split in SPLITS:
        for directory, _, names in os.walk(root / split):
            if any(name.endswith(".png") for name in names):
                found.append(str(Path(directory).relative_to(root)))
    return sorted(found)


def _frames(directory):
    return sorted(name for name in os.listdir(directory) if name.endswith(".png"))


# --------------------------------------------------------------------------- frames

@APP.function(image=IMAGE, cpu=32, memory=65536, ephemeral_disk=512 * 1024, volumes=VOLUMES, timeout=12 * 3600)
def prepare_frames(dataset: str):
    """Resizes the official 210x260 PNG frames to 256x256 with SC2SLR's own preprocess code, and archives them.

    The upstream script is driven with the repo's committed <split>_info.npy
    (after checking them against the official CSVs), exactly as its __main__
    does. It writes next to the originals, so it runs on a local copy; the
    datasets Volume stays read-only.
    """
    tar, manifest_path = _data_path(dataset, "fullFrame-256x256px"), Path(f"{_data_path(dataset, 'fullFrame-256x256px')}.json")
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    import importlib.util
    from functools import partial

    import numpy as np

    info = DATASETS_INFO[dataset]
    source = Path(info["source"])
    started_at, started = _now(), time.monotonic()
    committed = {split: np.load(f"{SC2SLR}/preprocess/{dataset}/{split}_info.npy", allow_pickle=True).item()
                 for split in SPLITS}
    csv_checks = {}
    for split, split_info in committed.items():
        rows = (source / info["csv"].format(split)).read_text(encoding="utf-8").splitlines()[1:]
        ids = [row.split("|")[0] for row in rows]
        committed_ids = [split_info[k]["fileid"] for k in range(len(split_info) - 1)]
        csv_checks[split] = {"csv_rows": len(rows), "committed_clips": len(committed_ids),
                             "missing_from_committed": sorted(set(ids) - set(committed_ids)),
                             "not_in_csv": sorted(set(committed_ids) - set(ids)),
                             "folder_example": split_info[0]["folder"]}
        if csv_checks[split]["not_in_csv"]:
            raise RuntimeError(f"{split}_info.npy lists clips that are not in the official CSV: {csv_checks[split]}")

    local = Path(f"/tmp/{dataset}")
    sources = [Path(entry.path) for clip in _clip_dirs(source / "features/fullFrame-210x260px")
               for entry in os.scandir(source / "features/fullFrame-210x260px" / clip)]

    def copy(path):
        target = local / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)

    with ThreadPoolExecutor(256) as pool:
        list(pool.map(copy, sources))
    copied = time.monotonic()

    spec = importlib.util.spec_from_file_location("sc2slr_preprocess", f"{SC2SLR}/{info['preprocess']}")
    upstream = importlib.util.module_from_spec(spec)
    sys.modules["sc2slr_preprocess"] = upstream  # its Pool pickles resize_dataset by module name
    spec.loader.exec_module(upstream)
    for split_info in committed.values():
        split_info["prefix"] = str(local / "features/fullFrame-210x260px")
        upstream.run_mp_cmd(10, partial(upstream.resize_dataset, dsize="256x256px", info_dict=split_info),
                            np.arange(len(split_info) - 1))
    resized = time.monotonic()

    frames, checks = {}, {}
    for split, split_info in committed.items():
        clips = [split_info[k] for k in range(len(split_info) - 1)]
        resized_counts = [len(list((local / "features/fullFrame-256x256px").glob(c["folder"]))) for c in clips]
        original_counts = [len(list((local / "features/fullFrame-210x260px").glob(c["folder"]))) for c in clips]
        frames[split] = sum(resized_counts)
        checks[split] = {
            "clips": len(clips),
            "resized_png": frames[split],
            "original_png_of_committed_clips": sum(original_counts),
            "clips_without_frames": sum(n == 0 for n in resized_counts),
            **csv_checks[split],
        }
    checks_passed = all(c["resized_png"] == c["original_png_of_committed_clips"] and c["clips_without_frames"] == 0
                        for c in checks.values())

    tar.parent.mkdir(parents=True, exist_ok=True)
    partial_tar = Path(f"{tar}.partial")
    subprocess.run(["tar", "--sort=name", "--owner=0", "--group=0", "--numeric-owner", "--mtime=@0",
                    "-cf", str(partial_tar), "-C", str(local), "features/fullFrame-256x256px"], check=True)
    manifest = {
        "source": str(source),
        "method": f"{info['preprocess']}: run_mp_cmd(10, resize_dataset(dsize='256x256px')) over the committed <split>_info.npy",
        "sc2slr_commit": _out("git", "-C", SC2SLR, "rev-parse", "HEAD"),
        "opencv": _out("python", "-c", "import cv2; print(cv2.__version__)"),
        "frames": frames,
        "checks": checks,
        "checks_passed": checks_passed,
        "bytes": partial_tar.stat().st_size,
        "sha256": _sha256(partial_tar),
        "started_at_utc": started_at,
        "finished_at_utc": _now(),
        "seconds": {"copy": round(copied - started), "resize": round(resized - copied), "total": round(time.monotonic() - started)},
    }
    partial_tar.rename(tar)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    RESULTS.commit()
    return manifest


# --------------------------------------------------------------------------- keypoints

def _pose_weights():
    import gdown

    directory = Path("/results/weights/alphapose")
    directory.mkdir(parents=True, exist_ok=True)
    for name, drive_id in POSE_WEIGHTS.items():
        if not (directory / name).exists():
            gdown.download(id=drive_id, output=str(directory / f"{name}.partial"), quiet=True)
            (directory / f"{name}.partial").rename(directory / name)
            RESULTS.commit()
    return {name: _sha256(directory / name) for name in POSE_WEIGHTS}, directory


@APP.function(image=POSE_IMAGE, gpu="A10G", cpu=8, memory=32768, ephemeral_disk=512 * 1024, volumes=VOLUMES,
              timeout=6 * 3600, retries=modal.Retries(max_retries=3, initial_delay=60.0))
def extract_keypoints(dataset: str, shard: int, num_shards: int, limit_clips: int = 0):
    """Runs AlphaPose (Halpe-136) once over every original frame of one shard of clips.

    Writes /results/keypoints/<dataset>/<clip>.npy, a (T, 136, 3) float32
    array of x, y in 256x256-frame pixels and confidence, in sorted frame
    order, keeping the highest-scoring person per frame; frames without a
    detection are zeros. limit_clips > 0 is for the preflight only.
    """
    import numpy as np

    out_root = Path(f"/results/keypoints/{dataset}")
    marker = out_root / "_shards" / f"{shard:03d}-of-{num_shards:03d}{'-limit' + str(limit_clips) if limit_clips else ''}.json"
    if marker.exists():
        return json.loads(marker.read_text())
    started_at, started = _now(), time.monotonic()
    weights_sha256, weights_dir = _pose_weights()
    det_target = Path(ALPHAPOSE) / "detector/yolo/data/yolov3-spp.weights"
    det_target.parent.mkdir(parents=True, exist_ok=True)
    if not det_target.exists():
        os.symlink(weights_dir / "yolov3-spp.weights", det_target)

    root = Path(DATASETS_INFO[dataset]["source"]) / "features/fullFrame-210x260px"
    clips = _clip_dirs(root)[shard::num_shards]
    if limit_clips:
        clips = clips[:limit_clips]
    flat, names, index = Path("/tmp/flat"), [], []
    flat.mkdir(parents=True, exist_ok=True)
    jobs = []
    for clip in clips:
        frames = _frames(root / clip)
        for t, frame in enumerate(frames):
            name = f"{len(names):08d}.png"
            names.append(name)
            index.append((clip, t, len(frames)))
            jobs.append((root / clip / frame, flat / name))
    with ThreadPoolExecutor(64) as pool:
        list(pool.map(lambda job: shutil.copyfile(*job), jobs))
    (flat.parent / "list.txt").write_text("\n".join(names) + "\n")
    copied = time.monotonic()

    out_dir = Path("/tmp/alphapose-out")
    command = ["python", "scripts/demo_inference.py", "--cfg", POSE_CFG,
               "--checkpoint", str(weights_dir / "halpe136_fast50_regression_256x192.pth"),
               "--indir", str(flat), "--list", str(flat.parent / "list.txt"), "--outdir", str(out_dir),
               "--detbatch", "16", "--posebatch", "128", "--gpus", "0", "--sp"]
    proc = subprocess.run(command, cwd=ALPHAPOSE, capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"AlphaPose exited {proc.returncode}: {proc.stdout[-2000:]} {proc.stderr[-4000:]}")
    inferred = time.monotonic()

    best = {}
    for person in json.loads((out_dir / "alphapose-results.json").read_text()):
        if person["image_id"] not in best or person["score"] > best[person["image_id"]]["score"]:
            best[person["image_id"]] = person
    scale = np.array([RESIZED_SIZE[0] / ORIGINAL_SIZE[0], RESIZED_SIZE[1] / ORIGINAL_SIZE[1], 1.0], dtype=np.float32)
    arrays = {clip: np.zeros((n, 136, 3), dtype=np.float32) for clip, _, n in index}
    for name, (clip, t, _) in zip(names, index):
        if name in best:
            arrays[clip][t] = np.array(best[name]["keypoints"], dtype=np.float32).reshape(136, 3) * scale
    for clip, array in arrays.items():
        target = out_root / f"{clip}.npy"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, array)
    record = {
        "dataset": dataset, "shard": shard, "num_shards": num_shards, "clips": len(clips), "frames": len(names),
        "frames_without_detection": len(names) - len(best),
        "alphapose_commit": _out("git", "-C", ALPHAPOSE, "rev-parse", "HEAD"),
        "pose_cfg": POSE_CFG, "weights_sha256": weights_sha256, "command": command,
        "started_at_utc": started_at, "finished_at_utc": _now(),
        "seconds": {"copy": round(copied - started), "alphapose": round(inferred - copied), "total": round(time.monotonic() - started)},
        "frames_per_second": round(len(names) / max(inferred - copied, 1e-9), 1),
        **_environment(ALPHAPOSE),
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(record, indent=2))
    RESULTS.commit()
    return record


@APP.function(image=IMAGE, cpu=8, memory=32768, ephemeral_disk=512 * 1024, volumes=VOLUMES, timeout=4 * 3600)
def pack_keypoints(dataset: str, num_shards: int):
    """Checks every clip has one keypoint row per frame and archives the keypoint tree."""
    import numpy as np

    tar = _data_path(dataset, "keypoints-halpe136")
    manifest_path = Path(f"{tar}.json")
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    RESULTS.reload()
    out_root = Path(f"/results/keypoints/{dataset}")
    shards = [json.loads((out_root / "_shards" / f"{i:03d}-of-{num_shards:03d}.json").read_text()) for i in range(num_shards)]
    root = Path(DATASETS_INFO[dataset]["source"]) / "features/fullFrame-210x260px"
    mismatched, frames, undetected = [], 0, 0
    for clip in _clip_dirs(root):
        array = np.load(out_root / f"{clip}.npy")
        n = len(_frames(root / clip))
        frames += n
        undetected += int((array[:, :, 2].sum(axis=1) == 0).sum())
        if array.shape != (n, 136, 3):
            mismatched.append(clip)
    if mismatched:
        raise RuntimeError(f"{len(mismatched)} clips have the wrong keypoint shape, e.g. {mismatched[:5]}")
    partial_tar = Path(f"{tar}.partial")
    subprocess.run(["tar", "--sort=name", "--owner=0", "--group=0", "--numeric-owner", "--mtime=@0", "--exclude=_shards",
                    "-cf", str(partial_tar), "-C", str(out_root), "."], check=True)
    manifest = {
        "frames": frames, "frames_without_detection": undetected,
        "shard_seconds_alphapose": sum(s["seconds"]["alphapose"] for s in shards),
        "alphapose_commit": shards[0]["alphapose_commit"], "pose_cfg": shards[0]["pose_cfg"],
        "weights_sha256": shards[0]["weights_sha256"],
        "bytes": partial_tar.stat().st_size, "sha256": _sha256(partial_tar), "created_at_utc": _now(),
    }
    partial_tar.rename(tar)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    RESULTS.commit()
    return manifest


# --------------------------------------------------------------------------- training

def _stage(dataset):
    """Unpacks frames and keypoints onto local disk where upstream's config looks for them."""
    staged = Path(DATASETS_INFO[dataset]["staged"])
    record = {}
    started = time.monotonic()
    for kind, target in (("fullFrame-256x256px", staged), ("keypoints-halpe136", staged / "keypoints-halpe136")):
        manifest = json.loads(Path(f"{_data_path(dataset, kind)}.json").read_text())
        if kind == "fullFrame-256x256px" and not manifest["checks_passed"]:
            raise RuntimeError(f"frame archive failed its checks: {manifest['checks']}")
        if not target.exists() or kind == "fullFrame-256x256px" and not (target / "features").exists():
            target.mkdir(parents=True, exist_ok=True)
            subprocess.run(["tar", "-xf", str(_data_path(dataset, kind)), "-C", str(target)], check=True)
        record[f"{kind}_tar_sha256"] = manifest["sha256"]
    record["stage_seconds"] = round(time.monotonic() - started, 1)
    return record


def _keep_checkpoints(work_dir):
    """Deletes epoch checkpoints except the two newest and the best-dev one (earliest on ties)."""
    checkpoints = _epoch_checkpoints(work_dir)
    if not checkpoints:
        return
    keep = set(sorted(checkpoints)[-2:])
    history = _dev_history(work_dir)
    with_checkpoint = {e: w for e, w in history.items() if e in checkpoints}
    if with_checkpoint:
        keep.add(min(with_checkpoint, key=lambda e: (with_checkpoint[e], e)))
    for epoch, path in checkpoints.items():
        if epoch not in keep:
            path.unlink(missing_ok=True)


def _run_upstream(command, work_dir, prune=False):
    """Runs one upstream command, keeping its log and committing outputs as it goes.

    Upstream main.py asks on stdin whether to wipe an existing work_dir; the
    answer is always no, so a resumed run keeps its checkpoints and logs.
    """
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    peak = {"mib": 0}

    def commit_periodically():
        while not stop.wait(300):
            if prune:
                _keep_checkpoints(work_dir)
            RESULTS.commit()

    def sample_memory():
        while not stop.wait(5):
            used = _out("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits").split()
            if used:
                peak["mib"] = max(peak["mib"], int(used[0]))

    threads = [threading.Thread(target=f, daemon=True) for f in (commit_periodically, sample_memory)]
    for thread in threads:
        thread.start()
    started_at, started = _now(), time.monotonic()
    with open(Path(work_dir) / "stdout.log", "a") as log:
        log.write(f"\n### {started_at} {' '.join(command)}\n")
        log.flush()
        proc = subprocess.run(command, cwd=SC2SLR, input="n\n", text=True, stdout=log, stderr=subprocess.STDOUT)
    stop.set()
    for thread in threads:
        thread.join()
    if prune:
        _keep_checkpoints(work_dir)
    RESULTS.commit()
    log_text = (Path(work_dir) / "stdout.log").read_text(errors="replace")
    return {
        # seq_eval() swallows scoring errors and logs a WER of 100 instead.
        "scoring_failed": "Unexpected error" in log_text,
        "nan_loss_skips_in_log": log_text.count("loss is nan"),
        "command": command,
        "cwd": SC2SLR,
        "exit_code": proc.returncode,
        "started_at_utc": started_at,
        "finished_at_utc": _now(),
        "duration_seconds": round(time.monotonic() - started, 1),
        "peak_gpu_memory_mib_nvidia_smi": peak["mib"],
    }


def _history(work_dir, split):
    """Upstream seq_eval logs '[ time ] Epoch N, dev 21.3' to <split>.txt; a resumed epoch overwrites its entry."""
    path = Path(work_dir) / f"{split}.txt"
    history = {}
    for line in path.read_text().splitlines() if path.exists() else []:
        match = re.search(r"Epoch (\d+), \w+ ([\d.]+)", line)
        if match:
            history[int(match.group(1))] = float(match.group(2))
    return history


def _dev_history(work_dir):
    return _history(work_dir, "dev")


def _epoch_checkpoints(work_dir):
    found = {}
    for path in Path(work_dir).glob("dev_*_epoch*_model.pt"):
        found[int(re.search(r"_epoch(\d+)_model\.pt$", path.name).group(1))] = path
    return found


def _command(dataset, work_dir, *extra, workers=14):
    # Paper Sec. 4.2: mini-batch 4, 120 epochs; everything else from configs/baseline.yaml.
    return ["python", "main.py", "--config", "./configs/baseline.yaml", "--dataset", dataset, "--device", "0",
            "--num-worker", str(workers), "--batch-size", "4", "--save-interval", "1", "--work-dir", work_dir, *extra]


def _write_preflight_subset(dataset, n_train=48, n_long=8, n_eval=24):
    """A '<dataset>-preflight' dataset for upstream main.py: real clips, subset lists.

    The longest training clips are included so peak memory is representative.
    """
    import numpy as np

    base = Path(SC2SLR)
    name = f"{dataset}-preflight"
    target = base / f"preprocess/{name}"
    target.mkdir(parents=True, exist_ok=True)
    evaluation = base / "evaluation/preflight"
    evaluation.mkdir(parents=True, exist_ok=True)
    for script in ("preprocess.sh", "mergectmstm.py"):
        shutil.copy(base / "evaluation/slr_eval" / script, evaluation / script)
    chosen = {}
    for split in SPLITS:
        info = np.load(base / f"preprocess/{dataset}/{split}_info.npy", allow_pickle=True).item()
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
        stm = (base / f"evaluation/slr_eval/{dataset}-groundtruth-{split}.stm").read_text().splitlines(keepends=True)
        (evaluation / f"{dataset}-groundtruth-{split}.stm").write_text("".join(l for l in stm if l.split(" ")[0] in ids))
        chosen[split] = {"clips": len(picked), "frames": sum(c["num_frames"] for c in picked)}
    dataset_config = (base / f"configs/{dataset}.yaml").read_text()
    (base / f"configs/{name}.yaml").write_text(dataset_config.replace("./evaluation/slr_eval", "./evaluation/preflight"))
    return name, chosen


@APP.function(image=IMAGE, gpu=GPU, cpu=16, memory=131072, ephemeral_disk=512 * 1024, volumes=VOLUMES, timeout=4 * 3600)
def preflight(run_id: str, dataset: str):
    """Real clips through upstream main.py: train an epoch, resume for a second, then test."""
    work_dir = f"/results/runs/{run_id}/"
    staged = _stage(dataset)
    name, subset = _write_preflight_subset(dataset)
    record = {"subset": subset, **staged, **_environment(SC2SLR), "steps": []}
    steps = (
        lambda: ["--num-epoch", "1"],
        lambda: ["--num-epoch", "2", "--load-checkpoints", str(_epoch_checkpoints(work_dir)[0])],
        lambda: ["--phase", "test", "--load-weights", str(_epoch_checkpoints(work_dir)[1])],
    )
    for extra in steps:
        step = _run_upstream(_command(name, work_dir, *extra()), work_dir)
        record["steps"].append(step)
        if step["exit_code"]:
            return record
    log = (Path(work_dir) / "log.txt").read_text()
    record["epoch_seconds"] = [int(m) * 60 + int(s) for m, s in re.findall(r"Epoch \d+ costs (\d+) mins (\d+) seconds", log)]
    record["train_clips_per_epoch"] = subset["train"]["clips"]
    record["dev_wer_history"] = _dev_history(work_dir)
    record["test_phase_wer"] = {split: _history(work_dir, split) for split in ("dev", "test")}
    record["checkpoint_bytes"] = _epoch_checkpoints(work_dir)[1].stat().st_size
    return record


DIAGNOSE_DRIVER = """
import runpy, sys
sys.path.insert(0, ".")  # the driver lives in /tmp; upstream modules are in the cwd
import torch, seq_scripts, slr_network
from torch.cuda.amp import GradScaler


class LoggingScaler(GradScaler):
    # Logs every step's loss, whether its gradients were non-finite, and the AMP scale.
    step_index = 0

    def scale(self, outputs):
        self.last_loss = outputs.item()
        return super().scale(outputs)

    def update(self, *args, **kwargs):
        found = sum(v.item() for state in self._per_optimizer_states.values()
                    for v in state["found_inf_per_device"].values())
        super().update(*args, **kwargs)
        LoggingScaler.step_index += 1
        print(f"DIAG step {LoggingScaler.step_index} loss {self.last_loss:.3f} nonfinite_grad {int(found)} scale {self.get_scale()}", flush=True)


seq_scripts.GradScaler = LoggingScaler
if sys.argv[1] == "ctc-without-cudnn":
    criterion_init = slr_network.SLRModel.criterion_init

    def patched(self):
        loss = criterion_init(self)
        ctc = loss["CTCLoss"]

        def ctc_loss_without_cudnn(*args):
            with torch.backends.cudnn.flags(enabled=False):
                return ctc(*args)

        loss["CTCLoss"] = ctc_loss_without_cudnn
        return loss

    slr_network.SLRModel.criterion_init = patched
sys.argv = ["main.py", *sys.argv[2:]]
runpy.run_path("main.py", run_name="__main__")
"""


@APP.function(image=IMAGE, gpu=GPU, cpu=16, memory=131072, ephemeral_disk=512 * 1024, volumes=VOLUMES, timeout=2 * 3600)
def diagnose(run_id: str, dataset: str, variant: str, n_train: int = 800):
    """One epoch on n_train real training clips, logging loss, non-finite gradients
    and AMP scale per step; variant is 'as-released' or 'ctc-without-cudnn'."""
    work_dir = f"/results/runs/{run_id}/"
    staged = _stage(dataset)
    name, subset = _write_preflight_subset(dataset, n_train=n_train)
    driver = Path("/tmp/diagnose_driver.py")
    driver.write_text(DIAGNOSE_DRIVER)
    command = ["python", str(driver), variant, *_command(name, work_dir, "--num-epoch", "1", "--log-interval", "50")[2:]]
    record = {"variant": variant, "subset": subset, **staged, **_environment(SC2SLR),
              **_run_upstream(command, work_dir)}
    steps = re.findall(r"DIAG step (\d+) loss (\S+) nonfinite_grad (\d) scale (\S+)",
                       (Path(work_dir) / "stdout.log").read_text())
    record["steps_logged"] = len(steps)
    record["nonfinite_grad_steps"] = sum(int(s[2]) for s in steps)
    record["nan_loss_skips"] = record.pop("nan_loss_skips_in_log")
    record["final_scale"] = float(steps[-1][3]) if steps else None
    record["loss_by_50_steps"] = [round(sum(float(s[1]) for s in steps[i:i + 50]) / len(steps[i:i + 50]), 1)
                                  for i in range(0, len(steps), 50)]
    return record


@APP.function(image=IMAGE, gpu=GPU, cpu=16, memory=131072, ephemeral_disk=512 * 1024, volumes=VOLUMES,
              timeout=24 * 3600, retries=modal.Retries(max_retries=3, initial_delay=60.0))
def train(run_id: str, dataset: str, epochs: int = 120, chunk: int = 20):
    """Trains up to `chunk` more epochs toward `epochs`, resuming from the newest
    epoch checkpoint, then spawns itself for the next chunk.

    Chunks keep each call well under Modal's 24 h limit. Upstream's --num-epoch
    only bounds the epoch loop (the LR milestones are fixed), so stopping and
    resuming at a chunk boundary trains the same schedule. Each chunk's record
    is kept at <work_dir>/segments/.
    """
    work_dir = f"/results/runs/{run_id}/"
    RESULTS.reload()
    record = {**_stage(dataset), **_environment(SC2SLR)}
    checkpoints = _epoch_checkpoints(work_dir)
    start = max(checkpoints) + 1 if checkpoints else 0
    if start >= epochs:
        return {**record, "already_complete": True, "dev_wer_history": _dev_history(work_dir)}
    target = min(epochs, start + chunk)
    extra = ["--num-epoch", str(target)]
    if checkpoints:
        extra += ["--load-checkpoints", str(checkpoints[max(checkpoints)])]
    record.update(_run_upstream(_command(dataset, work_dir, *extra), work_dir, prune=True))
    record["dev_wer_history"] = _dev_history(work_dir)
    record["epochs"] = [start, target]
    if record["exit_code"]:
        raise RuntimeError(f"upstream training exited {record['exit_code']}")
    if target < epochs:
        record["next_call_id"] = train.spawn(run_id, dataset, epochs, chunk).object_id
    segments = Path(work_dir) / "segments"
    segments.mkdir(exist_ok=True)
    (segments / f"{start:03d}-{target:03d}.json").write_text(json.dumps(record, indent=2))
    RESULTS.commit()
    return record


@APP.function(image=IMAGE, gpu=GPU, cpu=16, memory=131072, ephemeral_disk=512 * 1024, volumes=VOLUMES, timeout=4 * 3600)
def evaluate(train_run_id: str, eval_run_id: str, dataset: str):
    """Selects the lowest-dev-WER epoch (earliest on ties) and scores dev and test once."""
    RESULTS.reload()
    train_dir = f"/results/runs/{train_run_id}/"
    checkpoints = _epoch_checkpoints(train_dir)
    history = _dev_history(train_dir)
    best = min(history, key=lambda e: (history[e], e))
    if best not in checkpoints:
        raise RuntimeError(f"best-dev epoch {best} has no retained checkpoint")
    work_dir = f"/results/runs/{eval_run_id}/"
    record = {"selected_epoch": best, "selected_dev_wer_during_training": history[best],
              "checkpoint": str(checkpoints[best]), "checkpoint_sha256": _sha256(checkpoints[best]),
              **_stage(dataset), **_environment(SC2SLR)}
    record.update(_run_upstream(_command(dataset, work_dir, "--phase", "test", "--load-weights", str(checkpoints[best])), work_dir))
    record["wer"] = {split: _history(work_dir, split)[6667] for split in ("dev", "test")}
    for split in ("dev", "test"):
        hypothesis = Path(work_dir) / f"output-hypothesis-{split}.ctm"
        record[f"hypothesis_{split}_sha256"] = _sha256(hypothesis)
    return record


def _print(record):
    print(json.dumps(record, indent=2, default=str))


@APP.local_entrypoint()
def launch_prepare_frames(dataset: str):
    _print(prepare_frames.remote(dataset))


@APP.local_entrypoint()
def launch_keypoint_preflight(dataset: str, limit_clips: int = 4):
    _print(extract_keypoints.remote(dataset, 0, 1, limit_clips))


@APP.local_entrypoint()
def launch_keypoints(dataset: str, num_shards: int = 24):
    for record in extract_keypoints.starmap([(dataset, i, num_shards) for i in range(num_shards)]):
        print(record["shard"], record["frames"], record["frames_without_detection"], record["seconds"])
    _print(pack_keypoints.remote(dataset, num_shards))


@APP.local_entrypoint()
def launch_preflight(run_id: str, dataset: str):
    _print(preflight.remote(run_id, dataset))


@APP.local_entrypoint()
def launch_diagnose(run_id: str, dataset: str, variant: str):
    _print(diagnose.remote(run_id, dataset, variant))


@APP.local_entrypoint()
def launch_train(run_id: str, dataset: str, epochs: int = 120, chunk: int = 20):
    call = train.spawn(run_id, dataset, epochs, chunk)
    print(call.object_id)


@APP.local_entrypoint()
def launch_evaluate(train_run_id: str, eval_run_id: str, dataset: str):
    _print(evaluate.remote(train_run_id, eval_run_id, dataset))
