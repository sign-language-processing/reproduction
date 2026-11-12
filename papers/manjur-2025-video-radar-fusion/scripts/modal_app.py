"""Modal entry points for the Manjur et al. (2025) reproduction.

Run only through the repository wrapper, always in the Restricted Environment:
  .agents/skills/reproduce-paper/scripts/modal_repro_sign.sh run --env "$MODAL_ENV" \
      papers/manjur-2025-video-radar-fusion/scripts/modal_app.py::data --mode populate
"""
from pathlib import Path

import modal

HERE = Path(__file__).parent
HF_ENV = {"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub"}

app = modal.App("manjur-2025-radar-fusion")
# Restricted data and everything derived from it stay in environment $MODAL_ENV.
datasets = modal.Volume.from_name("datasets", version=2)
results = modal.Volume.from_name("manjur-2025-radar-fusion-results", version=2)
# The shared Hugging Face cache lives in environment main.
hf_cache = modal.Volume.from_name("huggingface-cache", environment_name="main", version=2)
kaggle = modal.Secret.from_name("kaggle-credentials", required_keys=["KAGGLE_API_TOKEN"])
# REPRO-SIGN shares 50 concurrent Modal GPUs; this paper may use at most 30 at once across
# all its jobs. Each GPU function is capped here; never launch two GPU fan-outs together.
MAX_GPUS = 30

UPSTREAM = "https://github.com/impressncsu/ICCV2025_MSLR.git"
UPSTREAM_REV = "7dd73c4e67b2959a95f828c103269723528b9271"
# The upstream scripts find their project root by this directory name.
REPO = "/opt/ICCV_MSLR_2025"
BASE_IMAGE = (
    "ghcr.io/sign-language-processing/reproduction"
    "@sha256:305b6165d306192996358ca312d9a751fa409f43063a76dc7758880a8f905291"
)


def with_upstream(image: modal.Image) -> modal.Image:
    return (
        image.run_commands(f"git clone {UPSTREAM} {REPO}", f"git -C {REPO} checkout {UPSTREAM_REV}")
        .add_local_dir(HERE.parent / "patches", "/opt/patches", copy=True)
        .run_commands(f"for p in /opt/patches/*.patch; do git -C {REPO} apply \"$p\"; done")
        .env(HF_ENV)
    )


# Radar and video models are Keras 2 code (K.abs Lambda layers, .h5 checkpoints):
# TensorFlow 2.15 is the last release with Keras 2 as tf.keras and needs Python <= 3.11,
# so it cannot share the Python 3.12 NGC PyTorch base image.
tf_image = with_upstream(
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "tensorflow[and-cuda]==2.15.1",
        "numpy==1.26.4",
        "opencv-python-headless==4.10.0.84",
        "h5py==3.11.0",
        "pandas==2.2.2",
        "scikit-learn==1.5.2",
        "matplotlib==3.9.2",
        "openpyxl==3.1.5",
    )
)

# HRNet pose extraction and SL-GCN (PyTorch) use the project base image.
HRNET_URL = (
    "https://download.openmmlab.com/mmpose/top_down/hrnet/"
    "hrnet_w48_coco_wholebody_384x288-6e061c6a_20200922.pth"
)
HRNET_SHA256 = "6e061c6a16ef994b1dc0a69d1d22473dbb8673afda4effca3baef0bd4c1e6976"
SAM_SLR = "https://github.com/jackyjsy/CVPR21Chal-SLR.git"
SAM_SLR_REV = "de6c53a6fbbc08a5d9a728e33614a4541836714c"  # cited SL-GCN basis: training configs
torch_image = with_upstream(
    modal.Image.from_registry(BASE_IMAGE).pip_install(
        "yacs==0.1.8", "natsort==8.4.0", "opencv-python-headless==4.11.0.86", "scikit-learn==1.5.2"
    )
).run_commands(
    f"git clone {SAM_SLR} /opt/sam-slr && git -C /opt/sam-slr checkout {SAM_SLR_REV}",
    # demo.py loads the checkpoint by this file name from its working directory.
    f"curl -fsSL -o {REPO}/Pose_estimation/{HRNET_URL.rsplit('/', 1)[1]} {HRNET_URL}",
    f"echo '{HRNET_SHA256}  {REPO}/Pose_estimation/{HRNET_URL.rsplit('/', 1)[1]}' | sha256sum -c",
)

data_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("unzip")
    # kaggle 2.2.4 imports six without declaring it.
    .pip_install("kaggle==2.2.4", "kagglesdk==0.1.38", "six==1.17.0")
    .env(HF_ENV)
    .add_local_file(HERE / "data.sh", "/opt/data.sh")
)


@app.function(
    image=data_image,
    secrets=[kaggle],
    cpu=8,
    memory=16384,
    timeout=6 * 3600,
    volumes={"/datasets": datasets, "/cache/huggingface": hf_cache},
)
def data(mode: str = "populate"):
    """Controlled population step: the only writable mount of `datasets`."""
    import subprocess

    subprocess.run(["bash", "/opt/data.sh", mode], check=True)
    datasets.commit()


ENV_CHECK = {
    "tf": "import tensorflow as tf, keras, cv2, numpy; print('tf', tf.__version__, 'keras', keras.__version__, 'cv2', cv2.__version__, 'numpy', numpy.__version__); print('gpus', tf.config.list_physical_devices('GPU'))",
    "torch": "import torch, cv2, numpy, yacs; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cv2', cv2.__version__, 'numpy', numpy.__version__); print('gpu', torch.cuda.get_device_name(0))",
}


@app.function(image=tf_image, gpu="L4", max_containers=MAX_GPUS, timeout=600, volumes={"/cache/huggingface": hf_cache})
def env_check_tf():
    import subprocess

    subprocess.run(["python", "-c", ENV_CHECK["tf"]], check=True)
    subprocess.run(["git", "-C", REPO, "status", "--short"], check=True)


@app.function(image=torch_image, gpu="L4", max_containers=MAX_GPUS, timeout=600, volumes={"/cache/huggingface": hf_cache})
def env_check_torch():
    import subprocess

    subprocess.run(["python", "-c", ENV_CHECK["torch"]], check=True)
    subprocess.run(["git", "-C", REPO, "status", "--short"], check=True)


DATA = "/datasets/multimedalis-iccv-mslr-2025-track-2"
WORK = "/outputs/work"
cpu_image = modal.Image.debian_slim(python_version="3.11").env(HF_ENV)
read_only = {
    "/datasets": datasets.with_mount_options(read_only=True),
    "/outputs": results,
    "/cache/huggingface": hf_cache,
}


@app.function(image=cpu_image, cpu=4, memory=8192, timeout=3600, volumes=read_only)
def worktree():
    """Mirror every sample folder as real directories of symlinks to the read-only raw
    files, so the authors' preprocessing can write its outputs beside them. Also writes
    the train label CSV (SAMPLE_x,label) and val/test id lists. Idempotent."""
    import json
    import os
    from pathlib import Path

    inv = json.loads(Path(DATA, "INVENTORY.json").read_text())
    raw = Path(DATA, "raw")
    work = Path(WORK)
    links = 0
    lists = {"train": [], "val": [], "test": []}
    for split in ("train", "val", "test"):
        src_root = raw / inv["split_dirs"][split]
        if split == "train":
            samples = [(c.name, s) for c in sorted(src_root.iterdir()) for s in sorted(c.iterdir())]
        else:
            samples = [(None, s) for s in sorted(src_root.iterdir())]
        for cls, sample in samples:
            dest = work / split / cls / sample.name if cls else work / split / sample.name
            dest.mkdir(parents=True, exist_ok=True)
            for f in sorted(sample.iterdir()):
                link = dest / f.name
                if not link.is_symlink():
                    link.symlink_to(f)
                    links += 1
                if f.suffix.lower() == ".mkv":
                    rgb = work / "rgb" / split / f"{sample.name}.mkv"
                    rgb.parent.mkdir(parents=True, exist_ok=True)
                    if not rgb.is_symlink():
                        rgb.symlink_to(f)
                        links += 1
            lists[split].append(f"{sample.name},{int(cls.split('_')[0])}" if cls else sample.name)
    ids = work / "ids"
    ids.mkdir(parents=True, exist_ok=True)
    (ids / "train_labels.csv").write_text("\n".join(lists["train"]) + "\n")
    (ids / "val.txt").write_text("\n".join(lists["val"]) + "\n")
    (ids / "test.txt").write_text("\n".join(lists["test"]) + "\n")
    results.commit()
    summary = {k: len(v) for k, v in lists.items()} | {"new_links": links}
    print(json.dumps(summary))
    return summary


PREP_SCRIPTS = {
    ("radar", "train"): "train_md_gen.py",
    ("radar", "val"): "val_md_gen.py",
    ("radar", "test"): "val_md_gen.py",
    ("video", "train"): "train_video_resized.py",
    ("video", "val"): "val_video_resized.py",
    ("video", "test"): "val_video_resized.py",
}


@app.function(image=tf_image, cpu=2, memory=8192, timeout=6 * 3600, volumes=read_only)
def prep(task: str, split: str, shard: int = 0, n_shards: int = 1, limit: int = 0):
    """Run the authors' radar (micro-Doppler PNG) or video (.h5) preprocessing script,
    unchanged except for patches/, on shard `shard` of `n_shards` of a split.
    Train is sharded by class folder, val/test by sample folder; `limit` keeps only the
    first N entries of the shard (preflight). Outputs land beside the symlinks in WORK."""
    import json
    import os
    import subprocess
    import tempfile
    import time
    from pathlib import Path

    src = Path(WORK, split)
    entries = sorted(src.iterdir())[shard::n_shards]
    if limit:
        entries = entries[:limit]
    tree = Path(tempfile.mkdtemp(prefix="shard-"), split)  # containers are reused between inputs
    tree.mkdir()
    for e in entries:
        (tree / e.name).symlink_to(e)
    env = dict(os.environ, MSLR_TRAIN_DIR=str(tree), MSLR_VAL_DIR=str(tree), MSLR_TEST_DIR=str(tree),
               MSLR_SPLIT="Test" if split == "test" else "Val")
    name = f"{task}-{split}-{shard:03d}of{n_shards:03d}" + (f"-limit{limit}" if limit else "")
    log = Path("/outputs/logs/prep", name + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log.open("w") as fh:
        proc = subprocess.run(["python", PREP_SCRIPTS[(task, split)]], env=env, stdout=fh, stderr=subprocess.STDOUT,
                              cwd=f"{REPO}/MSLR_radar_video/data_processing")
    seconds = time.time() - start
    suffix = ".png" if task == "radar" else ".h5"
    samples = [s for e in entries for s in (sorted(e.iterdir()) if split == "train" else [e])]
    produced = sum(1 for s in samples for f in s.iterdir() if f.name.endswith(suffix) and not f.is_symlink())
    results.commit()
    summary = {"name": name, "exit_code": proc.returncode, "seconds": round(seconds, 1),
               "samples": len(samples), "outputs": produced, "log": str(log)}
    print(json.dumps(summary))
    if proc.returncode:
        print(log.read_text()[-3000:])
    return summary


PREP_SHARDS = {"train": 126, "val": 45, "test": 76}  # train: one class folder per shard


@app.local_entrypoint()
def prep_all(task: str):
    """Fan prep() out over every shard of train, val and test; fail if any shard fails."""
    import json

    calls = [(task, split, k, n) for split, n in PREP_SHARDS.items() for k in range(n)]
    summaries = list(prep.starmap(calls))
    bad = [s for s in summaries if s["exit_code"] != 0]
    totals = {split: sum(s["outputs"] for s in summaries if s["name"].split("-")[1] == split) for split in PREP_SHARDS}
    print(json.dumps({"task": task, "shards": len(summaries), "failed": [s["name"] for s in bad],
                      "outputs": totals, "max_shard_seconds": max(s["seconds"] for s in summaries)}))
    if bad:
        raise SystemExit(f"{len(bad)} shard(s) failed")


def _gpu_sampler(stop, peaks):
    """Record peak GPU memory (MiB) reported by nvidia-smi until `stop` is set."""
    import subprocess

    while not stop.wait(5):
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.split()
        if out:
            peaks.append(int(out[0]))


@app.function(image=torch_image, gpu="L4", max_containers=MAX_GPUS, cpu=8, memory=16384, timeout=12 * 3600, volumes=read_only)
def pose(split: str, shard: int = 0, n_shards: int = 1, limit: int = 0, procs: int = 1, per_class: int = 0):
    """Run the authors' HRNet whole-body extraction (Pose_estimation/demo.py) on shard
    `shard` of `n_shards` of a split's RGB videos, as `procs` concurrent unmodified demo.py
    processes on disjoint video subsets sharing one GPU. Outputs go to
    WORK/pose/<split>/SAMPLE_x.npy; demo.py skips videos whose output exists, so reruns resume."""
    import json
    import os
    import subprocess
    import tempfile
    import threading
    import time
    from pathlib import Path

    videos = sorted(Path(WORK, "rgb", split).iterdir())[shard::n_shards]
    if per_class:  # class-balanced train subset: first per_class samples of each class folder
        keep = {s.name for c in Path(WORK, "train").iterdir() for s in sorted(c.iterdir())[:per_class]}
        videos = [v for v in videos if v.stem in keep]
    if limit:
        videos = videos[:limit]
    tmp = Path(tempfile.mkdtemp(prefix="pose-"))  # containers are reused between inputs
    inputs = [tmp / str(i) for i in range(procs)]
    for i, d in enumerate(inputs):
        d.mkdir()
        for v in videos[i::procs]:
            (d / v.name).symlink_to(v)
    out = Path(WORK, "pose", split)
    out.mkdir(parents=True, exist_ok=True)
    before = sum(1 for v in videos if (out / (v.stem + ".npy")).exists())
    name = (f"pose-{split}-{shard:03d}of{n_shards:03d}" + (f"-limit{limit}" if limit else "")
            + (f"-pc{per_class}" if per_class else "") + f"-p{procs}")
    log = Path("/outputs/logs/pose", name + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    stop, peaks = threading.Event(), []
    threading.Thread(target=_gpu_sampler, args=(stop, peaks), daemon=True).start()
    start = time.time()
    with log.open("w") as fh:
        running = [subprocess.Popen(["python", "demo.py"], cwd=f"{REPO}/Pose_estimation", stdout=fh,
                                    stderr=subprocess.STDOUT,
                                    env=dict(os.environ, POSE_INPUT_DIR=str(d), POSE_OUTPUT_DIR=str(out)))
                   for d in inputs]
        returncode = max(p.wait() for p in running)
    seconds = time.time() - start
    stop.set()
    after = sum(1 for v in videos if (out / (v.stem + ".npy")).exists())
    results.commit()
    summary = {"name": name, "exit_code": returncode, "seconds": round(seconds, 1), "videos": len(videos),
               "new_outputs": after - before, "total_outputs": after, "peak_gpu_mib": max(peaks, default=None),
               "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                     capture_output=True, text=True).stdout.strip()}
    print(json.dumps(summary))
    if returncode:
        print(log.read_text()[-3000:])
    return summary


TF_TRAIN = {"radar": "md_cnn_3branch.py", "video": "resnet21D_V2.py"}


def _subset_tree(per_class: int):
    """Directory of class folders holding symlinks to the first `per_class` sample folders
    of each training class (0 = all), for the authors' training scripts."""
    import tempfile
    from pathlib import Path

    tree = Path(tempfile.mkdtemp(prefix="train-"))
    for cls in sorted(Path(WORK, "train").iterdir()):
        (tree / cls.name).mkdir()
        samples = sorted(cls.iterdir())
        for s in samples[:per_class] if per_class else samples:
            (tree / cls.name / s.name).symlink_to(s)
    return tree


# 64 GiB: resnet21D_V2.py keeps a shuffle buffer of the whole training set (~28 GB) in RAM.
@app.function(image=tf_image, gpu="L4", max_containers=MAX_GPUS, cpu=4, memory=65536, timeout=24 * 3600, volumes=read_only)
def tf_train(model: str, run_id: str, per_class: int = 0, max_seconds: int = 0):
    """Run the authors' radar or video training script unchanged in /outputs/runs/<run_id>
    (its checkpoints and final files land there). per_class/max_seconds bound a preflight."""
    import json
    import os
    import subprocess
    import threading
    import time
    from pathlib import Path

    run = Path("/outputs/runs", run_id)
    run.mkdir(parents=True, exist_ok=False)
    tree = _subset_tree(per_class)
    cmd = ["python", f"{REPO}/MSLR_radar_video/cnn_models_radar+video/{TF_TRAIN[model]}"]
    if max_seconds:
        cmd = ["timeout", str(max_seconds)] + cmd
    stop, peaks = threading.Event(), []
    threading.Thread(target=_gpu_sampler, args=(stop, peaks), daemon=True).start()
    start = time.time()
    with (run / "stdout.log").open("w") as fh:
        proc = subprocess.run(cmd, cwd=run, env=dict(os.environ, MSLR_TRAIN_DIR=str(tree)),
                              stdout=fh, stderr=subprocess.STDOUT)
    seconds = time.time() - start
    stop.set()
    log = (run / "stdout.log").read_text(errors="replace")
    (run / "pip-freeze.txt").write_text(subprocess.run(["pip", "freeze"], capture_output=True, text=True).stdout)
    summary = {"run_id": run_id, "model": model, "exit_code": proc.returncode, "seconds": round(seconds, 1),
               "per_class": per_class, "epochs_started": log.count("\nEpoch "),
               "files": sorted(str(p.relative_to(run)) for p in run.rglob("*.h5")) + sorted(str(p.relative_to(run)) for p in run.glob("*.xlsx")),
               "peak_gpu_mib": max(peaks, default=None), "app_id": app.app_id,
               "function_call_id": modal.current_function_call_id()}
    (run / "execution.json").write_text(json.dumps(summary, indent=2))
    results.commit()
    print(json.dumps(summary))
    print(log[-2500:])
    return summary


@app.function(image=tf_image, gpu="L4", max_containers=MAX_GPUS, cpu=8, memory=32768, timeout=6 * 3600, volumes=read_only)
def tf_predict(split: str, radar_h5: str, video_h5: str, out_id: str, limit: int = 0, ids_csv: str = ""):
    """Run the authors' val_fusion_pred.py / test_fusion_pred.py with the given trained
    models installed as final_weights/{radar,video}_final.h5; copy results to /outputs/runs/<out_id>."""
    import json
    import os
    import shutil
    import subprocess
    import tempfile
    import time
    from pathlib import Path

    mslr = Path(REPO, "MSLR_radar_video")
    (mslr / "final_weights").mkdir(exist_ok=True)
    shutil.copy(radar_h5, mslr / "final_weights" / "radar_final.h5")
    shutil.copy(video_h5, mslr / "final_weights" / "video_final.h5")
    shutil.rmtree(mslr / "results", ignore_errors=True)  # keep only this call's outputs
    samples = sorted(Path(WORK, split).iterdir())
    if ids_csv:  # restrict to the ids of another output (preflight alignment)
        import csv
        keep = {f"SAMPLE_{r['id']}" for r in csv.DictReader(open(ids_csv))}
        samples = [x for x in samples if x.name in keep]
    tree = Path(tempfile.mkdtemp(prefix=f"{split}-"))
    for s in samples[:limit] if limit else samples:
        (tree / s.name).symlink_to(s)
    out = Path("/outputs/runs", out_id)
    out.mkdir(parents=True, exist_ok=False)
    script = "val_fusion_pred.py" if split == "val" else "test_fusion_pred.py"
    start = time.time()
    with (out / "stdout.log").open("w") as fh:
        proc = subprocess.run(["python", script], cwd=mslr / "prediction_script", stdout=fh, stderr=subprocess.STDOUT,
                              env=dict(os.environ, MSLR_VAL_DIR=str(tree), MSLR_TEST_DIR=str(tree)))
    seconds = time.time() - start
    if (mslr / "results").exists():
        shutil.copytree(mslr / "results", out / "results")
    import hashlib

    summary = {"out_id": out_id, "split": split, "exit_code": proc.returncode, "seconds": round(seconds, 1),
               "samples": len(list(tree.iterdir())), "radar_h5": radar_h5, "video_h5": video_h5,
               "radar_h5_sha256": hashlib.sha256(Path(radar_h5).read_bytes()).hexdigest(),
               "video_h5_sha256": hashlib.sha256(Path(video_h5).read_bytes()).hexdigest(),
               "outputs": sorted(str(p.relative_to(out)) for p in (out / "results").rglob("*") if p.is_file())
               if (out / "results").exists() else [], "app_id": app.app_id,
               "function_call_id": modal.current_function_call_id()}
    (out / "execution.json").write_text(json.dumps(summary, indent=2))
    results.commit()
    print(json.dumps(summary))
    if proc.returncode:
        print((out / "stdout.log").read_text(errors="replace")[-3000:])
    return summary


SLGCN = f"{REPO}/SL-GCN"
STREAMS = {"joint": "joint", "bone": "bone", "jm": "joint_motion", "bm": "bone_motion"}  # authors' tag -> data part
FOLD_SEED = 1  # inferred: fold assignment is unpublished; seed 1 matches main.py's init_seed


def _slgcn_dataset(root, name, pose_dir, labels=None):
    """Authors' generators: joint (sign_gendata_test.py, 27 keypoints), then bone and motion,
    into <root>/<name>/data/sign/27/testing_data/test_data_{joint,bone,joint_motion,bone_motion}.npy."""
    import subprocess
    from pathlib import Path

    base = Path(root, name)
    data = base / "data/sign/27/testing_data"
    data.mkdir(parents=True)
    (base / "data_gen").mkdir()
    label_path = "None"
    if labels is not None:
        label_path = str(base / "labels.csv")
        Path(label_path).write_text("".join(f"{n},{y}\n" for n, y in labels))
    with (base / "gendata.log").open("w") as fh:
        for cmd, cwd in (
            (["python", "sign_gendata_test.py", "--data_path", str(pose_dir), "--label_path", label_path,
              "--out_folder", str(data), "--points", "27"], f"{SLGCN}/data_gen"),
            (["python", f"{SLGCN}/data_gen/gen_bone_data_test.py"], base / "data_gen"),
            (["python", f"{SLGCN}/data_gen/gen_motion_data_test.py"], base / "data_gen"),
        ):
            subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, check=True)
    return data


@app.function(image=torch_image, cpu=8, memory=32768, timeout=6 * 3600, volumes=read_only)
def slgcn_data(run_id: str, folds: int = 5, replicate: int = 1):
    """Build SL-GCN inputs: per fold, the stratified train/held-out parts of the labelled
    training samples that have keypoints; plus all val and test keypoints (dummy labels).
    replicate > 1 repeats the labelled samples, only to benchmark full-size throughput."""
    import json
    from pathlib import Path

    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    root = Path("/outputs/runs", run_id, "slgcn_data")
    root.mkdir(parents=True, exist_ok=False)
    pose = Path(WORK, "pose")
    labels = [l.split(",") for l in Path(WORK, "ids/train_labels.csv").read_text().split()]
    labels = [(n, int(y)) for n, y in labels if (pose / "train" / f"{n}.npy").exists()] * replicate
    names = np.array([n for n, _ in labels]); ys = np.array([y for _, y in labels])
    summary = {"train_with_keypoints": len(labels), "folds": {}}
    for k, (tr, ho) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=FOLD_SEED).split(names, ys)):
        _slgcn_dataset(root, f"fold{k}_train", pose / "train", list(zip(names[tr], ys[tr])))
        _slgcn_dataset(root, f"fold{k}_heldout", pose / "train", list(zip(names[ho], ys[ho])))
        summary["folds"][k] = {"train": int(len(tr)), "heldout": int(len(ho))}
    for split in ("val", "test") if replicate == 1 else ():
        if (pose / split).exists() and any((pose / split).iterdir()):
            _slgcn_dataset(root, split, pose / split)
            summary[split] = sum(1 for _ in (pose / split).iterdir())
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    results.commit()
    print(json.dumps(summary))
    return summary


def _slgcn_workspace(run_dir):
    """Point SL-GCN's hardcoded ./work_dir and ./save_models into the results Volume."""
    import os
    from pathlib import Path

    for d in ("work_dir", "save_models"):
        Path(run_dir, d).mkdir(parents=True, exist_ok=True)
        link = Path(SLGCN, d)
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(Path(run_dir, d), link)


# 4 cores / 16 GiB: measured as fast as 8 / 32 (bench-slgcn-cpu4-001); one job per L4 (bench-slgcn-pack2-001).
@app.function(image=torch_image, gpu="L4", max_containers=MAX_GPUS, cpu=4, memory=16384, timeout=24 * 3600, volumes=read_only)
def slgcn_train(run_id: str, jobs: str, num_epoch: int = 250, data_run: str = "", resume: bool = False,
                authors_is_vector: bool = False):
    """Train SL-GCN stream x fold jobs ("joint:0,bone:0") concurrently on one GPU, each with
    SAM-SLR's published training config; substitutions only: 126 classes, data paths, one
    GPU, the authors' experiment naming (sign_<stream>_fold<k>_250) and a checkpoint every
    epoch. data_run reads the fold data of another run (benchmarks, retrains on the same folds).
    authors_is_vector=True sets is_vector False in both feeders, the value in the authors' fold
    configs (gate slgcn-retrain-settings, R1); SAM-SLR's train configs set True for bone/jm/bm.

    resume=True continues each job after an interruption with main.py's own `weights` and
    `start_epoch` options, from the last epoch that has both a loadable checkpoint and its
    held-out eval record. Model weights and the epoch-based LR schedule carry over; SGD
    momentum buffers and the data-order RNG restart. (The authors' main.py checks for a
    save_models/<exp>/epoch directory before its delete prompt; that path is a file prefix,
    so the prompt never fires.)"""
    import json
    import re
    import subprocess
    import threading
    import time
    from pathlib import Path

    import torch
    import yaml

    run = Path("/outputs/runs", run_id)
    data = Path("/outputs/runs", data_run or run_id, "slgcn_data")
    _slgcn_workspace(run)
    (run / "configs").mkdir(parents=True, exist_ok=True)
    (run / "logs").mkdir(parents=True, exist_ok=True)
    procs, resumed = [], {}
    for job in jobs.split(","):
        stream, fold = job.split(":")
        part = STREAMS[stream]
        cfg = yaml.safe_load(Path(f"/opt/sam-slr/SL-GCN/config/sign/train/train_{part}.yaml").read_text())
        exp = f"sign_{stream}_fold{fold}_250"
        cfg.update(Experiment_name=exp, device=[0], num_epoch=num_epoch, save_interval=1)
        cfg["model_args"]["num_class"] = 126
        for key, sub in (("train_feeder_args", f"fold{fold}_train"), ("test_feeder_args", f"fold{fold}_heldout")):
            d = data / sub / "data/sign/27/testing_data"
            cfg[key].update(data_path=str(d / f"test_data_{part}.npy"), label_path=str(d / "test_label.pkl"))
            if authors_is_vector:
                cfg[key]["is_vector"] = False
        cfg_path = run / "configs" / f"{exp}.yaml"
        if resume:
            evaluated = {int(m.group(1)) for f in (run / "work_dir" / exp / "eval_results").glob("epoch_*.pkl")
                         if (m := re.match(r"epoch_(\d+)_", f.name))}
            for last in sorted(evaluated, reverse=True):
                ckpt = run / "save_models" / exp / f"epoch-{last}.pt"
                try:
                    n_tensors = len(torch.load(ckpt, map_location="cpu"))
                    break
                except Exception as e:  # missing or truncated checkpoint: step back one epoch
                    print(f"{exp}: {ckpt.name} unusable ({e!r})")
            else:
                raise RuntimeError(f"{exp}: no loadable checkpoint with an eval record to resume from")
            cfg.update(weights=str(ckpt), start_epoch=last + 1)
            resumed[exp] = {"from_checkpoint": f"save_models/{exp}/{ckpt.name}", "start_epoch": last + 1,
                            "checkpoint_tensors": n_tensors, "epochs_evaluated_before": len(evaluated)}
            print(json.dumps({exp: resumed[exp]}))
            cfg_path = run / "configs" / f"{exp}.resume-from-epoch-{last}.yaml"
        elif any((run / "save_models" / exp).glob("*.pt")):
            raise RuntimeError(f"{exp}: checkpoints already exist; pass resume=True to continue them")
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        fh = (run / "logs" / f"{exp}.log").open("a" if resume else "w")
        procs.append((exp, subprocess.Popen(["python", "main.py", "--config", str(cfg_path)], cwd=SLGCN,
                                            stdout=fh, stderr=subprocess.STDOUT)))
    stop, peaks = threading.Event(), []
    threading.Thread(target=_gpu_sampler, args=(stop, peaks), daemon=True).start()
    start = time.time()
    codes = {exp: p.wait() for exp, p in procs}
    seconds = time.time() - start
    stop.set()
    summaries = []
    for exp, code in codes.items():
        evals = [(float(m.group(2)), int(m.group(1))) for f in (run / "work_dir" / exp / "eval_results").glob("epoch_*.pkl")
                 if (m := re.match(r"epoch_(\d+)_([0-9.]+)\.pkl", f.name))]
        best_acc, best_epoch = max(evals, key=lambda t: (t[0], -t[1])) if evals else (None, None)
        summary = {"run_id": run_id, "experiment": exp, "exit_code": code, "seconds": round(seconds, 1),
                   "concurrent_jobs": len(procs), "epochs_evaluated": len(evals), "best_heldout_acc": best_acc,
                   "best_epoch": best_epoch,
                   "checkpoint": f"save_models/{exp}/epoch-{best_epoch}.pt" if best_epoch is not None else None,
                   "peak_gpu_mib": max(peaks, default=None), "app_id": app.app_id,
                   "function_call_id": modal.current_function_call_id(), "resumed": resumed.get(exp)}
        (run / "logs" / f"{exp}.json").write_text(json.dumps(summary, indent=2))
        summaries.append(summary)
        if code:
            print((run / "logs" / f"{exp}.log").read_text(errors="replace")[-3000:])
    results.commit()
    print(json.dumps(summaries))
    return summaries


@app.function(image=torch_image, gpu="L4", max_containers=MAX_GPUS, cpu=8, memory=32768, timeout=6 * 3600, volumes=read_only)
def slgcn_test(run_id: str, split: str, author_weights: bool = False, vector_streams: str = "",
               data_run: str = "", stream_runs: str = ""):
    """Run the authors' 20 published fold test configs (paths substituted: data, labels,
    our selected checkpoints), then their run_fusion.py; save logit_slgcnn_<split>.csv.
    author_weights=True instead loads the authors' released save_models.zip checkpoints at
    exactly the paths the configs name (evidence check; outputs get an _author suffix).
    vector_streams ("bone,jm,bm") sets test_feeder_args.is_vector True for those streams, the
    normalization SAM-SLR's train and test configs use and our models were trained with; the
    authors' test configs leave it False (outputs get a _vec-<streams> suffix).
    data_run reads the val/test arrays of another run; stream_runs ("joint=slgcn-full-001")
    takes a stream's fold summaries and checkpoints from another run. Outputs go to run_id."""
    import json
    import shutil
    import subprocess
    from pathlib import Path

    import yaml

    run = Path("/outputs/runs", run_id)
    data = Path("/outputs/runs", data_run or run_id, "slgcn_data", split, "data/sign/27/testing_data")
    sources = dict(x.split("=") for x in filter(None, stream_runs.split(",")))
    assert set(sources) <= set(STREAMS), sources
    vec = sorted(filter(None, vector_streams.split(",")))
    assert set(vec) <= set(STREAMS), vec
    tag = ("_author" if author_weights else "") + (f"_vec-{'-'.join(vec)}" if vec else "")
    test_dir = run / f"test_{split}{tag}"
    test_dir.mkdir(parents=False, exist_ok=False)
    _slgcn_workspace(test_dir)
    if author_weights:  # unpack into the workspace's save_models/, where the configs point
        import zipfile

        with zipfile.ZipFile("/outputs/author_weights/save_models.zip") as z:
            for m in z.namelist():
                if m.endswith(".pt"):
                    target = test_dir / m  # save_models/<exp>/epoch-<e>.pt
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(z.read(m))
    fallbacks = []
    with (test_dir / "test.log").open("w") as fh:
        for cfg_file in sorted(Path(SLGCN, "config/sign/test/fold_check").rglob("*.yaml")):
            cfg = yaml.safe_load(cfg_file.read_text())
            part = Path(cfg["test_feeder_args"]["data_path"]).name.replace("test_data_", "").replace(".npy", "")
            stream, fold = cfg["Experiment_name"].split("_")[1], cfg["Experiment_name"].split("fold")[1]
            for key in ("train_feeder_args", "test_feeder_args"):
                cfg[key].update(data_path=str(data / f"test_data_{part}.npy"), label_path=str(data / "test_label.pkl"))
            if stream in vec:
                cfg["test_feeder_args"]["is_vector"] = True
            if author_weights:
                cfg["weights"] = str(test_dir / cfg["weights"])  # the published relative path, unchanged
            else:
                src = Path("/outputs/runs", sources.get(stream, run_id))
                trained = json.loads((src / "logs" / f"sign_{stream}_fold{fold}_250.json").read_text())
                cfg["weights"] = str(src / trained["checkpoint"])
            out = test_dir / "configs" / cfg_file.name
            out.parent.mkdir(exist_ok=True)
            out.write_text(yaml.safe_dump(cfg, sort_keys=False))
            subprocess.run(["python", "main.py", "--config", str(out)], cwd=SLGCN, stdout=fh,
                           stderr=subprocess.STDOUT, check=True)
            # main.py (test phase) scores against the dummy all-zero labels and writes
            # best_acc.pkl only if that "accuracy" is > 0, i.e. if some sample is predicted
            # as class 0; it always writes the identical score dict to epoch_<e>_<acc>.pkl.
            results_dir = test_dir / "work_dir" / cfg["Experiment_name"] / "eval_results"
            if not (results_dir / "best_acc.pkl").exists():
                (only,) = results_dir.glob("epoch_*.pkl")
                shutil.copy(only, results_dir / "best_acc.pkl")
                fallbacks.append(cfg["Experiment_name"])
        subprocess.run(["python", "run_fusion.py"], cwd=SLGCN, stdout=fh, stderr=subprocess.STDOUT, check=True)
    shutil.copy(Path(REPO, "Fusion", "logit_slgcn.csv"), run / f"logit_slgcnn_{split}{tag}.csv")
    # Table 2 needs each stream alone; run_fusion.py forms these fold averages internally
    # (mean of the 5 best_acc.pkl score dicts, ids sorted) but writes only the weighted sum.
    import pickle

    import numpy as np
    import pandas as pd

    for stream_tag in STREAMS:
        dicts = [pickle.load(open(test_dir / "work_dir" / f"testing_{stream_tag}_fold{k}" / "eval_results" / "best_acc.pkl", "rb"))
                 for k in range(5)]
        names = list(dicts[0])
        mean = np.mean([[np.asarray(dct[n], dtype=np.float64) for n in names] for dct in dicts], axis=0)
        df = pd.DataFrame(mean, columns=[f"logit{i}" for i in range(mean.shape[1])])
        df.insert(0, "id", [int(n.replace("SAMPLE_", "")) for n in names])
        df.sort_values("id").to_csv(run / f"logit_slgcn_{stream_tag}_{split}{tag}.csv", index=False)
    results.commit()
    rows = sum(1 for _ in open(run / f"logit_slgcnn_{split}{tag}.csv")) - 1
    summary = {"split": split, "rows": rows, "file": f"logit_slgcnn_{split}{tag}.csv", "best_acc_fallbacks": fallbacks,
               "author_weights": author_weights, "vector_streams": vec, "data_run": data_run or run_id,
               "stream_runs": sources}
    (test_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    results.commit()
    print(json.dumps(summary))
    return summary


@app.function(image=tf_image, cpu=2, memory=8192, timeout=900, volumes=read_only)
def inspect_inputs(classes: int = 6, per_class: int = 4):
    """Sanity statistics of the preprocessed inputs, loaded exactly as the authors' training
    scripts load them: micro-Doppler PNGs (grayscale, INTER_AREA 96x96, /255) and video .h5."""
    import json
    from pathlib import Path

    import cv2
    import h5py
    import numpy as np

    rows, md, vid = [], [], []
    for cls in sorted(Path(WORK, "train").iterdir(), key=lambda p: int(p.name.split("_")[0]))[:classes]:
        for s in sorted(cls.iterdir())[:per_class]:
            pngs = sorted(f for f in s.iterdir() if f.suffix == ".png")[:3]
            raw = cv2.imread(str(pngs[0]))
            ims = [cv2.resize(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE), (96, 96), interpolation=cv2.INTER_AREA) / 255.0 for f in pngs]
            h5 = next(f for f in s.iterdir() if f.suffix == ".h5")
            v = h5py.File(h5, "r")["video"][()]
            md.append(np.stack(ims)); vid.append(v)
            rows.append({"sample": f"{cls.name}/{s.name}", "png_raw_shape": list(raw.shape),
                         "md_mean": round(float(np.mean(ims)), 3), "md_std": round(float(np.std(ims)), 3),
                         "h5_shape": list(v.shape), "h5_mean": round(float(v.mean()), 3), "h5_std": round(float(v.std()), 3)})
    md, vid = np.stack(md), np.stack(vid)
    labels = np.repeat(np.arange(classes), per_class)
    def dist(x):
        flat = x.reshape(len(x), -1)
        d = np.linalg.norm(flat[:, None] - flat[None], axis=-1)
        same = d[(labels[:, None] == labels[None]) & ~np.eye(len(x), dtype=bool)].mean()
        diff = d[labels[:, None] != labels[None]].mean()
        return {"within_class_mean_l2": round(float(same), 2), "between_class_mean_l2": round(float(diff), 2)}
    out = {"samples": rows[:8], "md_distances": dist(md), "video_distances": dist(vid),
           "md_antenna_diff_mean": round(float(np.abs(md[:, 0] - md[:, 1]).mean()), 4)}
    print(json.dumps(out, indent=1))
    return out


def guard_deadline(start_utc: str, gpus: int, max_wall_s: float, max_gpu_hours: float, max_cost: float,
                   usd_per_gpu_hour: float) -> str:
    """Latest stop time that keeps a run inside all three declared ceilings, given that at most
    `gpus` GPU containers run at once (GPU-hours <= gpus x elapsed) and an all-in hourly price
    per GPU container (CPU and memory included; USD counted 1:1 as CHF)."""
    import datetime

    hours = min(max_wall_s / 3600, max_gpu_hours / gpus, max_cost / (gpus * usd_per_gpu_hour))
    start = datetime.datetime.fromisoformat(start_utc.replace("Z", "+00:00"))
    return (start + datetime.timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


# Stop-policy enforcement: runs as its own detached app beside the run it guards, so neither a
# local network drop nor the guarded app's own failure disables it. It calls the same gRPC
# methods as `modal app list` / `modal app stop` (modal 1.6.2), because the client injected
# into containers lacks the CLI's dependencies (pf-guard-001: "No module named 'rich'").
# Relies on modal 1.6.2 internals (_Client._stub, synchronize_api); recheck after upgrading modal.
@app.function(image=cpu_image, cpu=0.25, memory=512, timeout=24 * 3600, volumes={"/cache/huggingface": hf_cache})
def guard(target_app: str, deadline_utc: str, env_name: str, poll_seconds: int = 120, max_tasks: int = -1):
    """Stop Modal app `target_app` at `deadline_utc` unless it has stopped by itself; exit as
    soon as it has. The deadline bounds GPU-hours and cost only while at most the planned number
    of containers run, so with max_tasks >= 0 the guard also stops the target when it has more
    than max_tasks running tasks on two consecutive polls. Every observation is printed as JSON."""
    import asyncio
    import datetime
    import json

    from modal._utils.async_utils import synchronize_api
    from modal.client import _Client
    from modal_proto import api_pb2

    stopped = {api_pb2.APP_STATE_STOPPED: "stopped", api_pb2.APP_STATE_STOPPING: "stopping"}
    deadline = datetime.datetime.fromisoformat(deadline_utc.replace("Z", "+00:00"))

    async def main():
        client = await _Client.from_env()
        stop_requested, over = None, 0
        while True:
            now = datetime.datetime.now(datetime.timezone.utc)
            try:
                resp = await client._stub.AppList(api_pb2.AppListRequest(environment_name=env_name))
                app_state = next((a.state for a in resp.apps if a.app_id == target_app), None)
                state = "missing" if app_state is None else stopped.get(app_state, f"running ({app_state})")
                tasks = next((a.n_running_tasks for a in resp.apps if a.app_id == target_app), None)
            except Exception as e:  # keep enforcing through transient API errors
                state, tasks = f"unknown ({e!r})"[:200], None
            print(json.dumps({"at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "target": target_app, "state": state,
                              "tasks": tasks, "deadline": deadline_utc}), flush=True)
            if state == "stopped":
                return {"target": target_app, "stopped_by_guard": stop_requested is not None,
                        "stop_requested_at": stop_requested, "observed_stopped_at": now.isoformat()}
            over = over + 1 if max_tasks >= 0 and tasks is not None and tasks > max_tasks else 0
            if (now >= deadline or over >= 2) and state != "stopping":
                try:
                    await client._stub.AppStop(api_pb2.AppStopRequest(app_id=target_app,
                                                                      source=api_pb2.APP_STOP_SOURCE_CLI))
                    stop_requested = stop_requested or now.isoformat()
                    print(json.dumps({"event": "stop requested", "target": target_app, "at": now.isoformat(),
                                      "cause": "deadline" if now >= deadline else f"{tasks} tasks > {max_tasks}"}),
                          flush=True)
                except Exception as e:
                    print(json.dumps({"event": "stop failed", "error": repr(e)[:300]}), flush=True)
            wait = poll_seconds if now >= deadline else min(poll_seconds, (deadline - now).total_seconds())
            await asyncio.sleep(max(1.0, wait))

    # The container's client lives on Modal's own event loop; a fresh asyncio.run loop hangs on
    # its RPCs (pf-guard-002), so run the coroutine through Modal's synchronizer.
    result = synchronize_api(main)()
    print(json.dumps(result), flush=True)
    return result


# A remote function, not a local entrypoint: slgcn-train-full-001 ran this as a local
# entrypoint, and a local network drop cancelled all 20 remote trainings with it.
# slgcn-retrain-full-001: a fold's SystemExit, re-raised here while fetching map results, killed
# Modal's synchronizer thread ("Synchronizer thread unexpectedly died. Cause: SystemExit"); the
# other folds' inputs were cancelled and Modal reran this container, relaunching every fold. So
# slgcn_train raises RuntimeError, a failed fold is collected (return_exceptions only catches
# Exception) instead of cancelling the rest, and this container is nonpreemptible so a
# preemption cannot relaunch folds that are still training.
@app.function(image=cpu_image, cpu=0.25, memory=512, timeout=24 * 3600, nonpreemptible=True,
              volumes={"/cache/huggingface": hf_cache})
def slgcn_all(run_id: str, num_epoch: int = 250, folds: int = 5, per_gpu: int = 1, resume: bool = False,
              streams: str = "", data_run: str = "", authors_is_vector: bool = False):
    """Train every SL-GCN stream x fold (or only `streams`, e.g. "bone,jm,bm"), `per_gpu` jobs
    sharing each L4; fail if any run fails. data_run and authors_is_vector pass to slgcn_train."""
    import json

    chosen = [s for s in STREAMS if not streams or s in streams.split(",")]
    assert chosen and set(filter(None, streams.split(","))) <= set(STREAMS), streams
    jobs = [f"{stream}:{k}" for stream in chosen for k in range(folds)]
    calls = [(run_id, ",".join(jobs[i:i + per_gpu]), num_epoch, data_run, resume, authors_is_vector)
             for i in range(0, len(jobs), per_gpu)]
    assert len(calls) <= MAX_GPUS, f"{len(calls)} GPU jobs exceed the {MAX_GPUS}-GPU limit"
    results = list(slgcn_train.starmap(calls, return_exceptions=True))
    errors = {c[1]: repr(r)[:500] for c, r in zip(calls, results) if isinstance(r, BaseException)}
    summaries = [s for r in results if not isinstance(r, BaseException) for s in r]
    print(json.dumps([{k: s[k] for k in ("experiment", "exit_code", "seconds", "best_epoch", "best_heldout_acc")}
                      for s in summaries]))
    if errors:
        print(json.dumps({"failed_inputs": errors}))
    if errors or any(s["exit_code"] for s in summaries):
        raise SystemExit("an SL-GCN training run failed")


@app.function(image=torch_image, cpu=8, memory=32768, timeout=3 * 3600, volumes=read_only)
def slgcn_eval_data(run_id: str, split: str):
    """(Re)build the val or test SL-GCN inputs of an existing run from the current keypoints."""
    import shutil
    from pathlib import Path

    root = Path("/outputs/runs", run_id, "slgcn_data")
    shutil.rmtree(root / split, ignore_errors=True)
    _slgcn_dataset(root, split, Path(WORK, "pose", split))
    results.commit()
    n = sum(1 for _ in Path(WORK, "pose", split).iterdir())
    print({"split": split, "keypoint_files": n})
    return n


# Table/row -> (source file pattern in the fused run, column rule). {split} is val or test.
SYSTEMS = {
    "t1-radar": ("tf/{split}/results/logit_md_{split}.xlsx", "argmax"),
    "t1-video": ("tf/{split}/results/logit_video_{split}.xlsx", "argmax"),
    "t2-joint": ("slgcn/logit_slgcn_joint_{split}.csv", "argmax"),
    "t2-bone": ("slgcn/logit_slgcn_bone_{split}.csv", "argmax"),
    "t2-jm": ("slgcn/logit_slgcn_jm_{split}.csv", "argmax"),
    "t2-bm": ("slgcn/logit_slgcn_bm_{split}.csv", "argmax"),
    "t2-ensemble": ("slgcn/logit_slgcnn_{split}.csv", "argmax"),
    "t3-radar-video": ("tf/{split}/results/predictions/fusion_preds_{split}.csv", "pred"),
    "t3-radar-skeletal": ("fusion/fusion_md_slgcnn_{split}.csv", "pred"),
    "t3-skeletal-video": ("fusion/fusion_video_slgcnn_{split}.csv", "pred"),
    "t3-all": ("fusion/fusion_slgcnn_md_video_{split}.csv", "pred"),
}


@app.function(image=tf_image, cpu=4, memory=16384, timeout=3600, volumes=read_only)
def fuse(out_id: str, slgcn_run: str, tf_val: str, tf_test: str, require_full: bool = True):
    """Install the per-modality logits where the authors' fusion scripts expect them, run
    fusion_all.py and Multimodal_fusion.py unchanged, check they agree, then write one
    Kaggle submission per table row: id,pred over val + test ids (11,970 rows when full)."""
    import hashlib
    import json
    import shutil
    import subprocess
    from pathlib import Path

    import numpy as np
    import pandas as pd

    runs = Path("/outputs/runs")
    out = runs / out_id
    out.mkdir(parents=True, exist_ok=False)
    mslr = Path(REPO, "MSLR_radar_video")
    (mslr / "results").mkdir(exist_ok=True)
    Path(REPO, "Fusion").mkdir(exist_ok=True)
    for split, tf_run in (("val", tf_val), ("test", tf_test)):
        for kind in ("md", "video"):
            shutil.copy(runs / tf_run / "results" / f"logit_{kind}_{split}.xlsx", mslr / "results")
        shutil.copy(runs / slgcn_run / f"logit_slgcnn_{split}.csv", Path(REPO, "Fusion"))
        shutil.copytree(runs / tf_run, out / "tf" / split)
    shutil.copytree(runs / slgcn_run, out / "slgcn", ignore=shutil.ignore_patterns(
        "slgcn_data", "save_models", "work_dir", "test_*", "configs", "logs"))
    with (out / "fusion.log").open("w") as fh:
        subprocess.run(["python", "MSLR_radar_video/fusion_with_slgcnn/fusion_all.py"], cwd=REPO, stdout=fh,
                       stderr=subprocess.STDOUT, check=True)
        subprocess.run(["python", "Multimodal_fusion.py"], cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, check=True)
    shutil.copytree(mslr / "fusion_with_slgcnn" / "predictions", out / "fusion")
    shutil.copytree(Path(REPO, "Final_prediction"), out / "final_prediction")
    for split in ("val", "test"):  # the two scripts implement the same 3-modality rule
        a = pd.read_csv(out / "fusion" / f"fusion_slgcnn_md_video_{split}.csv").sort_values("id")
        b = pd.read_csv(out / "final_prediction" / f"Final_LIS_prediction_{split}.csv").sort_values("id")
        assert (a["id"].values == b["id"].values).all() and (a["pred"].values == b["pred"].values).all(), split

    expected = {s: [int(x.split("_")[1]) for x in Path(WORK, "ids", f"{s}.txt").read_text().split()]
                for s in ("val", "test")}
    manifest = {}
    (out / "submissions").mkdir()
    for system, (pattern, rule) in SYSTEMS.items():
        parts = []
        for split in ("val", "test"):
            f = out / pattern.format(split=split)
            df = pd.read_excel(f) if f.suffix == ".xlsx" else pd.read_csv(f)
            pred = df.iloc[:, 1:].to_numpy().argmax(axis=1) if rule == "argmax" else df["pred"].to_numpy()
            parts.append(pd.DataFrame({"id": df["id"].astype(int), "pred": pred.astype(int)}))
        sub = pd.concat(parts).sort_values("id").reset_index(drop=True)
        assert sub["id"].is_unique and sub["pred"].between(0, 125).all(), system
        if require_full:
            assert sorted(sub["id"]) == sorted(expected["val"] + expected["test"]), f"{system}: id set"
        path = out / "submissions" / f"{system}.csv"
        sub.to_csv(path, index=False)
        manifest[system] = {"file": f"submissions/{system}.csv", "rows": len(sub),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (out / "submissions" / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    results.commit()
    print(json.dumps(manifest, indent=1))
    return manifest


@app.local_entrypoint()
def slgcn_bench(run_id: str, jobs: str, cpu: float, memory: int, num_epoch: int = 3, data_run: str = "bench-slgcn-001"):
    """Throughput benchmark of slgcn_train with given CPU/memory requests on full-size fold data."""
    import json

    out = slgcn_train.with_options(cpu=cpu, memory=memory).remote(run_id, jobs, num_epoch, data_run)
    print(json.dumps(out))


POSE_SHARDS = {"train": 32, "val": 12, "test": 20}


@app.local_entrypoint()
def pose_all(procs: int = 4):
    """HRNet keypoints for every video, sharded; demo.py skips existing outputs, so this resumes."""
    import json

    calls = [(split, k, n, 0, procs) for split, n in POSE_SHARDS.items() for k in range(n)]
    summaries = list(pose.starmap(calls))
    print(json.dumps({"shards": len(summaries), "failed": [s["name"] for s in summaries if s["exit_code"]],
                      "new_outputs": sum(s["new_outputs"] for s in summaries),
                      "total_outputs": sum(s["total_outputs"] for s in summaries),
                      "max_shard_seconds": max(s["seconds"] for s in summaries)}))
    if any(s["exit_code"] for s in summaries):
        raise SystemExit("a pose shard failed")


@app.function(image=tf_image, cpu=2, memory=8192, timeout=1800, volumes=read_only)
def compare_logits(ours: str, theirs: str):
    """Align two logit tables (csv/xlsx, first column id) by id; report argmax agreement and
    numeric differences. Evidence only (no labels)."""
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    def load(path):
        f = Path(path)
        df = pd.read_excel(f) if f.suffix == ".xlsx" else pd.read_csv(f)
        return df.sort_values("id").reset_index(drop=True)

    a, b = load(ours), load(theirs)
    common = np.intersect1d(a["id"], b["id"])
    a, b = a[a["id"].isin(common)], b[b["id"].isin(common)]
    za, zb = a.iloc[:, 1:].to_numpy(np.float64), b.iloc[:, 1:].to_numpy(np.float64)
    out = {"ours": ours, "theirs": theirs, "rows_ours": int(len(load(ours))), "rows_theirs": int(len(load(theirs))),
           "common_ids": int(len(common)), "argmax_agreement": float((za.argmax(1) == zb.argmax(1)).mean()),
           "argmax_disagreements": int((za.argmax(1) != zb.argmax(1)).sum()),
           "max_abs_diff": float(np.abs(za - zb).max()), "mean_abs_diff": float(np.abs(za - zb).mean()),
           "pearson": float(np.corrcoef(za.ravel(), zb.ravel())[0, 1])}
    print(json.dumps(out))
    return out
