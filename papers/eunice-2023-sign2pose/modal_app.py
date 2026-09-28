"""Sign2Pose (Eunice et al., Sensors 2023) conditional reconstruction on Modal.

Extracts MediaPipe poses from the WLASL videos with poses.py (CPU, sharded), exports SPOTER CSVs, then trains
pinned SPOTER (the paper's cited basis) with its own train.py. Patches in patches/ are applied in name order;
ATTEMPT_UNPATCHED=1 builds the upstream code as published.

Usage (from the repository root, always through the workspace wrapper):
  W=.agents/skills/reproduce-paper/scripts/modal_repro_sign.sh
  $W run papers/eunice-2023-sign2pose/modal_app.py::extract_all --shards 64        # resumable
  $W run papers/eunice-2023-sign2pose/modal_app.py::export_csvs
  $W run papers/eunice-2023-sign2pose/modal_app.py::main --subset 100 --variant keyframes --epochs 300 --tag full
"""

import json
import os
import subprocess
import threading
import time
from pathlib import Path

import modal

PAPER_ID = "eunice-2023-sign2pose"
SPOTER_COMMIT = "0f909bf92690772f43f0062be41860ed85b461ad"
GPU = "L4"
HERE = Path(__file__).parent
PATCHES = [] if os.environ.get("ATTEMPT_UNPATCHED") else sorted((HERE / "patches").glob("*.patch"))

image = (
    modal.Image.from_registry("ghcr.io/sign-language-processing/reproduction:latest")
    .pip_install("pandas==2.2.3", "opencv-python-headless==4.10.0.84", "matplotlib==3.9.2", "tqdm==4.67.1")
    # The NGC base ships Hugging Face `datasets`, a regular package that shadows SPOTER's __init__-less datasets/
    # directory (ModuleNotFoundError: datasets.czech_slr_dataset). SPOTER never uses the HF package.
    .run_commands("pip uninstall -y datasets")
    .run_commands(f"git clone https://github.com/maty-bohacek/spoter.git /opt/spoter && "
                  f"git -C /opt/spoter checkout {SPOTER_COMMIT}")
)
for patch in PATCHES:
    image = image.add_local_file(str(patch), f"/opt/patches/{patch.name}", copy=True)
if PATCHES:
    image = image.run_commands("cd /opt/spoter && for p in /opt/patches/*.patch; do git apply -v $p; done")

pose_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("mediapipe==0.10.21", "simple-video-utils==0.9.1")
    .add_local_file(str(HERE / "poses.py"), "/opt/poses.py")
)

app = modal.App(PAPER_ID)
hf_cache = modal.Volume.from_name("huggingface-cache")
datasets = modal.Volume.from_name("datasets")
results = modal.Volume.from_name(f"{PAPER_ID}-results", create_if_missing=True)


def _commit_every_10_min():
    """Persist resume.pth and checkpoints so a restarted attempt can see them."""
    while True:
        time.sleep(600)
        results.commit()


# Retries restart a timed-out or crashed attempt; patches/0004 resumes it from the last committed epoch.
@app.function(image=image, gpu=GPU, timeout=24 * 60 * 60, retries=modal.Retries(max_retries=3),
              volumes={"/cache/huggingface": hf_cache, "/datasets": datasets.read_only(), "/results": results},
              env={"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub"})
def train(subset: int, variant: str, epochs: int, tag: str, seed: int = 379):
    run = f"{tag}-wlasl{subset}-{variant}-seed{seed}"
    out = Path("/results/runs") / run
    out.mkdir(parents=True, exist_ok=True)
    csv = lambda split: f"/results/csv/WLASL{subset}_{split}_{variant}.csv"
    cmd = ["python", "-m", "train", "--experiment_name", run, "--num_classes", str(subset), "--hidden_dim", "108",
           "--epochs", str(epochs), "--lr", "0.001", "--seed", str(seed),
           "--training_set_path", csv("train"), "--validation_set", "from-file",
           "--validation_set_path", csv("validation"), "--testing_set_path", csv("test")]
    started = time.time()
    threading.Thread(target=_commit_every_10_min, daemon=True).start()
    with open(out / "stdout.log", "a") as log:
        log.write(" ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=out, stdout=log, stderr=subprocess.STDOUT,
                              env={**os.environ, "PYTHONPATH": "/opt/spoter"})
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.used", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    freeze = subprocess.run(["pip", "freeze"], capture_output=True, text=True).stdout
    (out / "pip-freeze.txt").write_text(freeze)
    results.commit()
    tail = (out / "stdout.log").read_text().splitlines()[-40:]
    return {"run": run, "exit_code": proc.returncode, "wall_seconds": round(time.time() - started, 1),
            "gpu": gpu,
            "patches": sorted(os.listdir("/opt/patches")) if os.path.isdir("/opt/patches") else [], "tail": tail}


@app.function(image=pose_image, cpu=8, memory=8192, timeout=6 * 60 * 60,
              volumes={"/cache/huggingface": hf_cache, "/datasets": datasets.read_only(), "/results": results},
              env={"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub"})
def extract_shard(shard: int, shards: int, limit: int = 0):
    """Poses for the index rows whose video hashes to this shard; poses.py skips instances already written."""
    import csv, zlib
    with open("/datasets/WLASL/index.csv") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if zlib.crc32(r["file"].encode()) % shards == shard]
    rows = rows[:limit] if limit else rows
    with open("/tmp/index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=reader.fieldnames)
        w.writeheader()
        w.writerows(rows)
    folder = f"/results/preflight-limit{limit}" if limit else "/results/poses"  # preflight never mixes into poses/
    out = f"{folder}/shard-{shard:03d}-of-{shards:03d}.jsonl"
    os.makedirs(folder, exist_ok=True)
    started = time.time()
    proc = subprocess.run(["python", "/opt/poses.py", "extract", "--index", "/tmp/index.csv",
                           "--wlasl-json", "/datasets/WLASL/WLASL_v0.3.json", "--videos-root", "/datasets/WLASL",
                           "--out", out, "--workers", "8"], capture_output=True, text=True)
    results.commit()
    return {"shard": shard, "rows": len(rows), "exit_code": proc.returncode,
            "wall_seconds": round(time.time() - started, 1), "log_tail": (proc.stdout + proc.stderr)[-2000:]}


@app.local_entrypoint()
def extract_all(shards: int = 64, limit: int = 0):
    for r in extract_shard.starmap([(i, shards, limit) for i in range(shards)]):
        print({k: v for k, v in r.items() if k != "log_tail"}, flush=True)
        if r["exit_code"]:
            print(r["log_tail"], flush=True)


@app.function(image=pose_image, memory=32768, timeout=2 * 60 * 60,
              volumes={"/cache/huggingface": hf_cache, "/results": results},
              env={"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub"})
def export_csvs():
    """Concatenate pose shards and write SPOTER CSVs for every subset, with and without key frames."""
    import glob
    shards = sorted(glob.glob("/results/poses/shard-*.jsonl"))
    with open("/tmp/poses.jsonl", "w") as out:
        for s in shards:
            out.write(open(s).read())
    logs = []
    for subset in (100, 300, 1000, 2000):
        for kf in (["--keyframes"], []):
            logs.append(subprocess.run(["python", "/opt/poses.py", "csv", "--poses", "/tmp/poses.jsonl", "--subset",
                                        str(subset), *kf, "--out-dir", "/results/csv"],
                                       check=True, stdout=subprocess.PIPE, text=True).stdout)
    results.commit()
    errors = [json.loads(l) for l in open("/tmp/poses.jsonl") if '"error"' in l[:300]]
    print(f"{len(shards)} shards, {len(errors)} instances with errors: {[e['key'] + ' ' + e['error'] for e in errors]}")
    print("".join(logs))


@app.local_entrypoint()
def main(subset: int = 100, variant: str = "keyframes", epochs: int = 300, tag: str = "full", seed: int = 379):
    call = train.spawn(subset, variant, epochs, tag, seed)
    print(f"function_call_id={call.object_id}", flush=True)
    result = call.get()
    print("\n".join(result.pop("tail")))
    print(result)
