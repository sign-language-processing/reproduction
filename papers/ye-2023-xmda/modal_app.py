"""Run the pinned SignXmDA recipe (Ye et al. 2023) on the shared Modal data."""

import datetime as dt
import hashlib
import json
import os
import subprocess
import tarfile
import urllib.request
from pathlib import Path

import modal


PAPER_DIR = Path(__file__).resolve().parent
SLTUNET_REVISION = "b1d1d0e8b3b7275e10cd89229f2632b556df5de9"
RELEASE_URL = "https://data.statmt.org/bzhang/iclr2023_sltunet/{name}.tar.gz"
RELEASE_SHA256 = {
    "phoenix": "b0e708b7abe5689475905ad11ac578abb02eb0f9bf00b207bbd6d4342afe5152",
    "csldaily2": "55e3da80a0e754e8b8797c11f58f83da5779b276703979c29eb71530526f7c1a",
}
RELEASE_DIR = Path("/outputs/release")

SMKD_IMAGE = (
    modal.Image.from_registry("ghcr.io/sign-language-processing/reproduction:latest")
    .pip_install(
        "opencv-python-headless==4.10.0.84",
        "h5py==3.12.1",
        "pyarrow==18.1.0",
        "six==1.17.0",
        "pandas==2.2.3",
        "tqdm==4.67.1",
        # Same pin as the repository-root Dockerfile; the ghcr image lacks it.
        # It passes av.open(metadata_errors=...), which the image's av 19.0.2
        # rejects (TypeError); av 18.0.0 is its lowest allowed version.
        "simple-video-utils==0.7.4",
        "av==18.0.0",
    )
    .run_commands(
        "git clone https://github.com/bzhangGo/sltunet /sltunet",
        f"git -C /sltunet checkout {SLTUNET_REVISION}",
    )
    .env({"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub"})
    .add_local_file(PAPER_DIR / "extract_smkd_features.py", "/repro/extract_smkd_features.py")
)
PHOENIX = Path("/datasets/rwth-phoenix-2014-t")
FRAMES = PHOENIX / "raw/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px"
ANNOTATIONS = PHOENIX / "features/annotations/phoenix14t.pami0.{split}.annotations_only.gzip"
DATASET_ROOTS = {"phoenix14t": PHOENIX, "csl-daily": Path("/datasets/csl-daily")}
SMKD_RELEASES = {"phoenix14t": "phoenix", "csl-daily": "csldaily2"}
FEATURES_DIR = Path("/outputs/features")
DATASETS = modal.Volume.from_name("datasets", create_if_missing=False)
HF_CACHE = modal.Volume.from_name("huggingface-cache", create_if_missing=False)
RESULTS = modal.Volume.from_name("ye-2023-xmda-results", create_if_missing=True, version=2)
VOLUMES = {
    "/datasets": DATASETS.read_only(),
    "/cache/huggingface": HF_CACHE,
    "/outputs": RESULTS,
}
APP = modal.App("repro-ye-2023-xmda")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@APP.function(image=SMKD_IMAGE, volumes=VOLUMES, timeout=3600)
def fetch_release(name: str) -> dict:
    """Download, verify and unpack one SLTUNet release (contains an SMKD checkpoint)."""
    archive = RELEASE_DIR / f"{name}.tar.gz"
    url = RELEASE_URL.format(name=name)
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    if not archive.is_file() or _sha256(archive) != RELEASE_SHA256[name]:
        urllib.request.urlretrieve(url, archive)
    if _sha256(archive) != RELEASE_SHA256[name]:
        raise RuntimeError(f"checksum mismatch for {url}")
    target = RELEASE_DIR / name
    if not target.is_dir():
        with tarfile.open(archive) as handle:
            handle.extractall(target, filter="data")
    RESULTS.commit()
    return {
        "archive_sha256": RELEASE_SHA256[name],
        "files": sorted(
            f"{p.relative_to(target)} {p.stat().st_size}"
            for p in target.rglob("*")
            if p.is_file()
        ),
    }


CSL_DAILY = Path("/datasets/csl-daily")
MANIFEST_DIR = Path("/outputs/manifests")


def _hash_files(paths: list) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _phoenix_manifest() -> dict:
    import csv
    import gzip
    import pickle
    from concurrent.futures import ThreadPoolExecutor

    manual = PHOENIX / "raw/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual"
    report = {"splits": {}, "files": {}}
    videos = []
    for split in ("train", "dev", "test"):
        csv_path = manual / f"PHOENIX-2014-T.{split}.corpus.csv"
        with csv_path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle, delimiter="|"))
        annotation_path = Path(str(ANNOTATIONS).format(split=split))
        with gzip.open(annotation_path, "rb") as handle:
            annotations = pickle.load(handle)
        by_name = {f"{split}/{row['name']}": row for row in rows}
        mismatched = [
            a["name"] for a in annotations
            if a["name"] not in by_name
            or a["gloss"].split() != by_name[a["name"]]["orth"].split()
        ]
        folders = sorted(p for p in (FRAMES / split).iterdir() if p.is_dir())
        videos += folders
        report["files"][str(csv_path)] = _sha256(csv_path)
        report["files"][str(annotation_path)] = _sha256(annotation_path)
        report["splits"][split] = {
            "csv_rows": len(rows),
            "annotation_entries": len(annotations),
            "annotation_names_or_glosses_not_matching_csv": mismatched,
            "frame_folders": len(folders),
            "frame_folder_names_match_csv": sorted(p.name for p in folders) == sorted(r["name"] for r in rows),
        }

    def describe(folder: Path) -> tuple:
        frames = sorted(folder.glob("*.png"))
        return f"{folder.parent.name}/{folder.name}", len(frames), _hash_files(frames)

    with ThreadPoolExecutor(64) as pool:
        per_video = list(pool.map(describe, videos))
    for split in report["splits"]:
        counts = [n for name, n, _ in per_video if name.startswith(split + "/")]
        report["splits"][split].update(frames=sum(counts), empty_folders=counts.count(0))
    return report, per_video


def _csl_daily_manifest() -> dict:
    from concurrent.futures import ThreadPoolExecutor

    labels = CSL_DAILY / "sentence_label"
    rows = [line.split("|") for line in (labels / "split_1.txt").read_text().split()[1:]]
    names = {name for name, _ in rows}
    videos = sorted((CSL_DAILY / "videos").glob("*.mp4"))
    stems = {p.stem for p in videos}
    report = {
        "files": {str(p): _sha256(p) for p in sorted(labels.iterdir()) if p.is_file()},
        "split_counts": {s: sum(1 for _, x in rows if x == s) for s in ("train", "dev", "test")},
        "videos": len(videos),
        "video_bytes": sum(p.stat().st_size for p in videos),
        "split_names_without_video": sorted(names - stems),
        "videos_not_in_split": sorted(stems - names),
    }
    with ThreadPoolExecutor(64) as pool:
        per_video = list(pool.map(lambda p: (p.name, p.stat().st_size, _sha256(p)), videos))
    return report, per_video


@APP.function(image=SMKD_IMAGE, volumes=VOLUMES, timeout=6 * 3600, cpu=16, memory=16384)
def data_manifest(dataset: str) -> dict:
    """Count and content-hash a dataset tree; write the per-video manifest to /outputs."""
    report, per_video = {"phoenix14t": _phoenix_manifest, "csl-daily": _csl_daily_manifest}[dataset]()
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    path = MANIFEST_DIR / f"{dataset}.tsv"
    path.write_text("".join("\t".join(map(str, row)) + "\n" for row in per_video), encoding="utf-8")
    RESULTS.commit()
    return {**report, "manifest": str(path), "manifest_sha256": _sha256(path),
            "function_call_id": modal.current_function_call_id()}


XMDA_IMAGE = (
    modal.Image.from_dockerfile(PAPER_DIR / "Dockerfile", context_dir=PAPER_DIR, add_python="3.11")
    # For _config() in the Modal wrapper interpreter; SignXmDA runs on conda py3.7.
    .pip_install("pyyaml==6.0.2")
    .add_local_file(PAPER_DIR / "build_pickles.py", "/repro/build_pickles.py")
    .add_local_file(PAPER_DIR / "dump_gloss_embedding.py", "/repro/dump_gloss_embedding.py")
)
CONDA_PYTHON = "/root/miniconda3/bin/python"
RUNS_DIR = Path("/outputs/runs")
PGEN_PAIRS = {
    "phoenix14t": "/PGen/data/DA_paralle_sample/De/GPT.de.T5.gloss:/PGen/data/DA_paralle_sample/De/GPT.de",
    "csl-daily": "/PGen/data/DA_paralle_sample/Zh/GPT.zh.T5.gloss:/PGen/data/DA_paralle_sample/Zh/GPT.zh",
}
# Paper Table 10 per dataset; ff_size 2048 is a guess from the authors' fork
# Atrewin/slt configs/jizhi/default.yaml (Table 10 does not state it).
# CSL-Daily text is space-separated characters with level "word", as in the
# upstream Sign_XmDA.yaml (cslt.word.*.pickle.char); level "char" would join
# hypotheses without spaces and sacrebleu 13a would score whole sentences.
TABLE10 = {
    "phoenix14t": {"layers": 3, "max_output_length": 30},
    "csl-daily": {"layers": 1, "max_output_length": 50},
}
TEACHER_SEEDS = (56, 57, 58, 59)


def _environment() -> dict:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update(PYTHONNOUSERSITE="1", HF_HOME="/cache/huggingface",
                       HF_HUB_CACHE="/cache/huggingface/hub")
    return environment


def _set(config: dict, dotted: str, value) -> None:
    *parents, key = dotted.split(".")
    for parent in parents:
        config = config[parent]
    if key not in config and dotted not in OPTIONAL_KEYS:
        raise KeyError(f"upstream config has no {dotted}")
    config[key] = value


# Keys the upstream code reads with .get() but configs/Sign_XmDA.yaml omits.
OPTIONAL_KEYS = {"data.gls_vocab", "training.G2T_pretraining"}


def _config(dataset: str, kind: str, run_id: str, seed: int, teachers: str, preflight: bool,
            hparams: str = "table10") -> Path:
    """Upstream configs/Sign_XmDA.yaml with run paths and, unless hparams is "released", Table 10 values."""
    import yaml

    config = yaml.safe_load(Path("/SignXmDA/configs/Sign_XmDA.yaml").read_text())
    table = TABLE10[dataset]
    features = f"{dataset}.smkd"
    overrides = {
        "data.data_path": str(FEATURES_DIR) + "/",
        "data.version": "phoenix_2014_trans",
        "data.train": f"{features}.train.pickle",
        "data.dev": f"{features}.dev.pickle",
        "data.test": f"{features}.test.pickle",
        "training.random_seed": seed,
        "training.model_dir": str(RUNS_DIR / run_id),
        "training.forward_type": {"teacher": "gloss", "mixup": "mixup", "xmda": "mixup"}.get(kind, "sign"),
        "testing.forward_type": "gloss" if kind == "teacher" else "sign",
    }
    if hparams == "table10":
        overrides.update({
            "training.learning_rate": 0.001,
            "training.betas": [0.9, 0.98],
            "training.label_smoothing": 0.1,
            "training.translation_max_output_length": table["max_output_length"],
            "training.batch_size": 128,
        })
        for side, activation in (("encoder", "gelu"), ("decoder", "gelu")):
            overrides.update({
                f"model.{side}.num_layers": table["layers"],
                f"model.{side}.num_heads": 8,
                f"model.{side}.hidden_size": 512,
                f"model.{side}.ff_size": 2048,
                f"model.{side}.dropout": 0.3,
                f"model.{side}.embeddings.embedding_dim": 512,
                f"model.{side}.embeddings.dropout": 0.3,
                f"model.{side}.embeddings.activation_type": activation,
            })
    elif hparams != "released":
        raise ValueError(hparams)
    if kind == "teacher":
        overrides.update({
            "data.gloss2text_data": PGEN_PAIRS[dataset],
            "training.G2T_pretraining": True,
            # The upstream config's test split is the train pickle: test() then
            # writes the teacher's translations of the training set.
            "data.test": f"{features}.train.pickle",
        })
    if kind in ("kd", "xmda"):
        overrides["data.train"] = f"{features}.train.teachers.pickle"
    if kind in ("mixup", "xmda"):
        overrides["model.gloss_pre_embedding"] = str(RUNS_DIR / teachers / "gloss_embedding.pt")
        overrides["data.gls_vocab"] = str(RUNS_DIR / teachers / "gls.vocab")
    if preflight:
        # Full-size batches for a throughput measurement; tiny dev and decode grid.
        overrides.update({
            "data.random_dev_subset": 32,
            "training.epochs": 2,
            "training.validation_freq": 50,
            "training.logging_freq": 10,
            "testing.translation_beam_sizes": [1, 2],
            "testing.translation_beam_alphas": [-1],
            "testing.recognition_beam_sizes": [1],
        })
    for key, value in overrides.items():
        _set(config, key, value)
    path = Path(f"/tmp/{run_id}.yaml")
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def _results(log: str) -> dict:
    import re

    results = {}
    for split in ("DEV", "TEST"):
        block = log[log.rindex(f"[{split}] partition [Recognition & Translation] results"):]
        number = r"(-?[0-9.]+)"
        beam, alpha = re.search(rf"Best Translation Beam Size: {number} and Alpha: {number}", block).groups()
        results[split.lower()] = {
            "translation_beam_size": int(beam),
            "translation_beam_alpha": int(alpha),
            "wer": float(re.search(rf"WER {number}", block).group(1)),
            "bleu4": float(re.search(rf"BLEU-4 {number}", block).group(1)),
            "chrf": float(re.search(rf"CHRF {number}", block).group(1)),
            "rouge_l": float(re.search(rf"ROUGE {number}", block).group(1)),
        }
    return results


def _gpu() -> str:
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()


@APP.function(image=SMKD_IMAGE, gpu="L4", volumes=VOLUMES, timeout=24 * 3600, cpu=8, memory=32768)
def smkd_features(dataset: str, split: str, limit: int = 0) -> dict:
    """Write SMKD features for one split to /outputs/features/<dataset>.smkd.<split>."""
    started = dt.datetime.now(dt.timezone.utc)
    name = f"{dataset}.smkd.{split}" + (f".limit{limit}" if limit else "")
    out = FEATURES_DIR / name
    command = [
        "python", "/repro/extract_smkd_features.py",
        "--dataset", dataset,
        "--root", str(DATASET_ROOTS[dataset]),
        "--split", split,
        "--release", str(RELEASE_DIR / SMKD_RELEASES[dataset]),
        "--out", str(out),
        "--limit", str(limit),
    ]
    # The hash-verified 2022 checkpoint pickles numpy objects; torch>=2.6 refuses
    # that by default (weights_only=True).
    environment = {**os.environ, "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"}
    subprocess.run(command, cwd="/sltunet/smkd", env=environment, check=True)
    RESULTS.commit()
    summary = json.loads(Path(f"{out}.json").read_text())
    summary.pop("index")
    return {
        **summary,
        "function_call_id": modal.current_function_call_id(),
        "started_at": started.isoformat(),
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gpu": _gpu(),
        "command": " ".join(command),
        "npy_sha256": _sha256(Path(f"{out}.npy")),
        "json_sha256": _sha256(Path(f"{out}.json")),
    }


@APP.function(image=XMDA_IMAGE, volumes=VOLUMES, timeout=4 * 3600, cpu=4, memory=65536)
def pickles(dataset: str, teacher_runs: list = ()) -> list:
    """Write <dataset>.smkd.<split>.pickle, or the multi-reference train pickle."""
    import glob

    results = []
    splits = ["train"] if teacher_runs else ["train", "dev", "test"]
    for split in splits:
        name = f"{dataset}.smkd.{split}" + (".teachers" if teacher_runs else "")
        outputs = [sorted(glob.glob(str(RUNS_DIR / run / "best.IT_*.test.txt")))[-1] for run in teacher_runs]
        command = [
            CONDA_PYTHON, "/repro/build_pickles.py",
            "--features", str(FEATURES_DIR / f"{dataset}.smkd.{split}"),
            "--out", str(FEATURES_DIR / f"{name}.pickle"),
        ] + (["--teacher-outputs", *outputs] if outputs else [])
        subprocess.run(command, env=_environment(), check=True)
        results.append({"command": " ".join(command), "sha256": _sha256(FEATURES_DIR / f"{name}.pickle")})
    RESULTS.commit()
    return results


@APP.function(image=XMDA_IMAGE, volumes=VOLUMES, timeout=1800)
def gloss_embedding(teacher_run: str) -> dict:
    """Dump the best teacher checkpoint's GlossEmbeddings for gloss_pre_embedding."""
    import glob

    run = RUNS_DIR / teacher_run
    checkpoint = sorted(glob.glob(str(run / "best.IT_*.test.txt")))[-1].split("best.IT_")[1].split(".")[0]
    checkpoint = run / f"{int(checkpoint)}.ckpt"
    command = [CONDA_PYTHON, "/repro/dump_gloss_embedding.py", str(checkpoint), str(run / "gloss_embedding.pt")]
    subprocess.run(command, env=_environment(), check=True)
    RESULTS.commit()
    return {"command": " ".join(command), "checkpoint": str(checkpoint),
            "sha256": _sha256(run / "gloss_embedding.pt"), "gls_vocab_sha256": _sha256(run / "gls.vocab")}


def _watch_gpu(stop, peak: list) -> None:
    while not stop.wait(15):
        used = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
        peak[0] = max(peak[0], int(used.strip()))


def _commit_periodically(stop) -> None:
    while not stop.wait(600):
        RESULTS.commit()


@APP.function(image=XMDA_IMAGE, gpu="T4", volumes=VOLUMES, timeout=24 * 3600, cpu=4, memory=65536)
def train(dataset: str, kind: str, run_id: str, seed: int = 56, teachers: str = "",
          preflight: bool = False, hparams: str = "table10") -> dict:
    """Train with signjoey.training_XmDA (which ends with prediction_XmDA.test)."""
    import threading
    import time

    config = _config(dataset, kind, run_id, seed, teachers, preflight, hparams)
    command = [CONDA_PYTHON, "-m", "signjoey.training_XmDA", "--config", str(config)]
    started = dt.datetime.now(dt.timezone.utc)
    clock = time.monotonic()
    stop, peak = threading.Event(), [0]
    threads = [threading.Thread(target=_watch_gpu, args=(stop, peak), daemon=True),
               threading.Thread(target=_commit_periodically, args=(stop,), daemon=True)]
    for thread in threads:
        thread.start()
    try:
        completed = subprocess.run(command, cwd="/SignXmDA", env=_environment())
    finally:
        stop.set()
        RESULTS.commit()
    run = RUNS_DIR / run_id
    log = (run / "train.log").read_text(encoding="utf-8") if (run / "train.log").exists() else ""
    summary = {
        "run_id": run_id,
        "function_call_id": modal.current_function_call_id(),
        "command": " ".join(command),
        "config": (run / "config.yaml").read_text() if (run / "config.yaml").exists() else config.read_text(),
        "exit_code": completed.returncode,
        "started_at": started.isoformat(),
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "duration_seconds": time.monotonic() - clock,
        "peak_gpu_memory_mib": peak[0],
        "gpu": _gpu(),
        "image_id": os.environ.get("MODAL_IMAGE_ID"),
        "last_logged_step": _last_step(log),
        "artifacts": {p.name: _sha256(p) for p in sorted(run.glob("best.IT_*")) if p.is_file()},
    }
    if completed.returncode == 0:
        summary["results"] = _results(log)
        summary["artifacts"]["train.log"] = _sha256(run / "train.log")
    (run / "summary.json").write_text(json.dumps(summary, indent=2))
    RESULTS.commit()
    return summary


@APP.function(image=XMDA_IMAGE, gpu="T4", volumes=VOLUMES, timeout=6 * 3600, cpu=4, memory=65536)
def decode_grid(run_id: str) -> dict:
    """Decode a finished run's best checkpoint once per beam/alpha with prediction_XmDA.test."""
    import time

    import yaml

    run = RUNS_DIR / run_id
    config = yaml.safe_load((run / "config.yaml").read_text())
    step = sorted(run.glob("best.IT_*.dev_results.pkl"))[-1].name.split("best.IT_")[1].split(".")[0]
    checkpoint = run / f"{int(step)}.ckpt"
    out = run / "decode_grid"
    out.mkdir(exist_ok=True)
    started, clock, grid = dt.datetime.now(dt.timezone.utc), time.monotonic(), {}
    beams, alphas = config["testing"]["translation_beam_sizes"], config["testing"]["translation_beam_alphas"]
    for beam in beams:
        for alpha in alphas:
            # One setting per call so every setting's dev and test scores are logged.
            config["testing"].update(translation_beam_sizes=[beam], translation_beam_alphas=[alpha],
                                     recognition_beam_sizes=[1])
            name = out / f"b{beam}_a{alpha}"
            Path(f"{name}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
            command = [CONDA_PYTHON, "-c",
                       "import sys; from signjoey.prediction_XmDA import test; "
                       "test(sys.argv[1], ckpt=sys.argv[2], output_path=sys.argv[3])",
                       f"{name}.yaml", str(checkpoint), str(name)]
            completed = subprocess.run(command, cwd="/SignXmDA", env=_environment(),
                                       capture_output=True, text=True)
            Path(f"{name}.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(f"{name}: exit {completed.returncode}")
            grid[f"beam {beam}, alpha {alpha}"] = _results(completed.stdout + completed.stderr)
    RESULTS.commit()
    return {
        "run_id": run_id,
        "checkpoint": str(checkpoint),
        "function_call_id": modal.current_function_call_id(),
        "started_at": started.isoformat(),
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "duration_seconds": time.monotonic() - clock,
        "gpu": _gpu(),
        "image_id": os.environ.get("MODAL_IMAGE_ID"),
        "grid": grid,
        "artifacts": {p.name: _sha256(p) for p in sorted(out.iterdir()) if p.is_file()},
    }


def _last_step(log: str) -> int:
    import re

    steps = re.findall(r"Step: (\d+)", log)
    return int(steps[-1]) if steps else 0


@APP.local_entrypoint()
def main(step: str, split: str = "dev", limit: int = 0, dataset: str = "phoenix14t",
         kind: str = "baseline", run_id: str = "", seed: int = 56, teachers: str = "",
         preflight: bool = False, hparams: str = "table10", spawn: bool = False) -> None:
    if step == "train" and spawn:
        # A client blocked in .remote() was cancelled with its call when its session token
        # expired after about 5 h (phoenix14t-xmda-seed56-attempt-1). With run --detach,
        # spawn returns at once and the run's result is runs/RUN_ID/summary.json.
        call = train.spawn(dataset, kind, run_id, seed, teachers, preflight, hparams)
        print(json.dumps({"run_id": run_id, "function_call_id": call.object_id,
                          "summary": f"/outputs/runs/{run_id}/summary.json"}, indent=2))
        return
    if step == "fetch_release":
        result = fetch_release.remote(split)
    elif step == "data_manifest":
        result = data_manifest.remote(split)
    elif step == "pickles":
        result = pickles.remote(dataset)
    elif step == "smkd_features":
        result = smkd_features.remote(dataset, split, limit)
    elif step == "train":
        result = train.remote(dataset, kind, run_id, seed, teachers, preflight, hparams)
    elif step == "decode_grid":
        result = decode_grid.remote(run_id)
    elif step == "teacher_pickle":
        result = pickles.remote(dataset, teachers.split(","))
    elif step == "gloss_embedding":
        result = gloss_embedding.remote(teachers)
    else:
        raise ValueError(step)
    print(json.dumps(result, indent=2))
