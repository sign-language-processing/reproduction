"""Modal entry points for the Alyami & Luqman (2024) CSLR comparative reproduction.

The recipe is the four published implementations (VAC_CSLR for VAC and SMKD,
Temporal-Lift-Pooling, SEN_CSLR, CorrNet), pinned in the Dockerfile and run
through their own main.py. This file populates the shared datasets Volume,
resizes frames, writes each setting's split lists in the upstream preprocess
format, runs upstream commands, and keeps their outputs on the paper's results
Volume.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

PAPER_ID = "8d2b672d06ddb561f890798d7c1e434389c34202"
PAPER_DIR = Path(__file__).resolve().parent
APP = modal.App("repro-alyami-2024-cslr")
DATASETS = modal.Volume.from_name("datasets")
HF_CACHE = modal.Volume.from_name("huggingface-cache")
RESULTS = modal.Volume.from_name("alyami-2024-cslr-results", create_if_missing=True, version=2)
HF_ENV = {"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub"}

DATA_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("p7zip-full")
    .pip_install("gdown==5.2.0", "requests==2.32.3")
    .env(HF_ENV)
)
# Writable datasets mount: only the two population functions below use it.
POPULATE_VOLUMES = {"/datasets": DATASETS, "/cache/huggingface": HF_CACHE}

# ---------------------------------------------------------------------------
# Continuous GSL (Adaloglou et al.), Zenodo record 3941811, CC BY 4.0.
GSL_ROOT = Path("/datasets/gsl-continuous")
GSL_ZENODO_RECORD = "3941811"
# The dataset's public Drive folder holds the split files the Zenodo record lacks.
GSL_DRIVE_FOLDER = "18ruYi9MULMm1KQtUgdIhN0m-XilRhncg"
GSL_SPLIT_SHA256 = {  # hashed on 2026-10-26 (reproduction.json datasets[grsl])
    "GSL SI/gsl_split_SI_train.csv": "41e2e6d27df79a3cec2a6d14a73960442db5eaca58c7f144de404fc4f72cbfc8",
    "GSL SI/gsl_split_SI_dev.csv": "11d5f9534085904d9c68ac5cd56ff8c11d8602a6780dd0d70cd2baabcb85ad11",
    "GSL SI/gsl_split_SI_test.csv": "eee97832df609932d3568601d3cdb472411270bbf10f05b7286b7f43ee69b534",
    "GSL SD/gsl_train_unseen.txt": "f63e85fc5a950994bfd486d99e73661b6a8a6a78a5063cb78da3ade6cbae8a91",
    "GSL SD/gsl_dev_unseen.txt": "eace26ce6cafb95a8b8dfc707ff2dbbe0404b69262e85e107c1961cd89fd6033",
    "GSL SD/gsl_test_unseen.txt": "17f551c749843580cd79bb13fbe73d9b00618dff38d79a9e9f36ca67d8c3fff8",
}

# ---------------------------------------------------------------------------
# ArabSign RGB, already on the Volume as six per-signer 7z archives.
ARABSIGN_RGB = Path("/datasets/arab-sign/RGB")
ARABSIGN_IMPORT_MANIFEST = Path("/datasets/arab-sign/_drive_imports/72178996f6c31e6c/manifest.json")
ARABSIGN_VIDEOS = ARABSIGN_RGB / "videos"
ARABSIGN_SIGNERS = ("01", "02", "03", "04", "05", "06")
ARABSIGN_ENTRY = re.compile(r"^(?P<signer>\d{2})/(?P<split>train|test)/(?P<sentence>\d{4})/(?P<stem>[^/]+)\.mp4$")
# The dataset author's per-signer split lists (github.com/Hamzah-Luqman/ArabSign).
ARABSIGN_SPLITS_URL = ("https://github.com/Hamzah-Luqman/ArabSign/raw/"
                       "8f6e127a054052228027d6da98005f991d300a7b/ArabSign-EncoderDec.zip")
ARABSIGN_SPLITS_SHA256 = "6980aec2beea93fb2299198f7031991f684bb064c8093343128eb13789318eb0"


def _sha256(path, chunk=1 << 24):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _download(url, target, expected_md5=None, expected_size=None):
    """Streams url to target via a .partial file; returns its sha256 and md5."""
    import requests

    partial = Path(f"{target}.partial")
    sha, md5 = hashlib.sha256(), hashlib.md5()
    for attempt in range(4):
        try:
            with requests.get(url, stream=True, timeout=120) as response:
                response.raise_for_status()
                with open(partial, "wb") as f:
                    for block in response.iter_content(1 << 24):
                        f.write(block)
                        sha.update(block)
                        md5.update(block)
            break
        except requests.RequestException:
            if attempt == 3:
                raise
            sha, md5 = hashlib.sha256(), hashlib.md5()
            time.sleep(30 * (attempt + 1))
    if expected_size is not None and partial.stat().st_size != expected_size:
        raise RuntimeError(f"{url}: {partial.stat().st_size} bytes, expected {expected_size}")
    if expected_md5 is not None and md5.hexdigest() != expected_md5:
        raise RuntimeError(f"{url}: md5 {md5.hexdigest()}, expected {expected_md5}")
    partial.rename(target)
    return {"sha256": sha.hexdigest(), "md5": md5.hexdigest(), "bytes": target.stat().st_size}


@APP.function(image=DATA_IMAGE, cpu=8, memory=16384, volumes=POPULATE_VOLUMES, timeout=12 * 3600)
def populate_gsl():
    """Populates /datasets/gsl-continuous: Zenodo RGB zips, Drive split files, manifest.

    Idempotent: files already recorded in the manifest with the right size are
    kept; everything is verified against Zenodo's md5 or the recorded sha256.
    """
    import gdown
    import requests

    zenodo_dir, split_dir = GSL_ROOT / "zenodo-3941811", GSL_ROOT / "splits"
    zenodo_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = GSL_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"zenodo_files": {}}

    record = requests.get(f"https://zenodo.org/api/records/{GSL_ZENODO_RECORD}", timeout=60).json()
    wanted = [f for f in record["files"] if not f["key"].endswith("_Depth.zip")]  # RGB + supplementary

    def fetch(entry):
        key, target = entry["key"], zenodo_dir / entry["key"]
        known = manifest["zenodo_files"].get(key)
        if known and target.exists() and target.stat().st_size == entry["size"]:
            return key, known
        info = _download(entry["links"]["self"], target, entry["checksum"].removeprefix("md5:"), entry["size"])
        DATASETS.commit()
        return key, info

    with ThreadPoolExecutor(4) as pool:
        for key, info in pool.map(fetch, wanted):
            manifest["zenodo_files"][key] = info
            manifest_path.write_text(json.dumps(manifest, indent=2))

    listing = gdown.download_folder(id=GSL_DRIVE_FOLDER, skip_download=True, quiet=True, remaining_ok=True)
    splits = {}
    for relative, expected in GSL_SPLIT_SHA256.items():
        matches = [f for f in listing if f.path.replace("\\", "/").endswith(relative)]
        if len(matches) != 1:
            raise RuntimeError(f"{relative}: {len(matches)} matches in the Drive folder listing")
        target = split_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or _sha256(target) != expected:
            gdown.download(id=matches[0].id, output=str(target), quiet=True)
        digest = _sha256(target)
        if digest != expected:
            raise RuntimeError(f"{relative}: sha256 {digest}, recorded {expected}")
        splits[relative] = {"drive_id": matches[0].id, "drive_path": matches[0].path, "sha256": digest,
                            "lines": len(target.read_bytes().splitlines())}

    manifest.update({
        "dataset": "Continuous Greek Sign Language (GSL), Adaloglou et al.",
        "license": record["metadata"].get("license"),
        "zenodo_record": GSL_ZENODO_RECORD,
        "zenodo_doi": record.get("doi"),
        "drive_folder": GSL_DRIVE_FOLDER,
        "splits": splits,
        "populated_by": f"papers/alyami-2024-cslr-comparative/modal_app.py::populate_gsl (paper {PAPER_ID})",
        "updated_at_utc": _now(),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2))
    DATASETS.commit()
    return manifest


@APP.function(image=DATA_IMAGE, cpu=8, memory=16384, volumes=POPULATE_VOLUMES, timeout=4 * 3600)
def extract_arabsign_signer(signer: str):
    """Extracts one RGB archive to videos/<signer>/<split>/<sentence>/<stem>.mp4 on the Volume."""
    done = ARABSIGN_VIDEOS / f"{signer}.manifest.json"
    if done.exists():
        return json.loads(done.read_text())
    archive = ARABSIGN_RGB / f"{signer}.7z"
    expected = json.loads(ARABSIGN_IMPORT_MANIFEST.read_text())["files"][f"{signer}.7z"]
    archive_sha = _sha256(archive)
    if archive_sha != expected["sha256"]:
        raise RuntimeError(f"{archive}: sha256 {archive_sha}, import manifest says {expected['sha256']}")
    workspace = Path("/tmp/arabsign")
    subprocess.run(["7z", "x", "-bd", "-y", f"-o{workspace}", str(archive)], check=True, stdout=subprocess.DEVNULL)
    files, unparsed = {}, []
    for path in sorted(workspace.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(workspace).as_posix()
        if not ARABSIGN_ENTRY.match(relative) or not relative.startswith(f"{signer}/"):
            unparsed.append(relative)
            continue
        target = ARABSIGN_VIDEOS / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        files[relative] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    record = {"archive": archive.name, "archive_sha256": archive_sha, "mp4_count": len(files),
              "unparsed_entries": unparsed, "files": files, "extracted_at_utc": _now()}
    if record["mp4_count"] != expected["mp4_count"]:
        raise RuntimeError(f"{signer}: {record['mp4_count']} mp4, import manifest says {expected['mp4_count']}")
    done.write_text(json.dumps(record, indent=2))
    DATASETS.commit()
    return record


def _arabsign_split_lists():
    """The dataset author's <signer>_{train,test}.csv rows, keyed by archive path."""
    import io
    import zipfile

    import requests

    payload = requests.get(ARABSIGN_SPLITS_URL, timeout=120).content
    if hashlib.sha256(payload).hexdigest() != ARABSIGN_SPLITS_SHA256:
        raise RuntimeError("ArabSign-EncoderDec.zip does not match the pinned sha256")
    rows, csv_sha256 = {}, {}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for signer in ARABSIGN_SIGNERS:
            for split in ("train", "test"):
                name = f"ArabSign-EncoderDec/EncoderDec/data/{signer}_{split}.csv"
                raw = archive.read(name)
                csv_sha256[f"{signer}_{split}.csv"] = hashlib.sha256(raw).hexdigest()
                for line in raw.decode("utf-8-sig").splitlines():
                    fields = line.split(",")
                    tail = "/".join(fields[2].split("/")[-4:])
                    rows[f"{tail}.mp4"] = {"csv": f"{signer}_{split}.csv", "fields": fields}
    return rows, csv_sha256


@APP.function(image=DATA_IMAGE, cpu=2, memory=4096, volumes=POPULATE_VOLUMES, timeout=6 * 3600)
def extract_arabsign():
    """Extracts all six RGB archives and reconciles them with the author's split lists."""
    per_signer = list(extract_arabsign_signer.map(ARABSIGN_SIGNERS))
    DATASETS.reload()
    videos = {path for record in per_signer for path in record["files"]}
    rows, csv_sha256 = _arabsign_split_lists()
    manifest = {
        "source_archives": {r["archive"]: r["archive_sha256"] for r in per_signer},
        "mp4_count": len(videos),
        "split_list_source": ARABSIGN_SPLITS_URL,
        "split_list_csv_sha256": csv_sha256,
        "split_list_rows": len(rows),
        "listed_without_video": sorted(set(rows) - videos),
        "video_without_listing": sorted(videos - set(rows)),
        "split_folder_disagreements": sorted(p for p in rows if p.split("/")[1] != rows[p]["csv"].split("_")[1][:-4]),
        "extracted_by": f"papers/alyami-2024-cslr-comparative/modal_app.py::extract_arabsign (paper {PAPER_ID})",
        "updated_at_utc": _now(),
    }
    (ARABSIGN_VIDEOS / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    DATASETS.commit()
    return manifest


# ---------------------------------------------------------------------------
# Training image and runtime layout.
IMAGE = (modal.Image.from_dockerfile(PAPER_DIR / "Dockerfile", context_dir=PAPER_DIR)
         .add_local_file(PAPER_DIR / "ctc_check.py", "/opt/ctc_check.py"))
RUN_VOLUMES = {"/datasets": DATASETS.read_only(), "/cache/huggingface": HF_CACHE, "/results": RESULTS}
REPOS = {"vac": "/opt/repos/vac", "tlp": "/opt/repos/tlp", "sen": "/opt/repos/sen", "corrnet": "/opt/repos/corrnet"}
GPU = "L40S"  # 48 GB, the memory of each of the paper's RTX A6000 GPUs


def _out(*cmd, cwd=None):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd).stdout.strip()


@APP.function(image=IMAGE, cpu=4, memory=16384, volumes=RUN_VOLUMES, timeout=3600)
def inspect_environment():
    """Versions in the built image, and how the repos' committed Phoenix lists compare."""
    import numpy as np

    report = {
        "torch": _out("python", "-c", "import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version())"),
        "python": _out("python", "--version"),
        "pip_freeze_sha256": hashlib.sha256(_out("pip", "freeze").encode()).hexdigest(),
        "pip_freeze": _out("pip", "freeze").splitlines(),
        "repos": {name: _out("git", "-C", path, "rev-parse", "HEAD") for name, path in REPOS.items()},
        "image_id": os.environ.get("MODAL_IMAGE_ID"),
    }
    committed = {}
    for name in REPOS:
        base = Path(REPOS[name]) / "preprocess/phoenix2014"
        if not (base / "gloss_dict.npy").exists():
            committed[name] = "no committed phoenix2014 lists"
            continue
        entry = {"gloss_dict_size": len(np.load(base / "gloss_dict.npy", allow_pickle=True).item())}
        for split in ("train", "dev", "test"):
            info = np.load(base / f"{split}_info.npy", allow_pickle=True).item()
            clips = [info[k] for k in sorted(k for k in info if isinstance(k, int))]
            entry[split] = {"clips": len(clips), "prefix": info.get("prefix"), "first": clips[0],
                            "rows_sha256": hashlib.sha256("\n".join(c["original_info"] for c in clips).encode()).hexdigest()}
        committed[name] = entry
    report["committed_phoenix_lists"] = committed
    return report


# ---------------------------------------------------------------------------
# 256x256 frames. Every repo reads PNG frames at
# <dataset_root>/features/fullFrame-256x256px/<folder glob>; the archives below
# are built once and staged onto local disk by each run.
PHOENIX_RELEASE = Path("/datasets/rwth-phoenix-2014/raw/phoenix2014-release")
FRAMES_TAR = {name: Path(f"/results/data/{name}-fullFrame-256x256px.tar") for name in ("phoenix2014", "arabsign", "gsl")}
# dataset_root inside each staged archive: the directory holding features/fullFrame-256x256px.
FRAMES_ROOT_IN_TAR = {"phoenix2014": "phoenix-2014-multisigner", "arabsign": ".", "gsl": "."}
# Output of preprocess/dataset_preprocess.py (byte-identical in all four repos) for the
# multisigner set: info lists, gloss dict, STMs. VAC and TLP ship no lists and use it.
VAC_PHOENIX_PREPROCESS = Path("/results/data/vac-preprocess-phoenix2014")
REPOS_WITHOUT_PHOENIX_LISTS = ("vac", "tlp")
# Upstream csv2dict drops pandas row 2390 of train.corpus.csv, this clip.
PHOENIX_BROKEN_CLIP = "13April_2011_Wednesday_tagesschau_default-14"


def _pngs(root):
    return sum(name.endswith(".png") for _, _, names in os.walk(root) for name in names)


def _archive(local_root, member, tar):
    tar.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(f"{tar}.partial")
    subprocess.run(["tar", "--sort=name", "--owner=0", "--group=0", "--numeric-owner", "--mtime=@0",
                    "-cf", str(partial), "-C", str(local_root), member], check=True)
    digest = _sha256(partial)
    partial.rename(tar)
    return {"sha256": digest, "bytes": tar.stat().st_size}


@APP.function(image=IMAGE, cpu=32, memory=65536, ephemeral_disk=512 * 1024, volumes=RUN_VOLUMES, timeout=12 * 3600)
def prepare_frames_phoenix2014():
    """Runs VAC's preprocess/dataset_preprocess.py (identical in all four repos) on the multisigner set.

    The script resizes 210x260 -> 256x256 (cv2 INTER_LANCZOS4) next to the
    originals, so it runs on a local copy; the datasets Volume stays read-only.
    It also writes the info lists, gloss dict and STMs VAC expects.
    """
    tar, manifest_path = FRAMES_TAR["phoenix2014"], Path(f"{FRAMES_TAR['phoenix2014']}.json")
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    started_at, started = _now(), time.monotonic()
    local = Path("/tmp/phoenix2014")
    source = PHOENIX_RELEASE / "phoenix-2014-multisigner"
    files = [Path(d, n) for d, _, names in os.walk(source) for n in names
             if "/features/fullFrame-210x260px/" in d + "/" or "/annotations/" in d + "/"]

    def copy(path):
        target = local / "phoenix-2014-multisigner" / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)

    with ThreadPoolExecutor(256) as pool:
        list(pool.map(copy, files))
    copied = time.monotonic()
    work = Path("/tmp/vac-preprocess")
    shutil.copytree(f"{REPOS['vac']}/preprocess", work)
    command = ["python", "dataset_preprocess.py", "--process-image", "--multiprocessing",
               "--dataset-root", str(local / "phoenix-2014-multisigner")]
    proc = subprocess.run(command, cwd=work, capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(proc.stdout[-3000:] + proc.stderr[-3000:])
    resized = time.monotonic()
    checks = {}
    for split in ("train", "dev", "test"):
        checks[split] = {"resized_png": _pngs(local / "phoenix-2014-multisigner/features/fullFrame-256x256px" / split),
                         "original_png": _pngs(local / "phoenix-2014-multisigner/features/fullFrame-210x260px" / split)}
    broken = local / "phoenix-2014-multisigner/features/fullFrame-210x260px/train" / PHOENIX_BROKEN_CLIP / "1"
    checks["broken_clip_original_png"] = _pngs(broken)
    # Every split except train must be fully resized; train lacks only the clip csv2dict drops.
    passed = (checks["dev"]["resized_png"] == checks["dev"]["original_png"]
              and checks["test"]["resized_png"] == checks["test"]["original_png"]
              and checks["train"]["resized_png"] == checks["train"]["original_png"] - checks["broken_clip_original_png"])
    VAC_PHOENIX_PREPROCESS.mkdir(parents=True, exist_ok=True)
    for path in (work / "phoenix2014").iterdir():
        shutil.copyfile(path, VAC_PHOENIX_PREPROCESS / path.name)
    archive = _archive(local, "phoenix-2014-multisigner/features/fullFrame-256x256px", tar)
    manifest = {
        "source": str(source),
        "method": f"{REPOS['vac']}/preprocess/dataset_preprocess.py --process-image --multiprocessing",
        "vac_commit": _out("git", "-C", REPOS["vac"], "rev-parse", "HEAD"),
        "opencv": _out("python", "-c", "import cv2; print(cv2.__version__)"),
        "checks": checks, "checks_passed": passed,
        "vac_preprocess_outputs": {p.name: _sha256(p) for p in sorted(VAC_PHOENIX_PREPROCESS.iterdir())},
        **archive,
        "started_at_utc": started_at, "finished_at_utc": _now(),
        "seconds": {"copy": round(copied - started), "resize": round(resized - copied), "total": round(time.monotonic() - started)},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    RESULTS.commit()
    return manifest


def _arabsign_sample_id(relative):
    """'01/train/0001/01_0001_(10_03_21_20_37_17)_c.mp4' -> '01_0001_10_03_21_20_37_17'.

    Signer and sentence come from the path, which the author's split lists use;
    35 file stems carry a different prefix (e.g. 02/test/0007/03_01_0007_(...)_c.mp4).
    """
    match = ARABSIGN_ENTRY.match(relative)
    stamp = re.fullmatch(r".*_\(([\d_]+)\)_c", match["stem"]).group(1)
    return f"{match['signer']}_{match['sentence']}_{stamp}"


def _decode_arabsign(job):
    """Decodes one mp4 and writes upstream-style 256x256 PNGs (cv2 INTER_LANCZOS4, as resize_img)."""
    import cv2
    from simple_video_utils.frames import read_frames_exact
    from simple_video_utils.metadata import open_video, video_metadata_from_container

    source, target = job
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    with open_video(source, thread_type="NONE") as video:
        expected = video_metadata_from_container(video).nb_frames
        for frame, index in read_frames_exact(video, return_indices=True):
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(target / f"{index:06d}.png"), cv2.resize(bgr, (256, 256), interpolation=cv2.INTER_LANCZOS4))
            count += 1
    return target.name, count, expected


@APP.function(image=IMAGE, cpu=32, memory=65536, ephemeral_disk=512 * 1024, volumes=RUN_VOLUMES, timeout=12 * 3600)
def prepare_frames_arabsign():
    """Decodes every ArabSign RGB mp4 (simple-video-utils) to 256x256 PNGs under features/.../all/<sample>/."""
    from concurrent.futures import ProcessPoolExecutor

    import av
    import numpy as np
    from simple_video_utils.frames import read_frames_exact
    from simple_video_utils.metadata import open_video

    tar, manifest_path = FRAMES_TAR["arabsign"], Path(f"{FRAMES_TAR['arabsign']}.json")
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    started_at, started = _now(), time.monotonic()
    videos = sorted(p.relative_to(ARABSIGN_VIDEOS).as_posix() for p in ARABSIGN_VIDEOS.rglob("*.mp4"))
    # Channel order check: simple-video-utils frame 0 against PyAV's explicit rgb24 conversion.
    first = str(ARABSIGN_VIDEOS / videos[0])
    with open_video(first, thread_type="NONE") as video:
        svu = next(iter(read_frames_exact(video)))
    with av.open(first) as container:
        rgb = next(container.decode(video=0)).to_ndarray(format="rgb24")
    channel_check = {"svu_equals_rgb24": bool(np.array_equal(svu, rgb)), "shape": list(svu.shape), "dtype": str(svu.dtype)}
    if not channel_check["svu_equals_rgb24"]:
        raise RuntimeError(f"simple-video-utils frames are not rgb24: {channel_check}")
    root = Path("/tmp/arabsign/features/fullFrame-256x256px/all")
    jobs = [(str(ARABSIGN_VIDEOS / v), str(root / _arabsign_sample_id(v))) for v in videos]
    if len({target for _, target in jobs}) != len(jobs):
        raise RuntimeError("ArabSign sample ids are not unique")
    with ProcessPoolExecutor(32) as pool:
        decoded = list(pool.map(_decode_arabsign, jobs, chunksize=4))
    mismatched = [(name, count, expected) for name, count, expected in decoded if count != expected]
    frames = {name: count for name, count, _ in decoded}
    archive = _archive(Path("/tmp/arabsign"), "features/fullFrame-256x256px", tar)
    manifest = {
        "source": str(ARABSIGN_VIDEOS), "videos": len(videos), "frames_total": sum(frames.values()),
        "frames_per_sample": frames, "decoded_vs_container_nb_frames_mismatch": mismatched,
        "method": "simple-video-utils read_frames_exact (rgb24) -> cv2 RGB2BGR -> cv2.resize 256x256 INTER_LANCZOS4 -> PNG",
        "channel_check": channel_check,
        "simple_video_utils": _out("pip", "show", "simple-video-utils").splitlines()[1],
        "opencv": _out("python", "-c", "import cv2; print(cv2.__version__)"),
        **archive,
        "started_at_utc": started_at, "finished_at_utc": _now(), "seconds": round(time.monotonic() - started),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    RESULTS.commit()
    return {k: v for k, v in manifest.items() if k != "frames_per_sample"}


# ---------------------------------------------------------------------------
# Settings. Phoenix Signer-Dep runs each repo's own phoenix2014 recipe. Every
# other setting is written in the same upstream preprocess format (info lists,
# gloss dict, STMs) under a dataset name containing "phoenix": in TLP, SEN and
# CorrNet that substring is what selects the train/dev/test loaders and the
# frame-folder glob, and VAC always reads ./preprocess/phoenix2014/. No
# upstream code changes; only data files and a dataset config are added.
MODELS = {"vac": "vac", "smkd": "vac", "tlp": "tlp", "sen": "sen", "corrnet": "corrnet"}
SETTINGS = {
    "phoenix-sd": {"frames": "phoenix2014"},
    "phoenix-si": {"frames": "phoenix2014", "dataset": "phoenix2014-si5"},
    "phoenix-sd-preflight": {"frames": "phoenix2014", "dataset": "phoenix2014-preflight"},
    "grsl-si": {"frames": "gsl", "dataset": "phoenixfmt-grsl-si"},
    "grsl-unseen": {"frames": "gsl", "dataset": "phoenixfmt-grsl-unseen"},
    "arabsign-sd": {"frames": "arabsign", "dataset": "phoenixfmt-arabsign-sd"},
    **{f"arabsign-loso-{s}": {"frames": "arabsign", "dataset": f"phoenixfmt-arabsign-loso{s}"} for s in ARABSIGN_SIGNERS},
}


def _phoenix_si5_lists():
    """SI5 corpus rows, framed from the multisigner tree (SI5 is a subset of it)."""
    split_of = {}
    for split in ("train", "dev", "test"):
        text = (PHOENIX_RELEASE / f"phoenix-2014-multisigner/annotations/manual/{split}.corpus.csv").read_text(encoding="utf-8")
        split_of.update({line.split("|")[0]: split for line in text.splitlines()[1:]})
    lists = {}
    for split in ("train", "dev", "test"):
        text = (PHOENIX_RELEASE / f"phoenix-2014-signerindependent-SI5/annotations/manual/{split}.SI5.corpus.csv").read_text(encoding="utf-8")
        rows = []
        for line in text.splitlines()[1:]:
            fileid, folder, signer, label = line.split("|")
            if fileid == PHOENIX_BROKEN_CLIP:
                continue  # dropped from training exactly as upstream csv2dict drops it from multisigner train
            rows.append({"fileid": fileid, "folder": f"{split_of[fileid]}/{folder}",
                         "signer": signer, "label": label})
        lists[split] = rows
    return lists


GSL_SPLIT_FILES = {
    "grsl-si": {split: f"GSL SI/gsl_split_SI_{split}.csv" for split in ("train", "dev", "test")},
    "grsl-unseen": {split: f"GSL SD/gsl_{split}_unseen.txt" for split in ("train", "dev", "test")},
}


def _gsl_lists(setting):
    """GSL split files: 'police1_signer3_rep1_sentences/sentences0000|GLOSS GLOSS'."""
    lists = {}
    for split, relative in GSL_SPLIT_FILES[setting].items():
        path = GSL_ROOT / "splits" / relative
        if _sha256(path) != GSL_SPLIT_SHA256[relative]:
            raise RuntimeError(f"{relative} does not match its recorded sha256")
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            folder, label = line.rstrip("\r").split("|")
            rows.append({"fileid": folder.replace("/", "__"), "folder": f"all/{folder}/*.png",
                         "signer": re.search(r"_(signer\d+)_", folder).group(1), "label": label})
        lists[split] = rows
    return lists


def _arabsign_rows():
    """Author split-list rows that have a video, with the sign-language sentence as gloss label."""
    rows, _ = _arabsign_split_lists()
    truth = {}
    for line in (ARABSIGN_RGB / "ArabSignGroundTruth.txt").read_text(encoding="utf-16").splitlines()[1:]:
        if line.strip():
            sentence, arabic, signed = (part.strip() for part in line.split("\t"))
            truth[int(sentence)] = (" ".join(arabic.split()), " ".join(signed.split()))
    available = {p.relative_to(ARABSIGN_VIDEOS).as_posix() for p in ARABSIGN_VIDEOS.rglob("*.mp4")}
    out = []
    for relative, row in sorted(rows.items()):
        if relative not in available:
            continue
        fields = row["fields"]
        sentence = int(fields[1])
        arabic, signed = " ".join(fields[5].split()), " ".join(fields[6].split())
        if (arabic, signed) != truth[sentence]:
            raise RuntimeError(f"{relative}: CSV sentence columns differ from ArabSignGroundTruth.txt")
        sample = _arabsign_sample_id(relative)
        out.append({"fileid": sample, "folder": f"all/{sample}/*.png",
                    "signer": relative.split("/")[0], "split": relative.split("/")[1], "label": signed})
    return out


def _arabsign_lists(setting, dev_rule):
    """ArabSign Signer-Dep (author train/test lists) or one leave-one-signer-out fold.

    dev_rule decides the unpublished validation subset (gate arabsign-protocol).
    'preflight-random-709' is only for cost measurement and never for a target.
    """
    import random

    rows = _arabsign_rows()
    if setting == "arabsign-sd":
        train = [r for r in rows if r["split"] == "train"]
        test = [r for r in rows if r["split"] == "test"]
    else:
        held = setting.rsplit("-", 1)[1]
        train = [r for r in rows if r["signer"] != held]
        test = [r for r in rows if r["signer"] == held]
    if dev_rule != "preflight-random-709":
        raise ValueError(f"ArabSign dev rule {dev_rule!r} is not decided (gate arabsign-protocol)")
    dev = random.Random(0).sample(train, 709)
    chosen = {r["fileid"] for r in dev}
    return {"train": [r for r in train if r["fileid"] not in chosen], "dev": dev, "test": test}


def _setting_lists(setting, dev_rule=None):
    if setting == "phoenix-si":
        return _phoenix_si5_lists()
    if setting.startswith("grsl"):
        return _gsl_lists(setting)
    if setting.startswith("arabsign"):
        return _arabsign_lists(setting, dev_rule)
    raise ValueError(setting)


def _write_dataset(code, dataset, frames_root, lists, vac):
    """Writes info lists, gloss dict and STMs with the repo's own preprocess helpers."""
    import importlib.util

    import numpy as np
    import yaml

    spec = importlib.util.spec_from_file_location("upstream_preprocess", code / "preprocess/dataset_preprocess.py")
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    target, evaluation = code / "preprocess" / dataset, code / "evaluation" / dataset
    target.mkdir(parents=True, exist_ok=True)
    evaluation.mkdir(parents=True, exist_ok=True)
    for name in ("preprocess.sh", "mergectmstm.py"):
        shutil.copy(code / "evaluation/slr_eval" / name, evaluation / name)
    sign_dict, counts = {}, {}
    for split, rows in lists.items():
        info = {"prefix": str(frames_root)}
        for index, row in enumerate(rows):
            num_frames = len(list((Path(frames_root) / "features/fullFrame-256x256px").glob(row["folder"])))
            if num_frames == 0:
                raise RuntimeError(f"{dataset}/{split}: no frames for {row['fileid']}")
            info[index] = {"fileid": row["fileid"], "folder": row["folder"], "signer": row["signer"], "label": row["label"],
                           "num_frames": num_frames,
                           "original_info": f"{row['fileid']}|{row['folder']}|{row['signer']}|{row['label']}"}
        np.save(target / f"{split}_info.npy", info)
        if vac:  # VAC's BaseFeeder hardcodes ./preprocess/phoenix2014/{mode}_info.npy
            (code / "preprocess/phoenix2014").mkdir(exist_ok=True)
            np.save(code / f"preprocess/phoenix2014/{split}_info.npy", info)
        upstream.sign_dict_update(sign_dict, info)
        upstream.generate_gt_stm(info, evaluation / f"{dataset}-groundtruth-{split}.stm")
        counts[split] = {"clips": len(rows), "frames": sum(v["num_frames"] for k, v in info.items() if isinstance(k, int))}
    # gloss dict exactly as dataset_preprocess.py's __main__ builds it
    gloss_dict = {key: [idx + 1, value] for idx, (key, value) in enumerate(sorted(sign_dict.items(), key=lambda d: d[0]))}
    np.save(target / "gloss_dict.npy", gloss_dict)
    (code / f"configs/{dataset}.yaml").write_text(yaml.safe_dump({
        "dataset_root": str(frames_root), "dict_path": f"./preprocess/{dataset}/gloss_dict.npy",
        "evaluation_dir": f"./evaluation/{dataset}", "evaluation_prefix": f"{dataset}-groundtruth"}))
    return {"splits": counts, "glosses": len(gloss_dict),
            "lists_sha256": hashlib.sha256(json.dumps(lists, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}


def _run_config(code, model, dataset, epochs, steps):
    """baseline.yaml of the pinned repo with only the gate-approved keys changed (gate protocol-lr-epochs)."""
    import yaml

    config = yaml.safe_load((code / "configs/baseline.yaml").read_text())
    changes = {"dataset": dataset, "num_epoch": epochs,
               "optimizer_args.base_lr": 0.001, "optimizer_args.step": list(steps)}
    if model == "vac":  # VAC_CSLR README: VAC = SeqCTC 1.0 + ConvCTC 1.0
        changes.update({"loss_weights": {"SeqCTC": 1.0, "ConvCTC": 1.0},
                        "model_args.share_classifier": False, "model_args.weight_norm": False})
    if model == "smkd":  # README: VAC+SMKD additionally sets share_classifier and weight_norm
        changes.update({"loss_weights": {"SeqCTC": 1.0, "ConvCTC": 1.0},
                        "model_args.share_classifier": True, "model_args.weight_norm": True})
    before = json.loads(json.dumps(config))
    for key, value in changes.items():
        node, *path = key.split(".")
        if path:
            config[node][path[0]] = value
        else:
            config[node] = value
    (code / "configs/run.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    delta = {k: {"upstream": (before.get(k.split(".")[0], {}) or {}).get(k.split(".")[1]) if "." in k else before.get(k),
                 "run": v} for k, v in changes.items()}
    return delta


def _stage_frames(name):
    manifest = json.loads(Path(f"{FRAMES_TAR[name]}.json").read_text())
    if not manifest.get("checks_passed", True):
        raise RuntimeError(f"{name} frame archive failed its checks")
    root = Path("/data") / name
    started = time.monotonic()
    if not (root / ".staged").exists():
        root.mkdir(parents=True, exist_ok=True)
        subprocess.run(["tar", "-xf", str(FRAMES_TAR[name]), "-C", str(root)], check=True)
        (root / ".staged").write_text(manifest["sha256"])
    return (root / FRAMES_ROOT_IN_TAR[name]).resolve(), {"frames_tar": str(FRAMES_TAR[name]), "frames_tar_sha256": manifest["sha256"],
                  "stage_seconds": round(time.monotonic() - started, 1)}


def _prepare_code(model, setting, run_dir, epochs, steps, dev_rule=None, lists=None):
    """A private copy of the pinned repo with this setting's data files and run config."""
    repo = MODELS[model]
    code = run_dir / "code"
    if not code.exists():
        shutil.copytree(REPOS[repo], code, symlinks=True)
    frames_root, staged = _stage_frames(SETTINGS[setting]["frames"])
    record = {"repo": repo, "repo_head": _out("git", "-C", str(code), "rev-parse", "HEAD"),
              "tracked_diff_sha256": hashlib.sha256(_out("git", "-C", str(code), "diff", "HEAD").encode()).hexdigest(),
              **staged}
    if setting == "phoenix-sd":
        dataset = "phoenix14" if repo == "vac" else "phoenix2014"
        (code / "dataset").mkdir(exist_ok=True)
        if not (code / "dataset/phoenix2014").exists():  # configs: ./dataset/phoenix2014/phoenix-2014-multisigner
            (code / "dataset/phoenix2014").symlink_to(frames_root.parent)
        if repo in REPOS_WITHOUT_PHOENIX_LISTS:  # README: generate them with preprocess/dataset_preprocess.py
            (code / "preprocess/phoenix2014").mkdir(exist_ok=True)
            for path in VAC_PHOENIX_PREPROCESS.iterdir():
                shutil.copyfile(path, code / "preprocess/phoenix2014" / path.name)
            record["lists"] = "preprocess/dataset_preprocess.py output (prepare_frames_phoenix2014)"
        else:
            record["lists"] = "repo's committed preprocess/phoenix2014"
    else:
        dataset = SETTINGS[setting]["dataset"]
        lists = lists or _setting_lists(setting, dev_rule)
        record["lists"] = _write_dataset(code, dataset, frames_root, lists, vac=repo == "vac")
    record["config_delta"] = _run_config(code, model, dataset, epochs, steps)
    record["dataset"] = dataset
    return code, record


# ---------------------------------------------------------------------------
# Running upstream main.py.
LOG_LINE = re.compile(r"^\[ (?P<time>[^\]]+) \] Epoch (?P<epoch>\d+), (?P<split>\w+) .*?(?P<wer>[\d.]+)%")
TRAIN_RESOURCES = dict(gpu=GPU, cpu=16, memory=65536, ephemeral_disk=512 * 1024, volumes=RUN_VOLUMES)


def _environment(code):
    return {
        "gpu": _out("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"),
        "torch": _out("python", "-c", "import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version())"),
        "repo_head": _out("git", "-C", str(code), "rev-parse", "HEAD"),
        "tracked_diff_sha256": hashlib.sha256(_out("git", "-C", str(code), "diff", "HEAD").encode()).hexdigest(),
        "image_id": os.environ.get("MODAL_IMAGE_ID"),
        "task_id": os.environ.get("MODAL_TASK_ID"),
    }


def _host_memory_used_mib():
    info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return (int(info["MemTotal"].split()[0]) - int(info["MemAvailable"].split()[0])) // 1024


def _run_upstream(command, cwd, work_dir):
    """Runs one upstream command, keeping its log and committing outputs as it goes.

    TLP, SEN and CorrNet ask on stdin whether to wipe an existing work_dir; the
    answer is always no, so a resumed run keeps its checkpoints and logs.
    """
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    peak = {"gpu_mib": 0, "host_mib": 0}

    def commit_periodically():
        while not stop.wait(600):
            RESULTS.commit()

    def sample_memory():
        while not stop.wait(3):
            used = _out("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits").split()
            if used:
                peak["gpu_mib"] = max(peak["gpu_mib"], int(used[0]))
            peak["host_mib"] = max(peak["host_mib"], _host_memory_used_mib())

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
    scoring_failed = "Unexpected error" in (Path(work_dir) / "stdout.log").read_text(errors="replace")
    return {"command": command, "cwd": str(cwd), "exit_code": proc.returncode, "scoring_failed": scoring_failed,
            "started_at_utc": started_at, "finished_at_utc": _now(),
            "duration_seconds": round(time.monotonic() - started, 1),
            "peak_gpu_memory_mib_nvidia_smi": peak["gpu_mib"], "peak_host_memory_mib": peak["host_mib"]}


def _log_entries(work_dir, split):
    """(epoch, wer, unix time) per line upstream logged to <split>.txt; a resumed epoch overwrites."""
    entries = {}
    path = Path(work_dir) / f"{split}.txt"
    for line in path.read_text().splitlines() if path.exists() else []:
        match = LOG_LINE.match(line)
        if match and match["split"] == split:
            stamp = time.mktime(time.strptime(match["time"], "%a %b %d %H:%M:%S %Y"))
            entries[int(match["epoch"])] = (float(match["wer"]), stamp)
    return entries


def _epoch_checkpoints(work_dir):
    found = {}
    for path in Path(work_dir).glob("dev_*_epoch*_model.pt"):
        found[int(re.search(r"_epoch(\d+)_model\.pt$", path.name).group(1))] = path
    return found


def _command(work_dir, *extra):
    # --save-interval 1 keeps a checkpoint per epoch: needed to resume across
    # Modal's 24 h limit and to select the dev-best epoch for VAC/SMKD, whose
    # main.py saves no best model. It changes no training computation.
    return ["python", "main.py", "--config", "./configs/run.yaml", "--device", "0",
            "--save-interval", "1", "--work-dir", f"{work_dir}/", *extra]


def _phoenix_multisigner_lists():
    """The multisigner corpus as upstream csv2dict reads it (broken train clip dropped)."""
    lists = {}
    for split in ("train", "dev", "test"):
        text = (PHOENIX_RELEASE / f"phoenix-2014-multisigner/annotations/manual/{split}.corpus.csv").read_text(encoding="utf-8")
        rows = []
        for line in text.splitlines()[1:]:
            fileid, folder, signer, label = line.split("|")
            if fileid != PHOENIX_BROKEN_CLIP:
                rows.append({"fileid": fileid, "folder": f"{split}/{folder}", "signer": signer, "label": label})
        lists[split] = rows
    return lists


def _preflight_subset(lists, frames_root, n_train=120, n_long=16, n_eval=40, n_long_eval=8):
    """Random clips plus the longest ones, so throughput is typical and peak memory is covered."""
    import random

    def frames(row):
        return len(list((Path(frames_root) / "features/fullFrame-256x256px").glob(row["folder"])))

    rng = random.Random(0)
    subset = {}
    for split, rows in lists.items():
        sized = sorted(((frames(r), i) for i, r in enumerate(rows)), reverse=True)
        longest = [i for _, i in sized[: (n_long if split == "train" else n_long_eval)]]
        rest = [i for i in range(len(rows)) if i not in set(longest)]
        picked = longest + rng.sample(rest, (n_train if split == "train" else n_eval))
        subset[split] = [rows[i] for i in picked]
    full_frames = {split: sum(frames(r) for r in rows) for split, rows in lists.items()}
    return subset, full_frames


@APP.function(image=IMAGE, **TRAIN_RESOURCES, timeout=4 * 3600)
def preflight(model: str, setting: str, run_id: str):
    """Real clips through upstream main.py: 2 epochs, resume a 3rd, then the test phase.

    Also runs ctc_check.py (both arms) on the same subset. Returns the
    measurements the full-run estimate is built from.
    """
    run_dir = Path("/results/runs") / run_id
    frames_root, _ = _stage_frames(SETTINGS[setting]["frames"])
    if setting == "phoenix-sd":
        full_lists = _phoenix_multisigner_lists()
    else:
        full_lists = _setting_lists(setting, dev_rule="preflight-random-709")
    subset, full_frames = _preflight_subset(full_lists, frames_root)
    preflight_setting = "phoenix-sd-preflight" if setting == "phoenix-sd" else setting  # subset as generated lists
    code, record = _prepare_code(model, preflight_setting, run_dir, epochs=40, steps=(20, 35), lists=subset)
    record.update({"setting": setting, "model": model, "full_split_frames": full_frames,
                   "full_split_clips": {s: len(r) for s, r in full_lists.items()},
                   "environment": _environment(code), "steps": []})
    work = str(run_dir / "work")
    for arm in ("as-is", "ctc-without-cudnn"):
        out = run_dir / f"ctc_check-{arm}.json"
        proc = subprocess.run(["python", "/opt/ctc_check.py", "--arm", arm, "--batches", "60",
                               "--work-dir", f"{run_dir}/ctc-{arm}/", "--output", str(out)],
                              cwd=code, input="n\n", capture_output=True, text=True)
        record[f"ctc_check_{arm}"] = json.loads(out.read_text()) if out.exists() else {"exit_code": proc.returncode,
                                                                                         "tail": proc.stdout[-2000:] + proc.stderr[-2000:]}
    steps = (
        lambda: ["--num-epoch", "2"],
        lambda: ["--num-epoch", "3", "--load-checkpoints", str(_epoch_checkpoints(work)[1])],
        lambda: ["--phase", "test", "--load-weights", str(_epoch_checkpoints(work)[2])],
    )
    for extra in steps:
        step = _run_upstream(_command(work, *extra()), code, work)
        record["steps"].append(step)
        if step["exit_code"]:
            RESULTS.commit()
            return record
    dev, test = _log_entries(work, "dev"), _log_entries(work, "test")
    record["dev_wer_by_epoch"] = {e: w for e, (w, _) in dev.items()}
    record["test_phase_wer"] = {"dev": dev[6667][0], "test": test[6667][0]}
    record["subset_frames"] = {s: sum(len(list((frames_root / "features/fullFrame-256x256px").glob(r["folder"]))) for r in rows)
                               for s, rows in subset.items()}
    # Epoch 1 wall time (train + dev eval) from consecutive dev-log timestamps; test-phase
    # dev->test timestamps give eval seconds per test frame.
    record["epoch1_seconds"] = dev[1][1] - dev[0][1]
    record["eval_seconds_per_frame"] = (test[6667][1] - dev[6667][1]) / record["subset_frames"]["test"]
    record["checkpoint_bytes"] = _epoch_checkpoints(work)[2].stat().st_size
    (run_dir / "preflight.json").write_text(json.dumps(record, indent=2, default=str, ensure_ascii=False))
    RESULTS.commit()
    return record


@APP.function(image=IMAGE, **TRAIN_RESOURCES, timeout=24 * 3600, retries=modal.Retries(max_retries=3, initial_delay=60.0))
def train(model: str, setting: str, run_id: str, epochs: int = 40, steps: str = "20,35", dev_rule: str = ""):
    """Trains to `epochs`, resuming from the newest epoch checkpoint when one exists."""
    run_dir = Path("/results/runs") / run_id
    RESULTS.reload()
    code, record = _prepare_code(model, setting, run_dir, epochs, tuple(int(s) for s in steps.split(",")), dev_rule or None)
    work = str(run_dir / "work")
    record.update({"environment": _environment(code), "model": model, "setting": setting})
    checkpoints = _epoch_checkpoints(work)
    extra = []
    if checkpoints:
        if max(checkpoints) + 1 >= epochs:
            return {**record, "already_complete": True, "dev_wer_by_epoch": {e: w for e, (w, _) in _log_entries(work, "dev").items()}}
        extra = ["--load-checkpoints", str(checkpoints[max(checkpoints)])]
    record.update(_run_upstream(_command(work, *extra), code, work))
    record["dev_wer_by_epoch"] = {e: w for e, (w, _) in _log_entries(work, "dev").items()}
    with open(run_dir / "train_calls.jsonl", "a") as f:
        f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
    RESULTS.commit()
    if record["exit_code"]:
        raise RuntimeError(f"upstream training exited {record['exit_code']}")
    return record


@APP.function(image=IMAGE, **TRAIN_RESOURCES, timeout=4 * 3600)
def evaluate(train_run_id: str, eval_run_id: str):
    """Scores the lowest-dev-WER epoch (earliest on ties) once on dev and test."""
    train_dir, run_dir = Path("/results/runs") / train_run_id, Path("/results/runs") / eval_run_id
    first_call = json.loads((train_dir / "train_calls.jsonl").read_text().splitlines()[0])
    code = train_dir / "code"
    _stage_frames(SETTINGS[first_call["setting"]]["frames"])
    history = {e: w for e, (w, _) in _log_entries(train_dir / "work", "dev").items()}
    best = min(history, key=lambda e: (history[e], e))
    checkpoint = _epoch_checkpoints(train_dir / "work")[best]
    work = str(run_dir / "work")
    record = {"train_run_id": train_run_id, "selected_epoch": best, "selected_dev_wer_during_training": history[best],
              "epochs_logged": len(history), "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
              **_environment(code)}
    record.update(_run_upstream(_command(work, "--phase", "test", "--load-weights", str(checkpoint)), code, work))
    record["wer"] = {split: _log_entries(work, split)[6667][0] for split in ("dev", "test")}
    for split in ("dev", "test"):
        hypothesis = Path(work) / f"out.output-hypothesis-{split}.ctm"
        record[f"{split}_hypothesis_ctm_sha256"] = _sha256(hypothesis) if hypothesis.exists() else None
    (run_dir / "evaluation.json").write_text(json.dumps(record, indent=2, default=str))
    RESULTS.commit()
    return record


@APP.local_entrypoint()
def launch_preflight(model: str, setting: str, run_id: str):
    print(json.dumps(preflight.remote(model, setting, run_id), indent=2, default=str, ensure_ascii=False)[:30000])


@APP.local_entrypoint()
def launch_train(model: str, setting: str, run_id: str, epochs: int = 40, steps: str = "20,35", dev_rule: str = ""):
    print(train.spawn(model, setting, run_id, epochs, steps, dev_rule).object_id)


@APP.local_entrypoint()
def launch_evaluate(train_run_id: str, eval_run_id: str):
    print(json.dumps(evaluate.remote(train_run_id, eval_run_id), indent=2, default=str))


def _gsl_zip_frames(zip_path):
    """Resizes every jpg frame in one GSL zip to a 256x256 PNG (cv2 INTER_LANCZOS4, as resize_img)."""
    import zipfile

    import cv2
    import numpy as np

    root = Path("/tmp/gsl/features/fullFrame-256x256px/all")
    counts = {}
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if not name.lower().endswith(".jpg"):
                continue
            folder, frame = name.rsplit("/", 1)
            image = cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_COLOR)
            target = root / folder / f"{frame[:-4]}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target), cv2.resize(image, (256, 256), interpolation=cv2.INTER_LANCZOS4))
            counts[folder] = counts.get(folder, 0) + 1
    return Path(zip_path).name, counts


@APP.function(image=IMAGE, cpu=32, memory=65536, ephemeral_disk=512 * 1024, volumes=RUN_VOLUMES, timeout=12 * 3600)
def prepare_frames_gsl():
    """Resizes the frames of the 15 GSL RGB zips to 256x256 PNGs under features/.../all/<folder>/."""
    from concurrent.futures import ProcessPoolExecutor

    tar, manifest_path = FRAMES_TAR["gsl"], Path(f"{FRAMES_TAR['gsl']}.json")
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    started_at, started = _now(), time.monotonic()
    gsl_manifest = json.loads((GSL_ROOT / "manifest.json").read_text())
    zips = sorted(str(GSL_ROOT / "zenodo-3941811" / key) for key in gsl_manifest["zenodo_files"] if key != "supplementary.zip")
    with ProcessPoolExecutor(len(zips)) as pool:
        per_zip = dict(pool.map(_gsl_zip_frames, zips))
    frames = {folder: n for counts in per_zip.values() for folder, n in counts.items()}
    rows = {r["folder"][len("all/"):-len("/*.png")] for setting in ("grsl-si", "grsl-unseen")
            for split in _gsl_lists(setting).values() for r in split}
    archive = _archive(Path("/tmp/gsl"), "features/fullFrame-256x256px", tar)
    manifest = {
        "source": [Path(z).name for z in zips], "zenodo_sha256": {k: v["sha256"] for k, v in gsl_manifest["zenodo_files"].items()},
        "folders": len(frames), "frames_total": sum(frames.values()),
        "frames_per_zip": {name: sum(c.values()) for name, c in per_zip.items()},
        "split_rows_without_frames": sorted(rows - set(frames)), "checks_passed": not (rows - set(frames)),
        "method": "zip jpg -> cv2.imdecode -> cv2.resize 256x256 INTER_LANCZOS4 -> PNG",
        "opencv": _out("python", "-c", "import cv2; print(cv2.__version__)"),
        **archive,
        "started_at_utc": started_at, "finished_at_utc": _now(), "seconds": round(time.monotonic() - started),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    RESULTS.commit()
    return manifest


@APP.local_entrypoint()
def launch_prepare_frames(dataset: str):
    function = {"phoenix2014": prepare_frames_phoenix2014, "arabsign": prepare_frames_arabsign,
                "gsl": prepare_frames_gsl}[dataset]
    print(json.dumps(function.remote(), indent=2, default=str)[:20000])


@APP.local_entrypoint()
def launch_inspect_environment():
    print(json.dumps(inspect_environment.remote(), indent=2, default=str))


@APP.local_entrypoint()
def launch_populate_gsl():
    print(json.dumps(populate_gsl.remote(), indent=2))


@APP.local_entrypoint()
def launch_extract_arabsign():
    print(json.dumps(extract_arabsign.remote(), indent=2, ensure_ascii=False))
