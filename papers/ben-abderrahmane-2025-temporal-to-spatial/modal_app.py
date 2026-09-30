"""Modal entry points for the Ben-Abderrahmane 2025 ArSL ResNet reproduction."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import subprocess
import urllib.request
from pathlib import Path

import modal


PAPER_ID = "8b224ecbd42da766efba45d17b34b3a1255c2345"
DATASET_SLUG = "arab-sign"
RGB_DIR = Path("/datasets") / DATASET_SLUG / "RGB"
SOURCE_FOLDER = "https://drive.google.com/drive/folders/1twRzL8fqjbq5dp6jV-V00-O8gD1BZMTN"
# Drive file ids with the exact byte sizes read from Content-Range on 2026-09-18.
# The paper's ArabSign row "Total Size 18 GB" matches this modality alone.
RGB_ARCHIVES = {
    "01.7z": ("1jsjy4Fe9BffXxarBEFtTCGedRaVinJVt", 3471839354),
    "02.7z": ("1la1ePLMd8dxNDORvYaqOX3Y9jf8HMNot", 3465471202),
    "03.7z": ("1tqc_tFjWlVHqooir-69zJaj2v4UAMaiA", 3173623708),
    "04.7z": ("1Ge6N4LMRbCaTD5gBZXqdwQHxD3PdKlcU", 3164859125),
    "05.7z": ("15OHkXbJu0lyBQtP_9XD1ToMNW9Ixeplz", 3272313079),
    "06.7z": ("1D9q7yBFTgGdTLYV3NGjBL0acAvENjvLx", 3074093776),
}

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
# Paper §III: "Each input sample consists of 10 consecutive frames ... resulting in
# an input shape of (10, 33, 2)". The paper does not say which window; see README.
FRAME_COUNT = 10
WINDOW_START = 0
LANDMARK_COUNT = 33
CLASS_COUNT = 50

app = modal.App("repro-ben-abderrahmane-2025")
# A download and archive-inspection job needs no GPU base image.
image = modal.Image.debian_slim(python_version="3.12").apt_install("p7zip-full")
# Pose extraction uses the study image for simple-video-utils, plus the MediaPipe
# pin already proven in this repository by papers/ahmad-2022-intelligent-landmarks.
pose_image = (
    modal.Image.from_dockerfile(REPOSITORY_ROOT / "Dockerfile", context_dir=REPOSITORY_ROOT)
    .apt_install("p7zip-full")
    .pip_install("mediapipe==0.10.18")
)
# Paper §IV: Keras on a TensorFlow backend, trained on CPU. scikit-learn supplies the
# metric implementation, pinned as in papers/ahmad-2022-intelligent-landmarks.
train_image = modal.Image.from_registry("tensorflow/tensorflow:2.17.0").pip_install(
    "scikit-learn==1.6.1"
)
datasets = modal.Volume.from_name("datasets", create_if_missing=False)
cache = modal.Volume.from_name("huggingface-cache", create_if_missing=False)
results = modal.Volume.from_name(
    "ben-abderrahmane-2025-results", create_if_missing=True, version=2
)


def _download(file_id: str, destination: Path) -> str:
    """Stream one public Drive file to destination, returning its SHA-256."""
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    partial = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as response:
        if response.status != 200:
            raise RuntimeError(f"{destination.name}: HTTP {response.status}")
        with partial.open("wb") as handle:
            while chunk := response.read(8 * 1024 * 1024):
                digest.update(chunk)
                handle.write(chunk)
    partial.rename(destination)
    return digest.hexdigest()


def _entries(archive: Path) -> list[str]:
    """Return non-directory entry paths from an archive listing."""
    listing = subprocess.run(
        ["7z", "l", "-ba", "-slt", str(archive)],
        check=True, capture_output=True, text=True,
    ).stdout
    paths, current = [], None
    for line in listing.splitlines():
        if line.startswith("Path = "):
            current = line[len("Path = "):]
        elif line.startswith("Attributes = ") and current is not None:
            if "D" not in line[len("Attributes = "):].split("_")[0]:
                paths.append(current)
            current = None
    return paths


@app.function(
    image=image,
    volumes={"/datasets": datasets, "/cache/huggingface": cache},
    cpu=4,
    memory=8192,
    timeout=6 * 60 * 60,
    env={"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub"},
)
def populate_arabsign_rgb() -> dict[str, object]:
    """Idempotently add the six ArabSign RGB signer archives to the shared Volume.

    An earlier transfer populated Skeleton in full, Depth partially, and RGB with
    only its two text files. The paper runs MediaPipe pose over colour frames, so
    RGB is the modality the reproduction needs.
    """
    RGB_DIR.mkdir(parents=True, exist_ok=True)
    files, skipped = {}, []

    for name, (file_id, expected_size) in sorted(RGB_ARCHIVES.items()):
        target = RGB_DIR / name
        if target.exists() and target.stat().st_size == expected_size:
            skipped.append(name)
            sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
        else:
            if target.exists():
                raise RuntimeError(
                    f"{name}: refusing to overwrite an existing file of "
                    f"{target.stat().st_size} bytes, expected {expected_size}"
                )
            sha256 = _download(file_id, target)
            actual = target.stat().st_size
            if actual != expected_size:
                target.unlink()
                raise RuntimeError(f"{name}: got {actual} bytes, expected {expected_size}")
            datasets.commit()

        files[name] = {
            "drive_file_id": file_id,
            "size": expected_size,
            "sha256": sha256,
            "mp4_count": sum(1 for e in _entries(target) if e.lower().endswith(".mp4")),
        }

    record = {
        "source_url": SOURCE_FOLDER,
        "source_folder": "ArabSign/RGB",
        "destination": f"{DATASET_SLUG}/RGB",
        "paper_id": PAPER_ID,
        "transferred_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "modal_function_call_id": modal.current_function_call_id(),
        "permission_basis": (
            "Public download links published on the official ArabSign project page "
            "https://hamzah-luqman.github.io/ArabSign/. Licence recorded by the REPRO-SIGN "
            "dataset record as non-commercial only; the archives carry a citation request "
            "and no licence text. Transfer requested by the assignee on 2026-09-18."
        ),
        "scope": (
            "Preserve supplied files only. Verified by exact byte size against Content-Range, "
            "SHA-256 of the received bytes, and a 7z listing that parses the archive footer. "
            "No full archive extraction test was run. No redistribution, dataset identity, "
            "split, or training-readiness claim."
        ),
        "files": files,
        "skipped_already_present": skipped,
        "total_bytes": sum(entry["size"] for entry in files.values()),
        "total_mp4_count": sum(entry["mp4_count"] for entry in files.values()),
    }

    import_id = hashlib.sha256(f"{SOURCE_FOLDER}/RGB".encode()).hexdigest()[:16]
    import_dir = Path("/datasets") / DATASET_SLUG / "_drive_imports" / import_id
    import_dir.mkdir(parents=True, exist_ok=True)
    (import_dir / "manifest.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    datasets.commit()
    return record


def _missing_vs_skeleton(videos: list[tuple[str, str]]) -> dict[str, object]:
    """Name the RGB entries absent relative to the complete Skeleton modality."""
    def key(path: str) -> str:
        # RGB entries end "_c.mp4", Skeleton entries "_c_s.mat"; compare on the shared stem.
        head, _, stem = path.rpartition("/")
        stem = stem.rsplit(".", 1)[0]
        if stem.endswith("_s"):
            stem = stem[:-2]
        return f"{head}/{stem}"

    rgb = {key(path) for _, path in videos}
    skeleton, samples = set(), []
    for archive in sorted((Path("/datasets") / DATASET_SLUG / "Skeleton").glob("*.7z")):
        for path in _entries(archive):
            skeleton.add(key(path))
            if len(samples) < 3:
                samples.append(path)
    return {
        "skeleton_path_samples": samples,
        "rgb_keys": len(rgb),
        "skeleton_keys": len(skeleton),
        "missing_from_rgb": sorted(skeleton - rgb),
        "extra_in_rgb": sorted(rgb - skeleton)[:10],
        "missing_count": len(skeleton - rgb),
    }


def _component_tally(videos: list[tuple[str, str]]) -> dict[str, dict[str, int]]:
    """Tally the second path component per archive, e.g. a train/test directory level."""
    tally: dict[str, dict[str, int]] = {}
    for archive, path in videos:
        parts = path.split("/")
        key = parts[1] if len(parts) > 1 else "(flat)"
        tally.setdefault(key, {})[archive] = tally.setdefault(key, {}).get(archive, 0) + 1
    return tally


@app.function(
    image=image,
    volumes={"/datasets": datasets.read_only(), "/cache/huggingface": cache},
    cpu=4,
    memory=8192,
    timeout=60 * 60,
    env={"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub"},
)
def verify_arabsign_rgb() -> dict[str, object]:
    """Read-only audit of the RGB archives, establishing every dataset claim in README.md.

    Reports the video count, that every entry is an mp4 whose path parses under ENTRY,
    the number of sentence ids, the shipped train/test tally, the label-file hashes
    across modalities, and the cross-check against Skeleton that shows the RGB release
    is five videos short of the 9,335 both papers claim.
    """
    per_archive: dict[str, object] = {}
    extensions: dict[str, int] = {}
    samples: list[str] = []
    all_videos: list[tuple[str, str]] = []
    unparsed: list[str] = []

    for name in sorted(RGB_ARCHIVES):
        paths = _entries(RGB_DIR / name)
        videos = []
        for path in paths:
            extensions[Path(path).suffix.lower() or "(none)"] = (
                extensions.get(Path(path).suffix.lower() or "(none)", 0) + 1
            )
            if path.lower().endswith(".mp4"):
                videos.append(path)
                if ENTRY.match(path) is None:
                    unparsed.append(f"{name}:{path}")
        samples.extend(sorted(paths)[:2])
        all_videos.extend((name, path) for path in videos)
        per_archive[name] = {"entries": len(paths), "videos": len(videos)}

    sentences = {
        match["sentence"] for _, path in all_videos if (match := ENTRY.match(path))
    }

    # The claimed 9,335 is cross-checked against the modalities the earlier transfer left.
    other_modalities: dict[str, object] = {}
    for modality in ("Skeleton", "Depth"):
        archives = sorted((Path("/datasets") / DATASET_SLUG / modality).glob("*.7z"))
        counts = {archive.name: len(_entries(archive)) for archive in archives}
        other_modalities[modality] = {
            "archives_present": sorted(counts),
            "entries_per_archive": counts,
            "total_entries": sum(counts.values()),
        }

    labels = {}
    for modality in ("RGB", "Depth", "Skeleton"):
        candidate = Path("/datasets") / DATASET_SLUG / modality / "ArabSignGroundTruth.txt"
        if candidate.exists():
            raw = candidate.read_bytes()
            labels[modality] = {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

    return {
        "per_archive": per_archive,
        "total_videos": len(all_videos),
        "extensions": dict(sorted(extensions.items())),
        "path_samples": samples[:12],
        "sentence_ids_seen": len(sentences),
        "unparsed_count": len(unparsed),
        "unparsed_paths": unparsed[:20],
        "split_tally": _component_tally(all_videos),
        "label_files": labels,
        "other_modalities": other_modalities,
        "rgb_missing_vs_skeleton": _missing_vs_skeleton(all_videos),
    }


ENTRY = re.compile(r"^(?P<signer>\d{2})/(?P<split>train|test)/(?P<sentence>\d{4})/(?P<stem>.+)\.mp4$")


def _pose_frames(path: str, mode: str) -> list:
    """Return per-frame landmarks for one video; None where nothing was detected.

    Two readings of the paper are supported and are recorded as gate alternatives.
    "first10" follows the target paper §III ("10 consecutive frames") and keeps
    MediaPipe's temporal tracking, since the frames are adjacent. "uniform10"
    follows reference [19] ("convert the input video into a sequence of ten
    frames") and disables tracking, since sampled frames are not adjacent and a
    temporal prior across them would be false. thread_type="NONE" because a
    process pool supplies the parallelism; nested codec threads would
    oversubscribe the assigned CPUs. [19] states the standard pose model is used,
    which is model_complexity=1. MediaPipe is constructed per video so tracking
    state never carries between videos.
    """
    import mediapipe as mp
    import numpy as np
    from simple_video_utils.frames import read_frames_exact
    from simple_video_utils.metadata import open_video, video_metadata_from_container

    if mode not in ("first10", "uniform10"):
        raise ValueError(f"unknown sampling mode: {mode}")

    frames: list = []
    with mp.solutions.pose.Pose(
        static_image_mode=(mode == "uniform10"), model_complexity=1
    ) as pose, open_video(path, thread_type="NONE") as container:
        if mode == "first10":
            stream = read_frames_exact(
                container, start_frame=WINDOW_START, end_frame=WINDOW_START + FRAME_COUNT - 1
            )
            for frame in stream:
                result = pose.process(frame)
                frames.append(_landmarks(result))
        else:
            metadata = video_metadata_from_container(container)
            if not metadata.nb_frames or metadata.nb_frames < FRAME_COUNT:
                raise RuntimeError(f"{path}: {metadata.nb_frames} frames, need {FRAME_COUNT}")
            indices = np.rint(
                np.linspace(0, metadata.nb_frames - 1, FRAME_COUNT)
            ).astype(int)
            wanted = set(indices.tolist())
            retained: dict[int, object] = {}
            for frame, index in read_frames_exact(container, return_indices=True):
                if index in wanted:
                    retained[index] = _landmarks(pose.process(frame))
            missing_indices = [i for i in indices if i not in retained]
            if missing_indices:
                raise RuntimeError(f"{path}: could not decode frames {missing_indices}")
            frames = [retained[i] for i in indices]

    if len(frames) != FRAME_COUNT:
        raise RuntimeError(f"{path}: decoded {len(frames)} frames, expected {FRAME_COUNT}")
    return frames


def _landmarks(result) -> "object":
    if result.pose_landmarks is None:
        return None
    return [[landmark.x, landmark.y] for landmark in result.pose_landmarks.landmark]


def _pose_sequence(path: str, mode: str) -> tuple["object", list[int]]:
    """Return the paper's (10, 33, 2) input plus the indices that had no detection.

    Neither paper specifies a policy for undetected frames. Best effort: carry the
    nearest detected pose forward, or backward for a gap at the clip start. The
    input is a trajectory, so repeating the adjacent pose approximates the unknown
    far better than zeros, which the network would read as a body at the origin.
    A video with no detection at all cannot be filled and raises.
    """
    import numpy as np

    frames = _pose_frames(path, mode)
    missing = [index for index, value in enumerate(frames) if value is None]
    if len(missing) == FRAME_COUNT:
        raise RuntimeError(f"{path}: no pose detected in any of its {FRAME_COUNT} frames")

    previous = None
    for index, value in enumerate(frames):
        if value is None:
            frames[index] = previous
        else:
            previous = value
    following = None
    for index in range(FRAME_COUNT - 1, -1, -1):
        if frames[index] is None:
            frames[index] = following
        else:
            following = frames[index]

    sequence = np.asarray(frames, dtype=np.float32)
    if sequence.shape != (FRAME_COUNT, LANDMARK_COUNT, 2):
        raise RuntimeError(f"{path}: got {sequence.shape}, expected {(FRAME_COUNT, LANDMARK_COUNT, 2)}")
    return sequence, missing


def _extract_one(job: tuple[str, str, str]) -> tuple[str, "object", list[int]]:
    key, path, mode = job
    sequence, missing = _pose_sequence(path, mode)
    return key, sequence, missing





def _unpack(archive: str) -> tuple[list[tuple[str, str]], dict[str, tuple[str, int]]]:
    """Extract one signer archive locally and index its videos by entry path."""
    workspace = Path("/tmp/arabsign") / archive.removesuffix(".7z")
    workspace.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["7z", "x", "-bd", "-y", f"-o{workspace}", str(RGB_DIR / archive)],
        check=True, capture_output=True,
    )
    jobs, meta = [], {}
    for video in sorted(workspace.rglob("*.mp4")):
        relative = video.relative_to(workspace).as_posix()
        match = ENTRY.match(relative)
        if match is None:
            raise RuntimeError(f"unexpected entry path in {archive}: {relative}")
        jobs.append((relative, str(video)))
        meta[relative] = (match["split"], int(match["sentence"]) - 1)
    return jobs, meta


@app.function(
    image=pose_image,
    volumes={
        "/datasets": datasets.read_only(),
        "/cache/huggingface": cache,
        "/results": results,
    },
    cpu=32,
    memory=32768,
    timeout=4 * 60 * 60,
    env={"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub"},
)
def extract_pose_features(archive: str, mode: str = "first10", limit: int = 0) -> dict[str, object]:
    """Extract the paper's MediaPipe pose inputs for one signer archive.

    A positive `limit` processes only the first N videos and writes nothing; it
    exists to exercise the real path cheaply before the full extraction.
    """
    import multiprocessing as multiproc
    import numpy as np

    jobs, meta = _unpack(archive)
    jobs = [(key, path, mode) for key, path in jobs]

    if limit:
        jobs = jobs[:limit]

    with multiproc.get_context("fork").Pool(processes=32) as pool:
        produced = list(pool.imap_unordered(_extract_one, jobs, chunksize=4))
    extracted = {key: sequence for key, sequence, _ in produced}
    gaps = {key: indices for key, _, indices in produced if indices}

    keys = sorted(extracted)
    features = np.stack([extracted[key] for key in keys])
    labels = np.array([meta[key][1] for key in keys], dtype=np.int16)
    splits = np.array([meta[key][0] for key in keys])
    if labels.min() < 0 or labels.max() >= CLASS_COUNT:
        raise RuntimeError(f"{archive}: sentence ids outside 1..{CLASS_COUNT}")

    report = {
        "archive": archive,
        "mode": mode,
        "videos": len(keys),
        "train": int((splits == "train").sum()),
        "test": int((splits == "test").sum()),
        "sentences": int(len(set(labels.tolist()))),
        "feature_shape": list(features.shape),
        "value_range": [float(features.min()), float(features.max())],
        "videos_with_filled_frames": len(gaps),
        "filled_frames": sum(len(v) for v in gaps.values()),
        "filled_by_frame_index": dict(sorted(
            (index, sum(index in v for v in gaps.values()))
            for index in {i for v in gaps.values() for i in v}
        )),
        "filled_examples": {key: gaps[key] for key in sorted(gaps)[:10]},
    }
    if limit:
        report["smoke_test"] = f"first {limit} videos only; nothing written"
        return report

    destination = Path(f"/results/pose-{mode}")
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / f"{archive.removesuffix('.7z')}.npz"
    filled = np.array([len(gaps.get(key, ())) for key in keys], dtype=np.int8)
    np.savez_compressed(
        output, features=features, labels=labels, splits=splits,
        keys=np.array(keys), filled_frames=filled,
    )
    results.commit()
    report["output"] = f"modal://volume/ben-abderrahmane-2025-results/pose-{mode}/{output.name}"
    report["sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    return report


@app.function(
    image=pose_image,
    volumes={"/results": results.read_only(), "/cache/huggingface": cache},
    cpu=4,
    memory=16384,
    timeout=30 * 60,
    env={"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub"},
)
def summarize_features(mode: str = "first10") -> dict[str, object]:
    """Read-only audit of the extracted pose feature set."""
    import numpy as np

    per_signer, keys_seen, labels_all, low, high = {}, set(), [], [], []
    for path in sorted(Path(f"/results/pose-{mode}").glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            features, labels, splits = data["features"], data["labels"], data["splits"]
            keys, filled = data["keys"], data["filled_frames"]
        keys_seen.update(keys.tolist())
        labels_all.append(labels)
        low.append(float(features.min()))
        high.append(float(features.max()))
        per_signer[path.stem] = {
            "videos": len(keys),
            "train": int((splits == "train").sum()),
            "test": int((splits == "test").sum()),
            "shape": list(features.shape),
            "videos_filled": int((filled > 0).sum()),
            "frames_filled": int(filled.sum()),
            "finite": bool(np.isfinite(features).all()),
        }

    totals = {
        key: sum(signer[key] for signer in per_signer.values())
        for key in ("videos", "train", "test", "videos_filled", "frames_filled")
    }
    labels = np.concatenate(labels_all)
    counts = np.bincount(labels, minlength=CLASS_COUNT)
    return {
        "mode": mode,
        "per_signer": per_signer,
        "totals": totals,
        "unique_keys": len(keys_seen),
        "classes_present": int((counts > 0).sum()),
        "per_class_min": int(counts.min()),
        "per_class_max": int(counts.max()),
        "value_range": [min(low), max(high)],
        "filled_fraction_of_frames": totals["frames_filled"] / (totals["videos"] * FRAME_COUNT),
    }


# Paper §IV: Adam, categorical cross-entropy, batch 32, 100 epochs. The learning rate
# is not in the target paper; [19] §5 states 0.001 and §IV says its parameters were
# adopted. Neither paper states a seed; one is pinned so the run is repeatable, and a
# single run is reported because the paper reports single values rather than a mean.
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
SEED = 2026
EXPECTED_PARAMETERS = 755346


def _build_model():
    """Table I, with the block form that reaches its stated 755,346 total exactly."""
    from tensorflow import keras
    from tensorflow.keras import layers

    def residual_block(x, filters: int, stride: int):
        # Table I's block 1 is exactly two 3x3 convolutions, so there is no batch
        # normalisation. A 1x1 projection carries the shortcut when width or stride
        # changes; those projections are precisely the parameters Table I's rows omit.
        y = layers.Conv2D(filters, 3, strides=stride, padding="same", activation="relu")(x)
        y = layers.Conv2D(filters, 3, strides=1, padding="same")(y)
        shortcut = x
        if stride != 1 or x.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same")(x)
        return layers.ReLU()(layers.Add()([y, shortcut]))

    inputs = keras.Input(shape=(FRAME_COUNT, LANDMARK_COUNT, 2))
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = residual_block(x, 32, 1)
    x = residual_block(x, 64, 2)
    x = residual_block(x, 128, 2)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(CLASS_COUNT, activation="softmax")(x)
    return keras.Model(inputs, outputs)


@app.function(
    image=train_image,
    volumes={"/results": results, "/cache/huggingface": cache},
    cpu=16,
    memory=32768,
    timeout=6 * 60 * 60,
    env={"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub"},
)
def train_model(mode: str = "first10") -> dict[str, object]:
    """Train the Table I model on one sampling alternative and evaluate Table III."""
    import datetime as dt
    import time
    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import precision_recall_fscore_support
    from tensorflow import keras

    keras.utils.set_random_seed(SEED)

    features, labels, splits = [], [], []
    for path in sorted(Path(f"/results/pose-{mode}").glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            features.append(data["features"]); labels.append(data["labels"])
            splits.append(data["splits"])
    features = np.concatenate(features)
    labels = np.concatenate(labels).astype(np.int32)
    splits = np.concatenate(splits)
    if len(features) == 0:
        raise RuntimeError(f"no features found for mode {mode}")

    is_train = splits == "train"
    x_train, y_train = features[is_train], labels[is_train]
    x_test, y_test = features[~is_train], labels[~is_train]

    model = _build_model()
    parameters = int(model.count_params())
    if parameters != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"model has {parameters:,} parameters, expected the paper's {EXPECTED_PARAMETERS:,}"
        )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    started_at = dt.datetime.now(dt.timezone.utc)
    started = time.monotonic()
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2,
    )
    duration = time.monotonic() - started

    predictions = np.argmax(model.predict(x_test, batch_size=256, verbose=0), axis=1)
    accuracy = float((predictions == y_test).mean())
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="weighted", zero_division=0
    )
    record = {
        "mode": mode,
        "seed": SEED,
        "parameters": parameters,
        "train_samples": int(is_train.sum()),
        "test_samples": int((~is_train).sum()),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "duration_seconds": duration,
        "modal_function_call_id": modal.current_function_call_id(),
        "tensorflow": tf.__version__,
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "metrics": {
            "accuracy": accuracy,
            "precision_weighted": float(precision),
            "recall_weighted": float(recall),
            "f1_weighted": float(f1),
        },
        "paper_table_iii": {
            "accuracy": 0.9777, "precision": 0.9790, "recall": 0.9777, "f1": 0.9773,
        },
    }
    destination = Path(f"/results/runs/{mode}")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "run.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        destination / "history.npz",
        **{k: np.asarray(v, dtype=np.float32) for k, v in history.history.items()},
        predictions=predictions.astype(np.int16), y_test=y_test.astype(np.int16),
    )
    results.commit()
    return record


@app.local_entrypoint()
def populate() -> None:
    print(json.dumps(populate_arabsign_rgb.remote(), indent=2, sort_keys=True))


@app.local_entrypoint()
def train(mode: str = "first10") -> None:
    print(json.dumps(train_model.remote(mode), indent=2, sort_keys=True))


@app.local_entrypoint()
def summary(mode: str = "first10") -> None:
    print(json.dumps(summarize_features.remote(mode), indent=2, sort_keys=True))


@app.local_entrypoint()
def smoke(mode: str = "first10") -> None:
    print(json.dumps(extract_pose_features.remote("01.7z", mode=mode, limit=24), indent=2, sort_keys=True))


@app.local_entrypoint()
def extract(mode: str = "first10") -> None:
    reports = list(extract_pose_features.map(sorted(RGB_ARCHIVES), kwargs={"mode": mode}))
    print(json.dumps(reports, indent=2, sort_keys=True))
    print(json.dumps({
        "total_videos": sum(r["videos"] for r in reports),
        "total_train": sum(r["train"] for r in reports),
        "total_test": sum(r["test"] for r in reports),
    }, indent=2, sort_keys=True))


@app.local_entrypoint()
def verify() -> None:
    print(json.dumps(verify_arabsign_rgb.remote(), indent=2, sort_keys=True))
