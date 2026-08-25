"""Conditional reconstruction of the paper's ISL-HS landmark classifier.

The paper specifies the raw 440-feature extractor but omits reduction and
evaluation details.  This program makes every resulting choice explicit in its
run manifest; its values must not be presented as Table III reproduction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np
import sklearn
from simple_video_utils.frames import read_frames_exact
from simple_video_utils.metadata import open_video
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


# Paper Section III.B: use exactly the first 60 frames of each ISL-HS video.
FRAME_LIMIT = 60
# Decision: the paper has no seed.  Pin one for a repeatable conditional run.
SEED = 2026
# Decision: Section II.B says only "correlation and dimensionality reduction".
# A train-fold-only 0.95 absolute-Pearson filter makes that choice auditable.
CORRELATION_THRESHOLD = 0.95
VIDEO_NAME = re.compile(r"Person(?P<person>[1-6])/(?P<label>[a-z]) \((?P<take>[1-3])\)\.mov$")


@dataclass(frozen=True)
class Video:
    path: Path
    label: int
    group: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def videos(data_root: Path, per_class: int) -> list[Video]:
    discovered: list[Video] = []
    for path in sorted((data_root / "videos").rglob("*.mov")):
        match = VIDEO_NAME.search(path.relative_to(data_root / "videos").as_posix())
        if match is None:
            raise ValueError(f"unexpected ISL-HS video path: {path}")
        discovered.append(
            Video(
                path=path,
                label=ord(match["label"]) - ord("a"),
                group=f"person{match['person']}-{match['label']}-{match['take']}",
            )
        )
    if len(discovered) != 468:
        raise ValueError(f"expected 468 ISL-HS videos, found {len(discovered)}")
    # Preserve source ordering, but cap each class for a cheap representative preflight.
    selected = []
    for label in range(26):
        label_videos = [video for video in discovered if video.label == label]
        if len(label_videos) != 18:
            raise ValueError(f"expected 18 videos for label {chr(label + ord('a'))}")
        selected.extend(label_videos[:per_class])
    return selected


def raw_features(points: np.ndarray) -> np.ndarray:
    """Implement Algorithm 1: 420 ordered angles plus 20 ordered line values."""
    if points.shape != (21, 2):
        raise ValueError(f"expected 21 two-dimensional landmarks, got {points.shape}")
    delta = points[None, :, :] - points[:, None, :]
    slopes = np.divide(delta[:, :, 1], delta[:, :, 0])
    angles = np.arctan(slopes)
    landmark_values = angles[~np.eye(21, dtype=bool)]
    finger_indices = ((0, 4), (5, 8), (9, 12), (13, 16), (17, 20))
    finger_slopes = np.array([slopes[start, end] for start, end in finger_indices])
    line_values = np.abs(
        np.divide(
            finger_slopes[None, :] - finger_slopes[:, None],
            1 + finger_slopes[:, None] * finger_slopes[None, :],
        )
    )[~np.eye(5, dtype=bool)]
    values = np.concatenate((landmark_values, line_values)).astype(np.float32)
    # Decision: the equations omit a zero-division policy.  Preserve vertical
    # slopes through atan(inf), and map indeterminate/non-finite values to 0.
    return np.nan_to_num(values, nan=0.0, posinf=np.pi / 2, neginf=-np.pi / 2)


def extract(video_list: list[Video]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Decode with simple-video-utils, then detect MediaPipe Hands per video."""
    features: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    detected = 0
    examined = 0
    for video_number, video in enumerate(video_list, start=1):
        # Decision: paper passes sequential frames one by one.  Keep Hands'
        # default temporal tracking within a video, reset it between videos,
        # and restrict this single-hand source to one detected hand.
        with mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands, open_video(str(video.path), thread_type="AUTO") as container:
            for frame in read_frames_exact(container, start_frame=0, end_frame=FRAME_LIMIT - 1):
                examined += 1
                result = hands.process(frame)  # simple-video-utils supplies contiguous RGB.
                if not result.multi_hand_landmarks:
                    continue
                height, width = frame.shape[:2]
                points = np.array(
                    [[landmark.x * width, landmark.y * height] for landmark in result.multi_hand_landmarks[0].landmark],
                    dtype=np.float32,
                )
                features.append(raw_features(points))
                labels.append(video.label)
                groups.append(video.group)
                detected += 1
        if video_number % 10 == 0 or video_number == len(video_list):
            print(f"extracted {video_number}/{len(video_list)} videos; detected {detected}/{examined} frames", flush=True)
    if not features:
        raise ValueError("MediaPipe detected no hands in the selected ISL-HS videos")
    return np.stack(features), np.array(labels), np.array(groups), {"frames_examined": examined, "frames_detected": detected}


def correlation_columns(train: np.ndarray) -> np.ndarray:
    """Fit a deterministic, unsupervised redundancy filter on a training fold."""
    nonconstant = np.flatnonzero(np.std(train, axis=0) > 1e-12)
    if not len(nonconstant):
        raise ValueError("all raw features are constant")
    correlation = np.abs(np.corrcoef(train[:, nonconstant], rowvar=False))
    kept: list[int] = []
    for index, column in enumerate(nonconstant):
        if not kept or np.all(correlation[index, [np.where(nonconstant == kept_column)[0][0] for kept_column in kept]] < CORRELATION_THRESHOLD):
            kept.append(int(column))
    return np.array(kept)


def evaluate_protocol(features: np.ndarray, labels: np.ndarray, groups: np.ndarray, protocol: str, folds: int) -> dict[str, object]:
    if protocol == "frame_stratified":
        splitter = StratifiedKFold(n_splits=folds, shuffle=False)
        splits = splitter.split(features, labels)
    elif protocol == "video_grouped":
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=False)
        splits = splitter.split(features, labels, groups)
    else:
        raise ValueError(f"unknown protocol: {protocol}")
    results: list[dict[str, object]] = []
    for fold, (train, test) in enumerate(splits, start=1):
        columns = correlation_columns(features[train])
        classifier = RandomForestClassifier(n_estimators=100, n_jobs=8, random_state=SEED + fold)
        classifier.fit(features[train][:, columns], labels[train])
        prediction = classifier.predict(features[test][:, columns])
        results.append(
            {
                "fold": fold,
                "accuracy_percent": float(accuracy_score(labels[test], prediction) * 100),
                "train_frames": int(len(train)),
                "test_frames": int(len(test)),
                "retained_features": int(len(columns)),
            }
        )
    accuracies = [result["accuracy_percent"] for result in results]
    return {
        "protocol": protocol,
        "accuracy_percent_mean": float(np.mean(accuracies)),
        "accuracy_percent_std": float(np.std(accuracies, ddof=1)),
        "folds": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--videos-per-class", type=int, default=18)
    parser.add_argument("--folds", type=int, default=10)
    arguments = parser.parse_args()
    if not 1 <= arguments.videos_per_class <= 18:
        raise ValueError("videos-per-class must be in [1, 18]")
    selected = videos(arguments.data_root, arguments.videos_per_class)
    features, labels, groups, extraction = extract(selected)
    if min(np.bincount(labels, minlength=26)) < arguments.folds:
        raise ValueError("not enough detected frames per class for the requested folds")
    manifest = arguments.data_root / "manifest.json"
    result = {
        "source_manifest_sha256": sha256(manifest),
        "videos": len(selected),
        "raw_feature_count": int(features.shape[1]),
        "extraction": extraction,
        "decoding": "simple-video-utils 0.7.4, RGB display-oriented frames",
        "mediapipe": {"version": mp.__version__, "solution": "Hands", "static_image_mode": False, "max_num_hands": 1, "model_complexity": 1, "min_detection_confidence": 0.5, "min_tracking_confidence": 0.5},
        "feature_reduction": {"method": "train-fold-only greedy absolute Pearson-correlation filter", "threshold": CORRELATION_THRESHOLD, "zero_division": "vertical slopes retain atan(inf); indeterminate/non-finite values become 0"},
        "random_forest": {"n_estimators": 100, "random_state": SEED, "n_jobs": 8, "other_parameters": "scikit-learn defaults"},
        "sklearn_version": sklearn.__version__,
        "evaluations": [
            evaluate_protocol(features, labels, groups, "frame_stratified", arguments.folds),
            evaluate_protocol(features, labels, groups, "video_grouped", arguments.folds),
        ],
    }
    arguments.output_dir.mkdir(parents=True, exist_ok=False)
    (arguments.output_dir / "run.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
