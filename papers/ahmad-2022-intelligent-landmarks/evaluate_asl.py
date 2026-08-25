"""Conditional ASL Alphabet evaluation using the paper's landmark features."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from evaluate import CORRELATION_THRESHOLD, SEED, correlation_columns, raw_features


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract(data_root: Path, images_per_class: int) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[str, object]]:
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    classes = manifest["training_classes"]
    training_root = data_root / manifest["training_root"]
    features: list[np.ndarray] = []
    labels: list[int] = []
    detected = 0
    examined = 0
    # The paper supplies still images to MediaPipe. static_image_mode=True
    # therefore avoids carrying temporal tracking between independent images.
    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        for label, name in enumerate(classes):
            paths = sorted(path for path in (training_root / name).iterdir() if path.is_file())
            if len(paths) != 3000:
                raise ValueError(f"expected 3,000 images for {name}, found {len(paths)}")
            for path in paths[:images_per_class]:
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f"OpenCV could not read {path}")
                examined += 1
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result = hands.process(image)
                if not result.multi_hand_landmarks:
                    continue
                height, width = image.shape[:2]
                points = np.array(
                    [[landmark.x * width, landmark.y * height] for landmark in result.multi_hand_landmarks[0].landmark],
                    dtype=np.float32,
                )
                features.append(raw_features(points))
                labels.append(label)
                detected += 1
            print(f"extracted {label + 1}/{len(classes)} classes; detected {detected}/{examined} images", flush=True)
    if not features:
        raise ValueError("MediaPipe detected no hands in the selected ASL images")
    return np.stack(features), np.array(labels), {"images_examined": examined, "images_detected": detected}, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--images-per-class", type=int, default=3000)
    parser.add_argument("--folds", type=int, default=10)
    arguments = parser.parse_args()
    if not 1 <= arguments.images_per_class <= 3000:
        raise ValueError("images-per-class must be in [1, 3000]")
    features, labels, extraction, source = extract(arguments.data_root, arguments.images_per_class)
    if min(np.bincount(labels, minlength=28)) < arguments.folds:
        raise ValueError("not enough detected images per class for the requested folds")
    splitter = StratifiedKFold(n_splits=arguments.folds, shuffle=True, random_state=SEED)
    folds = []
    for fold, (train, test) in enumerate(splitter.split(features, labels), start=1):
        columns = correlation_columns(features[train])
        classifier = RandomForestClassifier(n_estimators=100, n_jobs=8, random_state=SEED + fold)
        classifier.fit(features[train][:, columns], labels[train])
        prediction = classifier.predict(features[test][:, columns])
        folds.append({"fold": fold, "accuracy_percent": float(accuracy_score(labels[test], prediction) * 100), "train_images": int(len(train)), "test_images": int(len(test)), "retained_features": int(len(columns))})
    accuracies = [fold["accuracy_percent"] for fold in folds]
    result = {
        "source_manifest_sha256": sha256(arguments.data_root / "manifest.json"),
        "stored_training_classes": source["stored_classes"],
        "training_classes": source["training_classes"],
        "excluded_classes": source["excluded_classes"],
        "raw_feature_count": int(features.shape[1]),
        "extraction": extraction,
        "mediapipe": {"version": mp.__version__, "solution": "Hands", "static_image_mode": True, "max_num_hands": 1, "model_complexity": 1, "min_detection_confidence": 0.5, "min_tracking_confidence": 0.5},
        "feature_reduction": {"method": "train-fold-only greedy absolute Pearson-correlation filter", "threshold": CORRELATION_THRESHOLD, "zero_division": "vertical slopes retain atan(inf); indeterminate/non-finite values become 0"},
        "random_forest": {"n_estimators": 100, "random_state": SEED, "n_jobs": 8, "other_parameters": "scikit-learn defaults"},
        "sklearn_version": sklearn.__version__,
        "evaluation": {"protocol": "conditional shuffled 10-fold stratified image CV", "accuracy_percent_mean": float(np.mean(accuracies)), "accuracy_percent_std": float(np.std(accuracies, ddof=1)), "folds": folds},
    }
    arguments.output_dir.mkdir(parents=True, exist_ok=False)
    (arguments.output_dir / "run.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
