"""Conditional CNN + Bi-LSTM reconstruction for Sangeetha & Divya Gowri, ICNGCS 2025 (Section IV results).

The paper released no code. Keras is used because the paper's Figure 2 shows Keras
output ("Found 3903 images belonging to 13 classes"); it runs on the PyTorch backend
of the study image. Every value below that is not cited to the paper is a
reconstruction decision documented in README.md.
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import keras
import numpy as np
from keras import layers
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

CLASSES = [str(i) for i in range(1, 14)]  # TLFS23 folders 1-13 = the 13 classes of Figure 2
SIZE = 224  # Sec. IV.B: 224x224 images
GAMMA = 1.5  # Sec. IV.D: "generally in the range of 0.5 to 2.0"
LUT = np.array([255 * (i / 255) ** (1 / GAMMA) for i in range(256)], dtype=np.uint8)


def preprocess(path: Path) -> np.ndarray:
    """Sec. IV.D in the paper's order: HSV skin mask, Gaussian filter, gamma, grayscale."""
    image = cv2.imread(str(path))
    mask = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), (0, 20, 70), (20, 255, 255))
    image = cv2.bitwise_and(image, image, mask=mask)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.LUT(image, LUT)
    return cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (SIZE, SIZE))


def load(root: Path, per_class: int, test_fraction: float, seed: int):
    """Sample `per_class` images per class, then split each class into train/test."""
    rng = np.random.default_rng(seed)
    split = {"train": ([], [], []), "test": ([], [], [])}
    n_test = round(per_class * test_fraction)
    for label, name in enumerate(CLASSES):
        files = sorted((root / name).glob("*.jpg"))
        chosen = [files[i] for i in rng.choice(len(files), per_class, replace=False)]
        for part, paths in (("test", chosen[:n_test]), ("train", chosen[n_test:])):
            for path in paths:
                split[part][0].append(preprocess(path))
                split[part][1].append(label)
                split[part][2].append(f"{name}/{path.name}")
    return {part: (np.stack(x)[..., None], np.array(y), names) for part, (x, y, names) in split.items()}


def build_model(lr: float) -> keras.Model:
    model = keras.Sequential([
        keras.Input((SIZE, SIZE, 1)),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.Rescaling(1 / 255),
        # Sec. IV.E: three 3x3 convolutional layers.
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),
        # Sec. IV.F: Bi-LSTM over the conv features; rows of the feature map are the timesteps.
        layers.Reshape((26, 26 * 128)),
        layers.Bidirectional(layers.LSTM(128)),
        # Sec. IV.F.ii: dense 256, 128, 64 with ReLU, each followed by dropout.
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(len(CLASSES), activation="softmax"),
    ])
    # Sec. I: Adam and categorical cross-entropy; Sec. IV.G: the learning rate is swept.
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--per-class", type=int, default=300, help="300 x 13 is closest to Figure 2's 3,903 images")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)

    run_file = args.output_dir / "run.json"
    if run_file.exists():
        return json.loads(run_file.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    keras.utils.set_random_seed(args.seed)
    started = time.time()
    data = load(args.data_root, args.per_class, args.test_fraction, args.seed)
    (x_train, y_train, _), (x_test, y_test, test_names) = data["train"], data["test"]
    print(f"loaded train={len(x_train)} test={len(x_test)} in {time.time() - started:.0f}s", flush=True)

    model = build_model(args.lr)
    history = model.fit(
        x_train, keras.utils.to_categorical(y_train, len(CLASSES)),
        batch_size=args.batch_size, epochs=args.epochs, shuffle=True, verbose=2,
    )
    model_path = args.output_dir / "model.keras"
    model.save(model_path)
    reloaded = keras.models.load_model(model_path)  # evaluate the saved model, not the in-memory one
    predictions = reloaded.predict(x_test, batch_size=64, verbose=0).argmax(1)

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average="weighted", zero_division=0)
    run = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "train_count": len(x_train),
        "test_count": len(x_test),
        "accuracy": 100 * accuracy_score(y_test, predictions),
        "precision_weighted": 100 * precision,
        "recall_weighted": 100 * recall,
        "f1_weighted": 100 * f1,
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "predictions": dict(zip(test_names, predictions.tolist())),
        "history": {key: [float(v) for v in values] for key, values in history.history.items()},
        "seconds": time.time() - started,
        "keras": keras.__version__,
        "backend": keras.backend.backend(),
        "train_py_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dataset_manifest_sha256": hashlib.sha256((args.data_root / "MANIFEST.sha256").read_bytes()).hexdigest(),
    }
    (args.output_dir / "freeze.txt").write_text(subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout)
    run_file.write_text(json.dumps(run, indent=2) + "\n")
    return run


if __name__ == "__main__":
    main()
