"""Bounded RBF-SVM reconstruction for the unreported Table III details.

The paper says that the CNN classifier stage is replaced by an RBF SVM, but
does not give a feature layer or SVM hyperparameters.  This uses the final
feature-learning output (SLR global-average-pool / OCNN flatten), and
scikit-learn's documented RBF defaults (C=1, gamma='scale').
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.svm import SVC

from train import AugmentedImages, BATCH_SIZE, SEED, indexed_data, ocnn_model, slr_model


def features(model: tf.keras.Model, data: AugmentedImages) -> tuple[np.ndarray, np.ndarray]:
    values, labels = [], []
    for images, targets in data:
        values.append(model(images, training=False).numpy())
        labels.append(targets.argmax(axis=1))
    return np.concatenate(values), np.concatenate(labels)


def score(name: str, model: tf.keras.Model, train_data: AugmentedImages, validation_data: AugmentedImages) -> dict:
    started = time.monotonic()
    train_x, train_y = features(model, train_data)
    validation_x, validation_y = features(model, validation_data)
    extraction_seconds = time.monotonic() - started
    classifier = SVC(kernel="rbf", C=1.0, gamma="scale", cache_size=2048)
    started = time.monotonic()
    classifier.fit(train_x, train_y)
    fit_seconds = time.monotonic() - started
    return {
        "feature_dimension": int(train_x.shape[1]),
        "training_samples": int(len(train_y)),
        "validation_samples": int(len(validation_y)),
        "training_accuracy": float(classifier.score(train_x, train_y) * 100),
        "validation_accuracy": float(classifier.score(validation_x, validation_y) * 100),
        "feature_extraction_seconds": extraction_seconds,
        "fit_seconds": fit_seconds,
        "system": name,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-per-class", type=int, required=True)
    parser.add_argument("--validation-augmentations", type=int, default=2)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {arguments.output}")

    (train_paths, train_labels), (validation_paths, validation_labels), _ = indexed_data(
        arguments.data_root, arguments.limit_per_class
    )
    # The paper says offline augmentation produced ten instances per training
    # image for SVM.  The preflight uses the same mechanism at a small scale.
    train_data = AugmentedImages(
        train_paths, train_labels, batch_size=BATCH_SIZE, repeats=10, shuffle=False, seed=SEED
    )
    validation_data = AugmentedImages(
        validation_paths,
        validation_labels,
        batch_size=BATCH_SIZE,
        repeats=arguments.validation_augmentations,
        shuffle=False,
        seed=SEED,
    )

    slr = slr_model()
    ocnn = ocnn_model()
    slr.load_weights(arguments.weights_root / "slr.weights.h5")
    ocnn.load_weights(arguments.weights_root / "ocnn.weights.h5")
    # “Feature learning stage” is unreported.  These are the final tensors
    # before each paper-described dense classifier stage.
    slr_features = tf.keras.Model(slr.input, slr.layers[-5].output)
    ocnn_features = tf.keras.Model(ocnn.input, ocnn.layers[-4].output)

    result = {
        "decision": {
            "feature_layers": {"slr": "global_average_pooling", "ocnn": "flatten"},
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "training_augmentation_instances_per_source": 10,
            "validation_augmentation_instances_per_source": arguments.validation_augmentations,
        },
        "results": {
            "slr_svm": score("SLR-SVM", slr_features, train_data, validation_data),
            "ocnn_svm": score("OCNN-SVM", ocnn_features, train_data, validation_data),
        },
    }
    arguments.output.mkdir(parents=True)
    (arguments.output / "run.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
