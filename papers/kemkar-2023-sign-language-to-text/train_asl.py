"""Conditional experiment for Table 3.2 / Fig. 3.5 (Kaggle ASL dataset, 36 classes).

No code for this dataset was published and the paper gives no split, input
size, or training schedule. Everything not in the paper is carried over from the
pinned notebook's Sign Language MNIST pipeline and is recorded as a guess in
reproduction.json. The evaluation set is every file in the archive (5030),
because that is the support reported in Table 3.2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate import report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from PIL import Image
    from tensorflow import keras

    keras.utils.set_random_seed(args.seed)

    paths = sorted(args.data_root.rglob("*.jpeg"))
    classes = sorted({path.parent.name for path in paths})
    x = np.stack([np.asarray(Image.open(path).convert("L").resize((28, 28)), dtype=np.float64) for path in paths]) / 255.0
    x = x.reshape(-1, 28, 28, 1)
    y = np.array([classes.index(path.parent.name) for path in paths])

    # Same proportions as the notebook's 19500 / 7955 train / validation split.
    order = np.random.RandomState(args.seed).permutation(len(paths))
    n_train = round(len(paths) * 19500 / 27455)
    train, valid = order[:n_train], order[n_train:]
    one_hot = np.eye(len(classes))[y]

    # Fig. 3.2 architecture with the output layer widened from 24 to 36 classes.
    model = keras.models.Sequential()
    for filters in (24, 48, 96):
        model.add(keras.layers.Conv2D(filters, (5, 5), padding="same", activation="relu"))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(len(classes), activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "model"
    history = model.fit(
        x[train],
        one_hot[train],
        epochs=10,
        validation_data=(x[valid], one_hot[valid]),
        callbacks=[
            keras.callbacks.ModelCheckpoint(str(model_dir), save_best_only=True),
            keras.callbacks.EarlyStopping(patience=5),
        ],
        verbose=2,
    )
    y_pred = keras.models.load_model(model_dir).predict(x).argmax(axis=1)

    metrics = {
        "seed": args.seed,
        "classes": classes,
        "samples": {"all_files": len(paths), "train": len(train), "validation": len(valid)},
        "history": history.history,
        "all_files": report(y, y_pred, len(classes)),
        "validation_only": {"accuracy": float((y[valid] == y_pred[valid]).mean())},
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "predictions.json").write_text(
        json.dumps({"paths": [str(path.relative_to(args.data_root)) for path in paths], "y_true": y.tolist(), "y_pred": y_pred.tolist(), "validation_indices": valid.tolist()}) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"all_files_accuracy": metrics["all_files"]["accuracy"], "validation_accuracy": metrics["validation_only"]["accuracy"]}))


if __name__ == "__main__":
    main()
