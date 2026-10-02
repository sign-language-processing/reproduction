"""Evaluate a saved Keras model on the Sign Language MNIST test split.

Preprocessing follows the pinned notebook's final "Performance on the Test Set"
cells. The notebook stops at accuracy; the per-class report and confusion matrix
behind the paper's Table 3.1 and Fig. 3.4 are computed here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def report(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    # One-hot inputs are the only form for which scikit-learn prints the
    # "micro avg" and "samples avg" rows shown in Tables 3.1 and 3.2.
    eye = np.eye(n_classes, dtype=int)
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "classification_report": classification_report(eye[y_true], eye[y_pred], output_dict=True, zero_division=0),
        "confusion_matrix_rows_true_cols_predicted": confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import pandas as pd
    import tensorflow as tf
    from sklearn.preprocessing import LabelBinarizer
    from tensorflow import keras

    train_labels = pd.read_csv(args.data_root / "sign_mnist_train.csv", usecols=["label"])["label"]
    label_binarizer = LabelBinarizer().fit(train_labels)
    test_df = pd.read_csv(args.data_root / "sign_mnist_test.csv")
    x_test = test_df.drop("label", axis=1)
    y_true = label_binarizer.transform(test_df["label"]).argmax(axis=1)
    model = keras.models.load_model(args.model_dir)

    n_classes = len(label_binarizer.classes_)
    metrics = {
        "model_dir": str(args.model_dir),
        "test_samples": len(test_df),
        "row_index_to_csv_label": [int(label) for label in label_binarizer.classes_],
    }
    predictions = {"y_true": y_true.tolist()}
    # The notebook evaluates the test split twice: raw 0-255 pixels, then
    # divided by 255 as in training. The second is its final reported protocol.
    for name, scale in (("pixels_divided_by_255", 255.0), ("raw_pixels", 1.0)):
        y_pred = model.predict(tf.reshape(x_test / scale, [-1, 28, 28, 1])).argmax(axis=1)
        metrics[name] = report(y_true, y_pred, n_classes)
        predictions[name] = y_pred.tolist()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "predictions.json").write_text(json.dumps(predictions) + "\n", encoding="utf-8")
    print(json.dumps({name: metrics[name]["accuracy"] for name in ("pixels_divided_by_255", "raw_pixels")}))


if __name__ == "__main__":
    main()
