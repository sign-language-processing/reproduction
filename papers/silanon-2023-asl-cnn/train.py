"""Conditional reconstruction of Silanon and Lertchuwongsa (2023), Table III.

The paper gives the dataset split, augmentation ranges, CNN layer shapes, and
ensemble rule, but omits implementation details such as the random seed and
optimizer.  Those choices are constants below and are emitted with every run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tensorflow as tf


SEED = 2026
IMAGE_SIZE = (64, 64)  # SLRNet-8's published input resolution.
CLASS_COUNT = 29
EPOCHS = 10  # Figures 5 and 6 plot exactly ten training epochs.
BATCH_SIZE = 128
VALIDATION_AUGMENTATIONS = 10
L2 = 0.02


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_generator() -> tf.keras.preprocessing.image.ImageDataGenerator:
    """Table I plus the paper's per-image [0, 1], mean-0/std-1 normalization."""
    return tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        horizontal_flip=True,
        brightness_range=(0.2, 1.2),
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1.0 / 255.0,
        samplewise_center=True,
        samplewise_std_normalization=True,
    )


class AugmentedImages(tf.keras.utils.Sequence):
    """A deterministic online or fixed tenfold-offline ImageDataGenerator path."""

    def __init__(self, paths, labels, *, batch_size, repeats, shuffle, seed):
        self.paths = list(paths)
        self.labels = np.asarray(labels, dtype=np.int32)
        self.batch_size = batch_size
        self.repeats = repeats
        self.shuffle = shuffle
        self.seed = seed
        self.generator = image_generator()
        # VolumeFS v2 has high per-file latency.  The paper's per-image
        # transform and sample order remain unchanged; threads merely overlap
        # independent JPEG reads.  A real 928-image measurement was 59.03 s
        # serial and 1.15 s with eight readers.
        self.pool = ThreadPoolExecutor(max_workers=8)
        self.random_lock = threading.Lock()
        self.epoch = 0
        self.order = np.arange(len(self.paths) * repeats)
        self.on_epoch_end()

    def __len__(self):
        return (len(self.order) + self.batch_size - 1) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(self.order)
        self.epoch += 1

    def __getitem__(self, batch):
        selected = self.order[batch * self.batch_size : (batch + 1) * self.batch_size]
        images = np.empty((len(selected), *IMAGE_SIZE, 3), dtype=np.float32)
        targets = np.zeros((len(selected), CLASS_COUNT), dtype=np.float32)
        def load(position_and_flattened):
            position, flattened = position_and_flattened
            source_index = int(flattened % len(self.paths))
            repeat = int(flattened // len(self.paths))
            image = tf.keras.utils.img_to_array(
                tf.keras.utils.load_img(self.paths[source_index], target_size=IMAGE_SIZE)
            )
            # ImageDataGenerator's seed gives an independent, reproducible
            # transform for every source image/augmentation/epoch combination.
            transform_seed = self.seed + source_index * 1009 + repeat * 9176
            if self.shuffle:
                transform_seed += (self.epoch - 1) * 104729
            # ImageDataGenerator seeds NumPy's process-global RNG while it
            # samples parameters.  Keep that tiny operation serialized so the
            # I/O threads cannot perturb one another's documented transform.
            with self.random_lock:
                transform = self.generator.get_random_transform(image.shape, seed=transform_seed)
            return position, self.generator.standardize(self.generator.apply_transform(image, transform)), self.labels[source_index]

        for position, image, label in self.pool.map(load, enumerate(selected)):
            images[position] = image
            targets[position, label] = 1.0
        return images, targets


def indexed_data(data_root: Path, limit_per_class: int | None):
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    classes = manifest["stored_classes"]
    if len(classes) != CLASS_COUNT:
        raise ValueError(f"expected {CLASS_COUNT} stored classes, got {len(classes)}")
    root = data_root / manifest["training_root"]
    paths, labels = [], []
    for label, name in enumerate(classes):
        samples = sorted(path for path in (root / name).iterdir() if path.is_file())
        if len(samples) != 3000:
            raise ValueError(f"expected 3,000 images for {name}, got {len(samples)}")
        if limit_per_class is not None:
            samples = samples[:limit_per_class]
        paths.extend(str(path) for path in samples)
        labels.extend([label] * len(samples))
    rng = np.random.default_rng(SEED)
    paths, labels = np.asarray(paths), np.asarray(labels)
    train, validation = [], []
    for label in range(CLASS_COUNT):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        split = int(len(indices) * 0.8)
        train.extend(indices[:split])
        validation.extend(indices[split:])
    return (paths[train], labels[train]), (paths[validation], labels[validation]), manifest


def slr_model() -> tf.keras.Model:
    """SLRNet-8 recreated from Figure 4 and its original 64x64 specification."""
    inputs = tf.keras.Input((*IMAGE_SIZE, 3))
    x = inputs
    for filters, kernel, pool in ((32, 5, False), (64, 5, False), (128, 3, True), (256, 3, True), (384, 3, True), (512, 3, False)):
        x = tf.keras.layers.Conv2D(filters, kernel, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        if pool:
            x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(84)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    return tf.keras.Model(inputs, tf.keras.layers.Dense(CLASS_COUNT, activation="softmax")(x), name="slrnet8")


def ocnn_model() -> tf.keras.Model:
    """OCNN from Section III.B and Table II; undocumented padding is explicit."""
    inputs = tf.keras.Input((*IMAGE_SIZE, 3))
    x = tf.keras.layers.Conv2D(64, 5, strides=2, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(64, 5, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, 4, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(256, 4, strides=1, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    outputs = tf.keras.layers.Dense(CLASS_COUNT, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(L2))(x)
    return tf.keras.Model(inputs, outputs, name="ocnn")


def train(model, train_data, validation_data, checkpoint: Path):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    callback = tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_accuracy", mode="max", save_best_only=True, save_weights_only=True)
    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=EPOCHS,
        callbacks=[callback],
        verbose=2,
    )
    model.load_weights(checkpoint)
    values = {key: [float(value) for value in series] for key, series in history.history.items()}
    selected_epoch = int(np.argmax(values["val_accuracy"]))
    # Table III reports the training and validation accuracies associated with
    # the checkpoint with maximum validation accuracy, rather than independent
    # maxima.  Retain that pairing for a like-for-like comparison.
    return {
        "history": values,
        "selected_epoch": selected_epoch + 1,
        "selected_training_accuracy": values["accuracy"][selected_epoch] * 100,
        "selected_validation_accuracy": values["val_accuracy"][selected_epoch] * 100,
    }


def accuracy(model, data):
    correct = total = 0
    for images, labels in data:
        predictions = model(images, training=False).numpy().argmax(axis=1)
        correct += int((predictions == labels.argmax(axis=1)).sum())
        total += len(predictions)
    return correct / total * 100


def ensemble_accuracy(slr, ocnn, data):
    correct = total = 0
    for images, labels in data:
        probabilities = (slr(images, training=False).numpy() + ocnn(images, training=False).numpy()) / 2
        correct += int((probabilities.argmax(axis=1) == labels.argmax(axis=1)).sum())
        total += len(labels)
    return correct / total * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-per-class", type=int)
    parser.add_argument("--validation-augmentations", type=int, default=VALIDATION_AUGMENTATIONS)
    arguments = parser.parse_args()
    if arguments.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {arguments.output_dir}")
    if arguments.limit_per_class is not None and arguments.limit_per_class < 5:
        raise ValueError("limit-per-class must allow a nonempty 80:20 split")
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    (train_paths, train_labels), (validation_paths, validation_labels), manifest = indexed_data(arguments.data_root, arguments.limit_per_class)
    train_data = AugmentedImages(train_paths, train_labels, batch_size=BATCH_SIZE, repeats=1, shuffle=True, seed=SEED)
    validation_data = AugmentedImages(validation_paths, validation_labels, batch_size=BATCH_SIZE, repeats=arguments.validation_augmentations, shuffle=False, seed=SEED)
    arguments.output_dir.mkdir(parents=True)
    slr = slr_model()
    ocnn = ocnn_model()
    slr_history = train(slr, train_data, validation_data, arguments.output_dir / "slr.weights.h5")
    ocnn_history = train(ocnn, train_data, validation_data, arguments.output_dir / "ocnn.weights.h5")
    result = {
        "paper": "Silanon and Lertchuwongsa (2023), Table III",
        "source_manifest_sha256": sha256(arguments.data_root / "manifest.json"),
        "samples": {"training": len(train_paths), "validation_original": len(validation_paths), "validation_augmented": len(validation_paths) * arguments.validation_augmentations},
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "tensorflow": tf.__version__,
        "augmentation": {"rotation_range": 10, "horizontal_flip": True, "brightness_range": [0.2, 1.2], "width_shift_range": 0.2, "height_shift_range": 0.2, "rescale": 1.0 / 255.0, "samplewise_center": True, "samplewise_std_normalization": True},
        "results": {"slr_validation_accuracy": accuracy(slr, validation_data), "ocnn_validation_accuracy": accuracy(ocnn, validation_data), "ensemble_validation_accuracy": ensemble_accuracy(slr, ocnn, validation_data)},
        "checkpoints": {"slr": slr_history, "ocnn": ocnn_history},
        "unproduced": {"slr_svm": "The paper gives only RBF kernel and an offline 696,000-image training set; its C/gamma/feature-layer choices are not reported.", "ocnn_svm": "The paper gives only RBF kernel and an offline 696,000-image training set; its C/gamma/feature-layer choices are not reported."},
    }
    (arguments.output_dir / "run.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["results"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
