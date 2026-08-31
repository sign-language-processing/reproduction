"""Train and evaluate the paper's eight ArSL2018 recognition models.

Clean-room implementation: the paper cites no released code. Model/optimizer/
epoch choices are taken from the paper's text (Section III.B and Table II);
every detail the paper omits is a documented reconstruction decision, listed
in README.md, not a paper fact.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# Paper §III.A "Preprocessing": images resized to 64x64, values normalized to
# [0, 1], "retained in RGB format" (the native ArASL2018 images are 64x64
# grayscale, so this means replicating the single channel into 3 channels,
# not an actual resize).
IMAGE_SIZE = 64
CLASS_COUNT = 32
CLASS_NAMES = [
    "ain", "al", "aleff", "bb", "dal", "dha", "dhad", "fa", "gaaf", "ghain",
    "ha", "haa", "jeem", "kaaf", "khaa", "la", "laam", "meem", "nun", "ra",
    "saad", "seen", "sheen", "ta", "taa", "thaa", "thal", "toot", "waw",
    "ya", "yaa", "zay",
]
# Paper §III.A "Data Splitting": 64% train / 16% validation / 20% test,
# stratified. The paper gives no seed; documented decision: seed 42.
SPLIT_SEED = 42
TEST_FRACTION = 0.20
VAL_FRACTION_OF_REMAINDER = 0.20  # 0.16 / (1 - 0.20)


def sha256_of(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the Hugging Face parquet snapshot into (images, labels) arrays."""
    parquet_files = sorted((data_root / "data").glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no parquet files under {data_root / 'data'}")
    frames = [pd.read_parquet(path) for path in parquet_files]
    table = pd.concat(frames, ignore_index=True)
    if len(table) != 54049:
        raise ValueError(f"expected 54049 ArASL2018 images, found {len(table)}")
    # Most rows are native 64x64 grayscale, but 648/54049 (1.2%) are native
    # 256x256 or 768x1024 (10 of those decode as RGB, not grayscale) --
    # exactly what Algorithm 1's resize step is for. Convert to grayscale
    # then resize every image to 64x64, rather than assuming it already is.
    images = np.empty((len(table), IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    non_native_count = 0
    for index, record in enumerate(table["image"]):
        image = Image.open(__import__("io").BytesIO(record["bytes"])).convert("L")
        if image.size != (IMAGE_SIZE, IMAGE_SIZE):
            non_native_count += 1
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        images[index] = np.array(image, dtype=np.uint8)
    if non_native_count != 648:
        raise ValueError(f"expected exactly 648 non-64x64 ArASL2018 rows, found {non_native_count}")
    labels = table["label"].to_numpy(dtype=np.int64)
    if labels.min() < 0 or labels.max() >= CLASS_COUNT:
        raise ValueError("label values fall outside the expected 0..31 range")
    return images, labels


def split_dataset(labels: np.ndarray, seed: int = SPLIT_SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=TEST_FRACTION, stratify=labels, random_state=seed,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=VAL_FRACTION_OF_REMAINDER,
        stratify=labels[train_val_idx],
        random_state=seed,
    )
    return train_idx, val_idx, test_idx


class BrightnessScale(tf.keras.layers.Layer):
    """Multiplicative brightness jitter, matching ImageDataGenerator's
    brightness_range=[0.8, 1.2] semantics (paper Algorithm 1, step 3)."""

    def __init__(self, lower: float = 0.8, upper: float = 1.2, **kwargs):
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper

    def call(self, inputs, training=None):
        if not training:
            return inputs
        batch = tf.shape(inputs)[0]
        factor = tf.random.uniform((batch, 1, 1, 1), self.lower, self.upper)
        return tf.clip_by_value(inputs * factor, 0.0, 1.0)

    def get_config(self):
        return {**super().get_config(), "lower": self.lower, "upper": self.upper}


def build_augmentation() -> tf.keras.Sequential:
    """Paper Algorithm 1, step 3: rotation ±30°, zoom ≤20%, width/height
    shift 20%, brightness [0.8, 1.2]. No horizontal flip: a mirrored
    fingerspelling handshape is a different (or invalid) sign, and the
    canonical Algorithm 1 pseudocode (unlike the illustrative Fig. 8 box)
    does not list one."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(30 / 360.0),
        tf.keras.layers.RandomTranslation(0.2, 0.2),
        tf.keras.layers.RandomZoom(0.2),
        BrightnessScale(0.8, 1.2),
    ], name="augmentation")


def make_dataset(images: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool, augment: bool) -> tf.data.Dataset:
    """Paper Algorithm 1, step 5 (Data Generator Initialization): the
    augmented generator is used for both training and validation; only the
    test generator skips augmentation and applies plain normalization."""
    rgb = np.repeat(images[..., None], 3, axis=-1).astype(np.float32) / 255.0
    dataset = tf.data.Dataset.from_tensor_slices((rgb, labels))
    if shuffle:
        dataset = dataset.shuffle(len(images), seed=SPLIT_SEED, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    if augment:
        augmentation = build_augmentation()
        dataset = dataset.map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.prefetch(tf.data.AUTOTUNE)


@dataclass(frozen=True)
class ModelSpec:
    """One row of Table II. `optimizer`/`epochs` are paper-stated; dropout/
    dense sizes not given by the paper carry a documented default."""

    key: str
    build: "callable"
    optimizer: str
    learning_rate: float | None
    epochs: int
    default_dropout: float
    hho: bool = False
    hho_bounds: dict = field(default_factory=dict)


def resizing_backbone(backbone_factory, resize_to: int | None, head: str, dropout: float, preprocess, unfreeze_last_layers: int = 0) -> tf.keras.Model:
    raw_input = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    # The shared pipeline normalizes to [0, 1] (paper Algorithm 1, applied
    # uniformly ahead of every model). Each tf.keras.applications backbone
    # below was pretrained with its own specific preprocessing that expects
    # raw [0, 255] input (mean-subtracted BGR for ResNet50, x/127.5-1 for
    # DenseNet, an internal Rescaling+Normalization layer for EfficientNet).
    # Feeding [0, 1] straight through skips all of that: DenseNet tolerates
    # it poorly (a working but far-below-paper result), ResNet50 badly
    # (~34% test accuracy after 60 full epochs vs. the paper's 99.16%), and
    # EfficientNet catastrophically (its internal Rescaling(1/255) divides
    # by 255 a second time, collapsing input to ~[0, 0.004] and never
    # escaping chance accuracy). Undoing the [0, 1] scaling here and
    # applying each backbone's own preprocess_input restores the
    # distribution its frozen, pretrained BatchNorm statistics expect.
    x = tf.keras.layers.Rescaling(255.0)(raw_input)
    if resize_to is not None:
        # Paper Fig. 4: EfficientNet-B0/B3 take a 224x224x3 input; the shared
        # preprocessing pipeline only produces 64x64x3, so this upsizes it.
        x = tf.keras.layers.Resizing(resize_to, resize_to)(x)
    if preprocess is not None:
        x = preprocess(x)
    base = backbone_factory(include_top=False, weights="imagenet", input_tensor=x)
    # Decision (second attempt; see README): "early layers frozen... focusing
    # training on remaining layers" is read literally here -- freeze the
    # early part of the backbone and leave its last `unfreeze_last_layers`
    # layers trainable, rather than freezing the whole backbone. The count
    # per model is hardcoded below (make_model_specs) to land close to the
    # from-scratch CNN's ~4.5M trainable parameters, which is otherwise not
    # attainable with the whole backbone frozen (every pretrained model was
    # training 8-18x fewer parameters than the CNN it's supposed to compete
    # against). Layer order/identity comes straight from `base.layers`,
    # which at this point (before the head is attached) is exactly the
    # backbone's own layers in their construction order.
    for layer in base.layers:
        layer.trainable = False
    if unfreeze_last_layers:
        for layer in base.layers[-unfreeze_last_layers:]:
            layer.trainable = True
    x = base.output
    if head == "efficientnet":
        # Paper Fig. 4: GlobalMaxPooling -> Dense(256, relu) -> Dropout(0.6) -> Dense(32, softmax).
        x = tf.keras.layers.GlobalMaxPooling2D()(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    elif head == "resnet50":
        # Paper Fig. 5: Flatten -> Dropout -> Dense(32, softmax). Dropout rate unspecified.
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    elif head == "densenet201":
        # Paper §III.B.4: Flatten -> Dropout(0.8) -> Dense(32, softmax).
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    elif head == "densenet121":
        # Paper §III.B.5 / Fig. 6: MaxPooling -> BatchNorm -> Flatten ->
        # Dense(512, relu) -> Dropout -> Dense(32, ...). The paper states a
        # final sigmoid activation, which is inconsistent with mutually
        # exclusive 32-way classification; softmax is used instead (documented).
        x = tf.keras.layers.MaxPooling2D(pool_size=2, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    else:
        raise ValueError(f"unknown head {head!r}")
    outputs = tf.keras.layers.Dense(CLASS_COUNT, activation="softmax")(x)
    return tf.keras.Model(raw_input, outputs, name=head)


def build_cnn(dropout: float, filter_scale: float = 1.0) -> tf.keras.Model:
    """Paper §III.B.1 / Fig. 2: increasing conv filters (32, 64, 128), ReLU,
    MaxPooling + BatchNormalization, Flatten, Dense(512, relu),
    Dropout(0.5 per text), Dense(32, softmax)."""

    def filters(base: int) -> int:
        return max(4, round(base * filter_scale))

    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = inputs
    for width in (filters(32), filters(64), filters(128)):
        x = tf.keras.layers.Conv2D(width, 3, activation="relu", padding="same")(x)
        x = tf.keras.layers.Conv2D(width, 3, activation="relu", padding="same")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(CLASS_COUNT, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="cnn")


def make_model_specs() -> dict[str, ModelSpec]:
    return {
        # Decision: the paper names no learning rate for the from-scratch CNN.
        # Adam's default (1e-3) was tried first and collapsed to predicting
        # the class prior from epoch 1 onward (loss flat at ln(32), never
        # recovering); 1e-4 (the value CNN-HHO's own search independently
        # converged to as its best candidate) trains normally, so it is used
        # as the plain CNN's documented default too.
        "cnn": ModelSpec("cnn", lambda dropout=0.5, **_: build_cnn(dropout), "adam", 1e-4, 30, 0.5),
        "cnn-hho": ModelSpec(
            "cnn-hho", lambda dropout=0.5, filter_scale=1.0, **_: build_cnn(dropout, filter_scale),
            "adam", None, 5, 0.5, hho=True,
            hho_bounds={
                "log_lr": (-4.0, -2.0),
                "batch_size": (16, 64),
                "filter_scale": (0.5, 1.5),
                "dropout": (0.2, 0.6),
            },
        ),
        # Unfreeze counts below were picked by building each backbone once,
        # counting tf.keras.backend.count_params(model.trainable_weights) as
        # a function of how many of the backbone's trailing layers are
        # unfrozen, and choosing the smallest count that gets within range of
        # the plain CNN's ~4.5M trainable parameters (see README). Not tuned
        # against any accuracy number -- only against a parameter count.
        "efficientnet-b0": ModelSpec(
            "efficientnet-b0",
            # EfficientNet-B0 tops out at ~4.34M trainable params even with
            # every one of its 234 layers unfrozen -- still short of the
            # CNN's 4.5M, so all of it is unfrozen here.
            lambda dropout=0.6, **_: resizing_backbone(tf.keras.applications.EfficientNetB0, 224, "efficientnet", dropout, None, unfreeze_last_layers=234),
            "adamax", 0.001, 70, 0.6,
        ),
        "efficientnet-b3": ModelSpec(
            "efficientnet-b3",
            lambda dropout=0.6, **_: resizing_backbone(tf.keras.applications.EfficientNetB3, 224, "efficientnet", dropout, None, unfreeze_last_layers=35),
            "adamax", 0.001, 84, 0.6,
        ),
        "resnet50": ModelSpec(
            "resnet50",
            lambda dropout=0.5, **_: resizing_backbone(tf.keras.applications.ResNet50, None, "resnet50", dropout, tf.keras.applications.resnet50.preprocess_input, unfreeze_last_layers=10),
            "adamax", None, 60, 0.5,
        ),
        "densenet201": ModelSpec(
            "densenet201",
            lambda dropout=0.8, **_: resizing_backbone(tf.keras.applications.DenseNet201, None, "densenet201", dropout, tf.keras.applications.densenet.preprocess_input, unfreeze_last_layers=125),
            "adamax", None, 150, 0.8,
        ),
        "densenet121": ModelSpec(
            "densenet121",
            lambda dropout=0.5, **_: resizing_backbone(tf.keras.applications.DenseNet121, None, "densenet121", dropout, tf.keras.applications.densenet.preprocess_input, unfreeze_last_layers=200),
            "adam", None, 12, 0.5,
        ),
        "densenet121-hho": ModelSpec(
            "densenet121-hho",
            lambda dropout=0.5, **_: resizing_backbone(tf.keras.applications.DenseNet121, None, "densenet121", dropout, tf.keras.applications.densenet.preprocess_input, unfreeze_last_layers=200),
            "adam", None, 5, 0.5, hho=True,
            hho_bounds={
                "log_lr": (-4.0, -2.0),
                "batch_size": (16, 64),
                "dropout": (0.2, 0.6),
            },
        ),
    }


def build_optimizer(name: str, learning_rate: float | None) -> tf.keras.optimizers.Optimizer:
    kwargs = {} if learning_rate is None else {"learning_rate": learning_rate}
    if name == "adam":
        return tf.keras.optimizers.Adam(**kwargs)
    if name == "adamax":
        return tf.keras.optimizers.Adamax(**kwargs)
    raise ValueError(f"unknown optimizer {name!r}")


def macro_metrics(model: tf.keras.Model, dataset: tf.data.Dataset, labels: np.ndarray) -> dict[str, float]:
    probabilities = model.predict(dataset, verbose=0)
    predictions = probabilities.argmax(axis=1)
    accuracy = float((predictions == labels).mean())
    # Decision: the paper reports single precision/recall/F1 numbers without
    # naming an averaging scheme; macro averaging (the standard default for
    # per-class sklearn/Keras reporting) is used.
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=list(range(CLASS_COUNT)), average="macro", zero_division=0,
    )
    return {"accuracy": accuracy, "precision": float(precision), "recall": float(recall), "f1": float(f1)}


def run_training(
    spec: ModelSpec,
    images: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    dropout: float,
    learning_rate: float | None,
    filter_scale: float = 1.0,
    seed: int = SPLIT_SEED,
) -> dict:
    tf.keras.utils.set_random_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.weights.h5"
    state_path = output_dir / "state.json"

    model = spec.build(dropout=dropout, filter_scale=filter_scale)
    optimizer = build_optimizer(spec.optimizer, learning_rate)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    train_dataset = make_dataset(images[train_idx], labels[train_idx], batch_size, shuffle=True, augment=True)
    val_dataset = make_dataset(images[val_idx], labels[val_idx], batch_size, shuffle=False, augment=True)

    first_epoch = 0
    if state_path.exists() and checkpoint_path.exists():
        model.load_weights(checkpoint_path)
        first_epoch = json.loads(state_path.read_text())["epoch"]
    if first_epoch >= epochs:
        raise ValueError("resume state already meets the requested epoch count")

    metrics_path = output_dir / "metrics.jsonl"
    with metrics_path.open("a" if first_epoch else "w", encoding="utf-8") as metrics_file:
        for epoch in range(first_epoch, epochs):
            start = time.time()
            history = model.fit(train_dataset, validation_data=val_dataset, epochs=1, verbose=0)
            record = {
                "epoch": epoch + 1,
                "loss": float(history.history["loss"][0]),
                "accuracy": float(history.history["accuracy"][0]),
                "val_loss": float(history.history["val_loss"][0]),
                "val_accuracy": float(history.history["val_accuracy"][0]),
                "seconds": time.time() - start,
            }
            metrics_file.write(json.dumps(record) + "\n")
            metrics_file.flush()
            model.save_weights(checkpoint_path)
            state_path.write_text(json.dumps({"epoch": epoch + 1}))
            print(json.dumps(record), flush=True)

    val_metrics = macro_metrics(model, val_dataset, labels[val_idx])
    test_dataset = make_dataset(images[test_idx], labels[test_idx], batch_size, shuffle=False, augment=False)
    test_metrics = macro_metrics(model, test_dataset, labels[test_idx])
    result = {
        "model": spec.key,
        "epochs": epochs,
        "batch_size": batch_size,
        "dropout": dropout,
        "filter_scale": filter_scale,
        "learning_rate": learning_rate,
        "train_count": int(len(train_idx)),
        "val_count": int(len(val_idx)),
        "test_count": int(len(test_idx)),
        "final_train_loss": record["loss"],
        "final_train_accuracy": record["accuracy"],
        "val_loss": record["val_loss"],
        "val_accuracy_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    (output_dir / "run.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def levy_flight(dim: int, beta: float = 1.5) -> np.ndarray:
    """Mantegna's algorithm, as used by Harris Hawks Optimization (Heidari
    et al. 2019, paper ref. [23])."""
    from math import gamma, pi, sin

    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma, dim)
    v = np.random.normal(0, 1, dim)
    return u / (np.abs(v) ** (1 / beta))


def run_hho(
    spec: ModelSpec,
    images: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    output_dir: Path,
    population: int,
    iterations: int,
    proxy_epochs: int,
    proxy_fraction: float,
    seed: int = SPLIT_SEED,
) -> dict:
    """Standard Harris Hawks Optimization (Heidari et al. 2019) over the
    hyperparameters the paper names for this model (§III.B.6-7): learning
    rate, batch size, and dropout (plus filter scale for CNN-HHO).

    Decision: population/iteration counts and the proxy-training fitness
    (fewer epochs on a stratified subsample) are not given by the paper,
    which only says HHO "optimizes...to maximize validation accuracy";
    evaluating full-length training for every candidate would be far more
    expensive than the paper's own reported training cost.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    bound_names = list(spec.hho_bounds.keys())
    lower = np.array([spec.hho_bounds[name][0] for name in bound_names])
    upper = np.array([spec.hho_bounds[name][1] for name in bound_names])
    dim = len(bound_names)

    proxy_train_idx, _ = train_test_split(
        train_idx, train_size=proxy_fraction, stratify=labels[train_idx], random_state=seed,
    )

    def decode(vector: np.ndarray) -> dict:
        values = dict(zip(bound_names, vector))
        decoded = {
            "learning_rate": float(10 ** values["log_lr"]),
            "batch_size": int(round(np.clip(values["batch_size"], *spec.hho_bounds["batch_size"]))),
            "dropout": float(np.clip(values["dropout"], *spec.hho_bounds["dropout"])),
        }
        if "filter_scale" in values:
            decoded["filter_scale"] = float(np.clip(values["filter_scale"], *spec.hho_bounds["filter_scale"]))
        return decoded

    def fitness(vector: np.ndarray) -> float:
        params = decode(vector)
        model = spec.build(dropout=params["dropout"], filter_scale=params.get("filter_scale", 1.0))
        model.compile(
            optimizer=build_optimizer(spec.optimizer, params["learning_rate"]),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        train_dataset = make_dataset(images[proxy_train_idx], labels[proxy_train_idx], params["batch_size"], shuffle=True, augment=True)
        val_dataset = make_dataset(images[val_idx], labels[val_idx], params["batch_size"], shuffle=False, augment=True)
        history = model.fit(train_dataset, validation_data=val_dataset, epochs=proxy_epochs, verbose=0)
        return float(history.history["val_accuracy"][-1])

    positions = rng.uniform(lower, upper, size=(population, dim))
    fitness_values = np.array([fitness(position) for position in positions])
    trace = [{"iteration": 0, "best_fitness": float(fitness_values.max()), "best_params": decode(positions[fitness_values.argmax()])}]

    for iteration in range(1, iterations + 1):
        energy_factor = 2 * (1 - iteration / iterations)
        best_index = fitness_values.argmax()
        best_position = positions[best_index]
        mean_position = positions.mean(axis=0)
        for i in range(population):
            jump_energy = 2 * (1 - rng.random())
            escaping_energy = energy_factor * (2 * rng.random() - 1)
            if abs(escaping_energy) >= 1:
                if rng.random() >= 0.5:
                    random_hawk = positions[rng.integers(population)]
                    candidate = random_hawk - rng.random() * np.abs(random_hawk - 2 * rng.random() * positions[i])
                else:
                    candidate = (best_position - mean_position) - rng.random() * (lower + rng.random() * (upper - lower))
            else:
                if rng.random() >= 0.5:
                    candidate = best_position - escaping_energy * np.abs(jump_energy * best_position - positions[i])
                else:
                    soft = best_position - escaping_energy * np.abs(jump_energy * best_position - positions[i])
                    dive = soft + rng.random(dim) * levy_flight(dim)
                    candidate = dive if fitness(np.clip(dive, lower, upper)) > fitness(np.clip(soft, lower, upper)) else soft
            candidate = np.clip(candidate, lower, upper)
            candidate_fitness = fitness(candidate)
            if candidate_fitness > fitness_values[i]:
                positions[i] = candidate
                fitness_values[i] = candidate_fitness
        best_index = fitness_values.argmax()
        trace.append({"iteration": iteration, "best_fitness": float(fitness_values[best_index]), "best_params": decode(positions[best_index])})
        print(json.dumps(trace[-1]), flush=True)

    best_index = fitness_values.argmax()
    best_params = decode(positions[best_index])
    search_path = output_dir / "hho_search.json"
    search_path.write_text(json.dumps({
        "population": population,
        "iterations": iterations,
        "proxy_epochs": proxy_epochs,
        "proxy_fraction": proxy_fraction,
        "proxy_train_count": int(len(proxy_train_idx)),
        "trace": trace,
        "best_params": best_params,
    }, indent=2) + "\n", encoding="utf-8")
    return best_params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(make_model_specs().keys()))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--preflight", action="store_true", help="tiny end-to-end smoke test, not a retained run")
    parser.add_argument("--hho-population", type=int, default=6)
    parser.add_argument("--hho-iterations", type=int, default=4)
    parser.add_argument("--hho-proxy-epochs", type=int, default=3)
    parser.add_argument("--hho-proxy-fraction", type=float, default=0.25)
    arguments = parser.parse_args()

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    spec = make_model_specs()[arguments.model]

    images, labels = load_dataset(arguments.data_root)
    train_idx, val_idx, test_idx = split_dataset(labels, arguments.seed)

    if arguments.preflight:
        rng = np.random.default_rng(arguments.seed)
        train_idx = rng.choice(train_idx, size=min(256, len(train_idx)), replace=False)
        val_idx = rng.choice(val_idx, size=min(64, len(val_idx)), replace=False)
        test_idx = rng.choice(test_idx, size=min(64, len(test_idx)), replace=False)
        epochs = 2
    else:
        epochs = arguments.epochs or spec.epochs

    if spec.hho and not arguments.preflight:
        best_params = run_hho(
            spec, images, labels, train_idx, val_idx, arguments.output_dir,
            arguments.hho_population, arguments.hho_iterations,
            arguments.hho_proxy_epochs, arguments.hho_proxy_fraction, arguments.seed,
        )
    elif spec.hho:
        best_params = {"learning_rate": 1e-3, "batch_size": arguments.batch_size, "dropout": spec.default_dropout, "filter_scale": 1.0}
    else:
        best_params = {"learning_rate": spec.learning_rate, "batch_size": arguments.batch_size, "dropout": spec.default_dropout, "filter_scale": 1.0}

    result = run_training(
        spec, images, labels, train_idx, val_idx, test_idx, arguments.output_dir,
        epochs=epochs,
        batch_size=best_params["batch_size"],
        dropout=best_params["dropout"],
        learning_rate=best_params["learning_rate"],
        filter_scale=best_params.get("filter_scale", 1.0),
        seed=arguments.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
