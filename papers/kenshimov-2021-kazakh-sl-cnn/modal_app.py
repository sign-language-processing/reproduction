"""Modal jobs for Kenshimov et al. 2021, Table 1 (Kazakh fingerspelling CNNs).

populate: extract the authors' archive into the shared datasets Volume, once.
train:    train one model for one or more seeds and score it on the test split.

Protocol sources are cited inline; every choice the paper leaves open is listed
in reproduction.json.guesses_and_deviations.
"""

import os

import modal

PAPER = "f912577b"
SLUG = "ksl-fingerspelling-kenshimov-2021"
ARCHIVE_SHA256 = "baf1f1436f2d827b03d88c6b42db783d4e9ca0b84b7bb82fc8ee2f133e494ec2"
DATA_ROOT = f"/datasets/{SLUG}"
IMAGES = f"{DATA_ROOT}/resized_gray_dataset"

IMG_H, IMG_W = 120, 320  # Read_data.ipynb: img.resize((320, 120))
PER_CLASS = 50  # paper Sec. 5.1: 50 images for each of 42 classes
EPOCHS = 20  # paper Sec. 5.2
BATCH_SIZE = 64  # Compare_nets.ipynb log of the original runs: "batch 64"
N_CLASSES = 42
# GPU type, chosen at launch. EfficientNetB7 at batch 64 needs more than a 15 GB T4.
GPU = os.environ.get("KSL_GPU", "T4")

app = modal.App(f"ksl-{PAPER}")
datasets = modal.Volume.from_name("datasets", version=2)
cache = modal.Volume.from_name("huggingface-cache", version=2)
results = modal.Volume.from_name(f"ksl-{PAPER}-results", create_if_missing=True, version=2)

CACHE_ENV = {"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub"}
populate_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("py7zr==0.22.0", "pillow==10.4.0")
    .env(CACHE_ENV)
)
train_image = (
    modal.Image.from_registry(
        # nvidia/cuda 11.2.2-cudnn8-runtime-ubuntu20.04, the CUDA/cuDNN pair of TF 2.8
        "nvidia/cuda@sha256:081a31f56decc6c460df1808ed1c4867fb30b0fbfa8929258b10e3e5d6dc1a2e",
        add_python="3.10",
    )
    .pip_install(
        "tensorflow==2.8.0",
        "keras==2.8.0",
        "numpy==1.23.5",
        "protobuf==3.20.3",
        "scikit-learn==1.3.2",
        "pillow==10.4.0",
        "h5py==3.10.0",
    )
    .env(
        {
            **CACHE_ENV,
            "PYTHONUNBUFFERED": "1",
            "TF_CPP_MIN_LOG_LEVEL": "1",
        }
    )
)


def sha256_file(path):
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@app.function(
    image=populate_image,
    cpu=4,
    memory=8192,
    timeout=3600,
    volumes={"/datasets": datasets, "/cache/huggingface": cache},
)
def populate():
    """Extract the uploaded archive and write a per-file manifest. Idempotent."""
    import json
    import os

    import py7zr
    from PIL import Image

    manifest_path = f"{DATA_ROOT}/manifest.json"
    if os.path.exists(manifest_path):
        print(json.dumps({"already_populated": json.load(open(manifest_path))["summary"]}))
        return
    archive = f"{DATA_ROOT}/source/resized_gray_dataset.7z"
    actual = sha256_file(archive)
    if actual != ARCHIVE_SHA256:
        raise RuntimeError(f"archive sha256 {actual} != {ARCHIVE_SHA256}")
    with py7zr.SevenZipFile(archive) as handle:
        handle.extractall(DATA_ROOT)

    files, per_class, unreadable = {}, {}, []
    for name in sorted(os.listdir(IMAGES)):
        for file in sorted(os.listdir(f"{IMAGES}/{name}")):
            path = f"{IMAGES}/{name}/{file}"
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception:
                unreadable.append(f"{name}/{file}")
            files[f"{name}/{file}"] = sha256_file(path)
            per_class[name] = per_class.get(name, 0) + 1
    summary = {
        "archive_sha256": actual,
        "classes": len(per_class),
        "images": len(files),
        "per_class": per_class,
        "unreadable": unreadable,
    }
    with open(f"{DATA_ROOT}/MANIFEST.sha256", "w") as handle:
        handle.writelines(f"{digest}  {path}\n" for path, digest in files.items())
    summary["manifest_sha256"] = sha256_file(f"{DATA_ROOT}/MANIFEST.sha256")
    with open(manifest_path, "w") as handle:
        json.dump({"summary": summary}, handle, indent=1)
    datasets.commit()
    print(json.dumps({"populated": summary}))


def load_subset(per_class):
    """First `per_class` files of each class folder, in lexicographic name order."""
    import os

    import numpy as np
    from PIL import Image

    class_names = sorted(os.listdir(IMAGES))
    assert len(class_names) == N_CLASSES, class_names
    images, labels, names = [], [], []
    for label, name in enumerate(class_names):
        chosen = sorted(os.listdir(f"{IMAGES}/{name}"))[:per_class]
        assert len(chosen) == per_class, (name, len(chosen))
        for file in chosen:
            # Read_data.ipynb: Image.open(...).convert('L'); img.resize((320, 120))
            image = Image.open(f"{IMAGES}/{name}/{file}").convert("L").resize((IMG_W, IMG_H))
            images.append(np.array(image))
            labels.append(label)
            names.append(f"{name}/{file}")
    # Read_data.ipynb: float32, reshape to (N, 120, 320, 1), x_data /= 255
    x = np.array(images, dtype="float32").reshape((-1, IMG_H, IMG_W, 1)) / 255
    return x, np.array(labels), np.array(names), class_names


def build(model):
    import tensorflow as tf
    from tensorflow.keras import layers

    shape = (IMG_H, IMG_W, 1)
    if model == "lenet":
        # The authors' builder from Compare_nets.ipynb, without its later
        # augmentation layers. 1,071,162 parameters = Table 1's 1.07 M.
        return tf.keras.Sequential(
            [
                layers.Input(shape),
                layers.Conv2D(6, 5, activation="tanh"),
                layers.AveragePooling2D(pool_size=3, strides=2),
                layers.Conv2D(16, 5, padding="same", activation="tanh"),
                layers.AveragePooling2D(pool_size=2, strides=2),
                layers.Conv2D(16, 5, padding="same", activation="tanh"),
                layers.MaxPool2D(pool_size=2, strides=2),
                layers.Flatten(),
                layers.Dense(120, activation="tanh"),
                layers.Dense(84, activation="tanh"),
                layers.Dense(N_CLASSES, activation="softmax"),
            ],
            name="lenet",
        )
    if model == "alexnet":
        # Not in either notebook. The widely copied Keras AlexNet with 1x1
        # filters in the last two convolutions (Compare_nets.ipynb: "The 2022
        # version used 1x1 filters there") has 35,498,090 parameters on this
        # input = Table 1's 35.5 M.
        return tf.keras.Sequential(
            [
                layers.Input(shape),
                layers.Conv2D(96, 11, strides=4, activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPool2D(pool_size=3, strides=2),
                layers.Conv2D(256, 5, padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPool2D(pool_size=3, strides=2),
                layers.Conv2D(384, 3, padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.Conv2D(384, 1, padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.Conv2D(256, 1, padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPool2D(pool_size=3, strides=2),
                layers.Flatten(),
                layers.Dense(4096, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(4096, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(N_CLASSES, activation="softmax"),
            ],
            name="alexnet",
        )
    if model == "resnet50":
        return tf.keras.applications.ResNet50(weights=None, input_shape=shape, classes=N_CLASSES)
    if model == "efficientnetb7":
        return tf.keras.applications.EfficientNetB7(
            weights=None, input_shape=shape, classes=N_CLASSES
        )
    raise ValueError(model)


@app.function(
    image=train_image,
    gpu=GPU,
    cpu=4,
    memory=16384,
    timeout=7200,
    retries=0,
    volumes={
        "/datasets": datasets.read_only(),
        "/cache/huggingface": cache,
        "/results": results,
    },
)
def train(
    models: str,
    run_id: str,
    seeds: str = "42",
    epochs: int = EPOCHS,
    per_class: int = PER_CLASS,
    pixel_scale: str = "unit",
):
    """Train each model once per seed; write /results/<run_id>/<model>-seed<seed>/.

    pixel_scale "unit" is the protocol (Read_data.ipynb divides by 255). "raw"
    feeds 0-255 pixels and exists only for a diagnostic run.
    """
    import json
    import os
    import subprocess
    import time

    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split

    x, labels, names, class_names = load_subset(per_class)
    assert pixel_scale in ("unit", "raw"), pixel_scale
    if pixel_scale == "raw":
        x = x * 255
    y = tf.keras.utils.to_categorical(labels, N_CLASSES)
    indices = np.arange(len(x))
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    os.makedirs(f"/results/{run_id}", exist_ok=True)
    freeze = subprocess.run(["python", "-m", "pip", "freeze"], capture_output=True, text=True)
    with open(f"/results/{run_id}/pip-freeze.txt", "w") as handle:
        handle.write(freeze.stdout)

    jobs = [(m, int(v)) for m in models.split(",") for v in seeds.split(",")]
    for model, seed in jobs:
        out = f"/results/{run_id}/{model}-seed{seed}"
        os.makedirs(out, exist_ok=True)
        tf.config.experimental.reset_memory_stats("GPU:0")
        # Read_data.ipynb: two unstratified train_test_split calls, 0.2 then 0.5.
        # The original calls were unseeded; the seed is this attempt's choice.
        train_idx, rest_idx = train_test_split(indices, test_size=0.2, random_state=seed)
        val_idx, test_idx = train_test_split(rest_idx, test_size=0.5, random_state=seed)

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        net = build(model)
        # Paper Sec. 5.2: rmsprop optimizer, categorical cross entropy, 20 epochs.
        net.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
        started = time.time()
        history = net.fit(
            x[train_idx],
            y[train_idx],
            validation_data=(x[val_idx], y[val_idx]),
            epochs=epochs,
            batch_size=BATCH_SIZE,
            verbose=2,
        )
        fit_seconds = time.time() - started
        test_loss, test_accuracy = net.evaluate(
            x[test_idx], y[test_idx], batch_size=BATCH_SIZE, verbose=0
        )
        predicted = net.predict(x[test_idx], batch_size=BATCH_SIZE, verbose=0).argmax(axis=1)

        # Checkpoint round trip: the saved weights reproduce the test metrics.
        net.save_weights(f"{out}/final.weights.h5")
        reloaded = build(model)
        reloaded.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
        reloaded.load_weights(f"{out}/final.weights.h5")
        reload_loss, reload_accuracy = reloaded.evaluate(
            x[test_idx], y[test_idx], batch_size=BATCH_SIZE, verbose=0
        )

        metrics = {
            "run_id": run_id,
            "model": model,
            "seed": seed,
            "epochs": epochs,
            "batch_size": BATCH_SIZE,
            "per_class": per_class,
            "pixel_scale": pixel_scale,
            "total_params": int(net.count_params()),
            "test_accuracy": float(test_accuracy),
            "test_loss": float(test_loss),
            "test_correct": int((predicted == labels[test_idx]).sum()),
            "test_count": int(len(test_idx)),
            "test_classes_present": int(len(set(labels[test_idx].tolist()))),
            "reloaded_test_accuracy": float(reload_accuracy),
            "reloaded_test_loss": float(reload_loss),
            "split_counts": [int(len(train_idx)), int(len(val_idx)), int(len(test_idx))],
            "history": {key: [float(v) for v in values] for key, values in history.history.items()},
            "confusion_matrix": confusion_matrix(
                labels[test_idx], predicted, labels=list(range(N_CLASSES))
            ).tolist(),
            "class_names": class_names,
            "test_files": names[test_idx].tolist(),
            "test_true": labels[test_idx].tolist(),
            "test_predicted": predicted.tolist(),
            "fit_seconds": round(fit_seconds, 1),
            "peak_gpu_memory_bytes": int(tf.config.experimental.get_memory_info("GPU:0")["peak"]),
            "environment": {
                "tensorflow": tf.__version__,
                "gpu": gpu,
                "source_sha256": sha256_file(__file__),
                "manifest_sha256": sha256_file(f"{DATA_ROOT}/MANIFEST.sha256"),
                "function_call_id": modal.current_function_call_id(),
                "weights_sha256": sha256_file(f"{out}/final.weights.h5"),
                "pip_freeze_sha256": sha256_file(f"/results/{run_id}/pip-freeze.txt"),
            },
        }
        with open(f"{out}/metrics.json", "w") as handle:
            json.dump(metrics, handle, indent=1)
        results.commit()
        summary = {
            key: metrics[key]
            for key in (
                "model",
                "seed",
                "total_params",
                "test_accuracy",
                "test_loss",
                "test_correct",
                "test_count",
                "reloaded_test_accuracy",
                "fit_seconds",
                "peak_gpu_memory_bytes",
            )
        }
        summary["metrics_sha256"] = sha256_file(f"{out}/metrics.json")
        summary["environment"] = metrics["environment"]
        print("RESULT " + json.dumps(summary))
