"""Modal entry points for the Kemkar et al. 2023 reproduction."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parent
UPSTREAM_URL = "https://github.com/kemkartanya/Sign-Language-Recognition"
UPSTREAM_COMMIT = "415849b52a3c6db79ea03a40545c4f1445dea386"
# The upstream repository is a re-run copy of this earlier tutorial repository
# (same notebook source, same model.png, different trained checkpoints).
ARTICLE_URL = "https://github.com/Sathwick-Reddy-M/Sign-Language-Recognition"
ARTICLE_COMMIT = "46de18cb6f7fe1f4a5ced4eccdc65b2cd7793fae"
MNIST = "/datasets/sign-language-mnist/files"

app = modal.App("kemkar-2023-sign-language-to-text")
# The paper's environment lives in its own virtualenv because Modal's
# in-container client needs a newer Python and protobuf than TensorFlow 2.9.1
# allows. Python 3.9.13 is the upstream notebook's kernel and the first five
# pins are upstream requirements.txt; the rest only make the notebook
# executable headlessly.
PYTHON = "/opt/env/bin/python"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv==0.4.30")
    .run_commands(
        "uv venv --seed --python 3.9.13 /opt/env",
        f"uv pip install --python {PYTHON} tensorflow==2.9.1 scikit-learn==1.1.1 numpy==1.23.1 pandas==1.4.3 pillow==9.2.0"
        " protobuf==3.19.6 matplotlib==3.5.2 matplotlib-inline==0.1.3 nbconvert==7.2.10 ipykernel==6.15.1",
    )
    # The pinned commit is fetched as GitHub's commit archive.
    .run_commands(
        f"python -c \"import urllib.request; urllib.request.urlretrieve('{UPSTREAM_URL}/archive/{UPSTREAM_COMMIT}.tar.gz', '/tmp/upstream.tar.gz')\""
        " && mkdir /opt/upstream && tar -xzf /tmp/upstream.tar.gz -C /opt/upstream --strip-components=1 && rm /tmp/upstream.tar.gz"
    )
    .run_commands(
        f"python -c \"import urllib.request; urllib.request.urlretrieve('{ARTICLE_URL}/archive/{ARTICLE_COMMIT}.tar.gz', '/tmp/article.tar.gz')\""
        " && mkdir /opt/article && tar -xzf /tmp/article.tar.gz -C /opt/article --strip-components=1 && rm /tmp/article.tar.gz"
    )
    .add_local_file(ROOT / "data.sh", "/app/data.sh")
    .add_local_file(ROOT / "evaluate.py", "/app/evaluate.py")
    .add_local_file(ROOT / "train_asl.py", "/app/train_asl.py")
)
datasets = modal.Volume.from_name("datasets", create_if_missing=False)
cache = modal.Volume.from_name("huggingface-cache", create_if_missing=False)
results = modal.Volume.from_name("kemkar-2023-sign-language-to-text-results", create_if_missing=True)
ENV = {"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub"}
RUN = dict(
    image=image,
    cpu=8,
    memory=16384,
    timeout=4 * 60 * 60,
    volumes={"/datasets": datasets.read_only(), "/cache/huggingface": cache, "/results": results},
    env=ENV,
)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_output(run_name: str) -> Path:
    if run_name in {"", ".", ".."} or run_name != Path(run_name).name:
        raise ValueError("run name must be a single non-empty path component")
    output = Path("/results") / run_name
    if output.exists():
        raise FileExistsError("output exists; retain it rather than overwrite evidence")
    output.mkdir(parents=True)
    return output


def run_logged(command: list[str], output: Path, cwd: str | None = None) -> None:
    import os
    import subprocess

    with (output / "log.txt").open("a", encoding="utf-8") as log:
        log.write(f"$ {' '.join(command)}\n")
        log.flush()
        env = {**os.environ, "PATH": f"/opt/env/bin:{os.environ['PATH']}"}
        subprocess.run(command, check=True, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)


def finish(output: Path, started: str, commands: list[list[str]], data_files: list[str]) -> dict:
    import hashlib
    import os
    import subprocess

    freeze = subprocess.run([PYTHON, "-m", "pip", "freeze"], check=True, capture_output=True, text=True).stdout
    (output / "pip-freeze.txt").write_text(freeze, encoding="utf-8")
    cpu = [line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")]
    run = {
        "started_at_utc": started,
        "finished_at_utc": now(),
        "commands": [" ".join(command) for command in commands],
        "upstream": {"url": UPSTREAM_URL, "commit": UPSTREAM_COMMIT, "notebook_sha256": hashlib.sha256(Path("/opt/upstream/sign-language-recognition.ipynb").read_bytes()).hexdigest()},
        "data_sha256": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in data_files},
        "python": subprocess.run([PYTHON, "--version"], check=True, capture_output=True, text=True).stdout.strip(),
        "cpu_model": cpu[0] if cpu else None,
        "cpu_count": os.cpu_count(),
        "modal": {key: os.environ.get(key) for key in ("MODAL_IMAGE_ID", "MODAL_TASK_ID", "MODAL_ENVIRONMENT", "MODAL_REGION")},
        "metrics": json.loads((output / "metrics.json").read_text()),
    }
    run["sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.iterdir()) if path.is_file()}
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    results.commit()
    del run["metrics"]
    return run


@app.function(image=image, timeout=60 * 60, volumes={"/datasets": datasets, "/cache/huggingface": cache}, env=ENV)
def populate_asl_dataset() -> dict:
    import subprocess

    completed = subprocess.run(["bash", "/app/data.sh"], check=True, capture_output=True, text=True)
    datasets.commit()
    return json.loads(completed.stdout)


@app.function(**RUN)
def mnist_released(run_name: str) -> dict:
    """Evaluate the checkpoint committed in the upstream repository; no training."""
    output, started = new_output(run_name), now()
    command = [PYTHON, "/app/evaluate.py", "--model-dir", "/opt/upstream/models/experiment-dropout-0", "--data-root", MNIST, "--output-dir", str(output)]
    run_logged(command, output)
    return finish(output, started, [command], [f"{MNIST}/sign_mnist_train.csv", f"{MNIST}/sign_mnist_test.csv"])


@app.function(**RUN)
def mnist_article_checkpoint(run_name: str) -> dict:
    """Evaluate the checkpoint committed in the earlier tutorial repository; no training."""
    output, started = new_output(run_name), now()
    command = [PYTHON, "/app/evaluate.py", "--model-dir", "/opt/article/models/experiment-dropout-0", "--data-root", MNIST, "--output-dir", str(output)]
    run_logged(command, output)
    run = finish(output, started, [command], [f"{MNIST}/sign_mnist_train.csv", f"{MNIST}/sign_mnist_test.csv"])
    return {**run, "article": {"url": ARTICLE_URL, "commit": ARTICLE_COMMIT}}


@app.function(**RUN)
def mnist_notebook(run_name: str) -> dict:
    """Execute the upstream notebook unmodified, then evaluate its freshly trained final model."""
    import shutil

    output, started = new_output(run_name), now()
    # The notebook reads data/alphabet/*.csv and writes models/*: link the shared
    # dataset in and drop the committed checkpoints so only fresh ones can load.
    work = Path("/tmp/work")
    shutil.copytree("/opt/upstream", work, ignore=shutil.ignore_patterns("data", "models"))
    (work / "models").mkdir()
    (work / "data" / "alphabet").mkdir(parents=True)
    for name in ("sign_mnist_train.csv", "sign_mnist_test.csv"):
        (work / "data" / "alphabet" / name).symlink_to(f"{MNIST}/{name}")

    execute = ["/opt/env/bin/jupyter", "nbconvert", "--to", "notebook", "--execute", "--ExecutePreprocessor.timeout=-1", "--output", "executed.ipynb", "sign-language-recognition.ipynb"]
    evaluate = [PYTHON, "/app/evaluate.py", "--model-dir", str(work / "models" / "experiment-dropout-0"), "--data-root", MNIST, "--output-dir", str(output)]
    run_logged(execute, output, cwd=str(work))
    run_logged(evaluate, output)
    shutil.copy(work / "executed.ipynb", output / "executed.ipynb")
    shutil.copy(work / "models" / "experiment-dropout-0-history", output / "experiment-dropout-0-history")
    shutil.copytree(work / "models" / "experiment-dropout-0", output / "experiment-dropout-0")
    return finish(output, started, [execute, evaluate], [f"{MNIST}/sign_mnist_train.csv", f"{MNIST}/sign_mnist_test.csv"])


@app.function(**RUN)
def asl_conditional(run_name: str, seed: int = 42) -> dict:
    output, started = new_output(run_name), now()
    command = [PYTHON, "/app/train_asl.py", "--data-root", "/datasets/asl-dataset/files", "--output-dir", str(output), "--seed", str(seed)]
    run_logged(command, output, cwd="/app")
    return finish(output, started, [command], ["/datasets/asl-dataset/manifest.json"])
