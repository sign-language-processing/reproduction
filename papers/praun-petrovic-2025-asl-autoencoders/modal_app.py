"""Modal entry point for the conditional ASL autoencoder reproduction (arXiv:2501.06942v1)."""

import json
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
app = modal.App("01da9a64-asl-autoencoders")
image = modal.Image.from_registry("ghcr.io/sign-language-processing/reproduction@sha256:305b6165d306192996358ca312d9a751fa409f43063a76dc7758880a8f905291").add_local_file(ROOT / "train.py", "/app/train.py")
datasets = modal.Volume.from_name("datasets")
cache = modal.Volume.from_name("huggingface-cache")
results = modal.Volume.from_name("01da9a64-asl-autoencoders-results", create_if_missing=True)
ENV = {"HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub", "PYTHONHASHSEED": "42"}
DATA = "/datasets/asl-alphabet/source/asl_alphabet_train/asl_alphabet_train"


@app.function(image=image, gpu="L4", cpu=8, memory=32768, timeout=3 * 60 * 60,
              volumes={"/datasets": datasets.read_only(), "/cache/huggingface": cache, "/results": results}, env=ENV)
def run(model: str, run_name: str, extra: str = "") -> dict:
    if run_name != Path(run_name).name or not run_name:
        raise ValueError("run name must be a single path component")
    out = Path("/results") / run_name
    cmd = ["python", "/app/train.py", "--model", model, "--data-root", DATA, "--output-dir", str(out), *extra.split()]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    subprocess.run(f"pip freeze > {out}/pip-freeze.txt; nvidia-smi > {out}/nvidia-smi.txt", shell=True, check=True)
    results.commit()
    return json.loads((out / "metrics.json").read_text())


@app.local_entrypoint()
def main(model: str, run_name: str, extra: str = ""):
    print(json.dumps({k: v for k, v in run.remote(model, run_name, extra).items() if k != "epochs_log"}, indent=2))
