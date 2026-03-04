from pathlib import Path

import modal

COSMOS_DIR = Path(__file__).resolve().parent.parent

app = modal.App("cosmos-predict1")

image = modal.Image.from_dockerfile(
    path=COSMOS_DIR / "Dockerfile",
    context_dir=COSMOS_DIR,
)
# TODO: add local dir with training code

hf_secret = modal.Secret.from_name("huggingface-secret")
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
checkpoints = modal.Volume.from_name("cosmos-checkpoints", create_if_missing=True)
videos = modal.Volume.from_name("videos-256")



@app.function(
    image=image,
    secrets=[hf_secret],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/workspace/checkpoints": checkpoints,
    },
    timeout=3600,
)
def download_checkpoints():
    import os
    import shutil

    from huggingface_hub import snapshot_download

    TOKENIZER_MODELS = [
        "CV8x8x8-720p",
        "DV8x16x16-720p",
        "CV4x8x8-360p",
        "DV4x8x8-360p",
    ]

    for model in TOKENIZER_MODELS:
        repo_id = f"nvidia/Cosmos-Tokenize1-{model}"
        dest = f"/workspace/checkpoints/Cosmos-Tokenize1-{model}"
        if os.path.exists(dest) and os.listdir(dest):
            print(f"Already exists: {dest}, skipping")
            continue
        print(f"Downloading {repo_id}...")
        path = snapshot_download(repo_id=repo_id)
        shutil.copytree(path, dest, dirs_exist_ok=True)
        print(f"Copied to {dest}")
    hf_cache.commit()
    checkpoints.commit()


GPU_BATCH_SIZE = {
    "A100-80GB": 9,   # 80 GB
    "H100": 9,        # 80 GB
    "H200": 17,       # 141 GB
    "B200": 21,       # 178 GB usable
}

GPU_TYPE = "B200"
GPU_COUNT = 1
BATCH_SIZE = GPU_BATCH_SIZE[GPU_TYPE]
NUM_WORKERS = 8


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    cpu=8,
    memory=32 * 1024,
    volumes={
        "/workspace/checkpoints": checkpoints,
        "/workspace/datasets/hdvila/videos": videos,
    },
    timeout=600,
)
def train():
    import os
    import subprocess

    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print(result.stdout)

    env = {
        **os.environ,
        "OUTPUT_ROOT": "checkpoints",
        "BATCH_SIZE": str(BATCH_SIZE),
        "NUM_WORKERS": str(NUM_WORKERS),
    }
    cmd = [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={GPU_COUNT}",
        "-m", "cosmos_predict1.tokenizer.training.train",
        "--config=cosmos_predict1/tokenizer/training/configs/config.py",
        "--",
        "experiment=Cosmos_Tokenize1_DV8x16x16_256p_HDVILA",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)


@app.local_entrypoint()
def main():
    download_checkpoints.remote()
    train.remote()
