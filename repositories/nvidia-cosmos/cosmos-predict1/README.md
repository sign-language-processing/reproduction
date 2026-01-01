# Cosmos-Predict1

GitHub: https://github.com/nvidia-cosmos/cosmos-predict1

## Build

```bash
docker build -t cosmos-predict1:latest -f repositories/nvidia-cosmos/cosmos-predict1/Dockerfile .
```

## Run

Training environment test:
```bash
docker run --rm --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  cosmos-predict1:latest \
  bash -c "python scripts/test_environment.py --training"
```

### Training

```bash
export BASE_DIR=$(pwd)/repositories/nvidia-cosmos/cosmos-predict1

# Download the pretrained tokenizer model
mkdir -p $BASE_DIR/checkpoints
export HUGGINGFACE_HUB_CACHE=/shared/.cache/huggingface/hub
for m in CV8x8x8-720p DV8x16x16-720p CV4x8x8-360p DV4x8x8-360p; do
  s=$(hf download nvidia/Cosmos-Tokenize1-$m | tail -n 1)
  ln -sfn "$s" "$BASE_DIR/checkpoints/Cosmos-Tokenize1-$m"
done

# Create a directory with a sample video
rm -rf $BASE_DIR/videos
mkdir -p $BASE_DIR/videos
wget -O tmp.mp4 https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4
ffmpeg -i tmp.mp4 -vf "crop='min(iw,ih)':'min(iw,ih)',scale=64:64" -c:a copy "$BASE_DIR/videos/video.mp4"
rm tmp.mp4

# The docker instance:
# 1. Shares the cache with the host to avoid re-downloading models.
# 2. Mounts the checkpoints directory to have access to pretrained models.
# 3. Mounts the video dataset directory to replace hdvilas used in the config.
docker build -t cosmos-predict1:latest -f repositories/nvidia-cosmos/cosmos-predict1/Dockerfile . && \
docker run --name cosmos-train --rm --gpus all \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --memory=100g --memory-swap=100g \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /shared/.cache:/root/.cache \
  -v /shared/.cache:/shared/.cache \
  -v "$BASE_DIR/checkpoints":/workspace/checkpoints \
  -v "$BASE_DIR/videos":/workspace/datasets/hdvila/videos/ \
  -e OUTPUT_ROOT=checkpoints \
  cosmos-predict1:latest \
  torchrun --standalone --nnodes=1 --nproc_per_node=1 \
    -m cosmos_predict1.tokenizer.training.train \
    --config=cosmos_predict1/tokenizer/training/configs/config.py -- \
    experiment=Cosmos_Tokenize1_CV8x8x8_64p_HDVILA
```

Debugging:

```bash
docker exec -it cosmos-train /bin/bash
pip install py-spy
HIGHEST_CPU_PID=$(ps -eo pid,%cpu --sort=-%cpu | awk 'NR==2{print $1}')
py-spy top --pid $HIGHEST_CPU_PID

# Profiling GPU
nvidia-smi dmon -s pucm
```
