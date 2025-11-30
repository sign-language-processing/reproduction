# VACE Reproduction

https://github.com/ali-vilab/VACE

## Build

```bash
docker build -t ali-vilab-vace:latest -f repositories/ali-vilab/VACE/Dockerfile .
```

## Run

All run commands use NVIDIA recommended flags for PyTorch containers:
- `--ipc=host` - Use host's shared memory (required for PyTorch multiprocessing)
- `--ulimit memlock=-1` - Unlimited locked memory
- `--ulimit stack=67108864` - 64MB stack size

```bash
docker run --rm --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /shared/.cache/huggingface:/root/.cache/huggingface \
  -it ali-vilab-vace:latest bash
```

### Wan

Run inference:

```bash
# Download the Wan2.1-VACE model
hf download Wan-AI/Wan2.1-VACE-1.3B
ln -s ~/.cache/huggingface/hub/models--Wan-AI--Wan2.1-VACE-1.3B/snapshots/*/. models/Wan2.1-VACE-1.3B

# Download only the models you need from "VACE-Annotators" using "include"
hf download ali-vilab/VACE-Annotators --include "depth/*"
ln -s ~/.cache/huggingface/hub/models--ali-vilab--VACE-Annotators/snapshots/*/. models/VACE-Annotators

python vace/vace_pipeline.py \
  --base wan \
  --task depth \
  --video assets/videos/test.mp4 \
  --prompt 'xxx'
```

### LTX

Run inference:

```bash
python vace/vace_ltx_inference.py \
  --ckpt_path ./VACE-LTX-Video-0.9/ltx-video-2b-v0.9.safetensors \
  --src_video examples/videos/girl.mp4 \
  --prompt "A girl is walking"
```
