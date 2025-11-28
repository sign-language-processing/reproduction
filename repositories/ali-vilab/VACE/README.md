# VACE Reproduction

https://github.com/ali-vilab/VACE

## Build

Base image:

```bash
docker build -t ali-vilab-vace:latest -f repositories/ali-vilab/VACE/Dockerfile .
```

Wan variant (requires base image):

```bash
docker build -t ali-vilab-vace:wan -f repositories/ali-vilab/VACE/wan.Dockerfile .
```

LTX variant (requires base image):

```bash
docker build -t ali-vilab-vace:ltx -f repositories/ali-vilab/VACE/ltx.Dockerfile .
```

## Run

All run commands use NVIDIA recommended flags for PyTorch containers:
- `--ipc=host` - Use host's shared memory (required for PyTorch multiprocessing)
- `--ulimit memlock=-1` - Unlimited locked memory
- `--ulimit stack=67108864` - 64MB stack size

### Wan

```bash
docker run --rm --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /shared/.cache/huggingface:/root/.cache/huggingface \
  -it ali-vilab-vace:wan bash
```

Download model and create symlink:

```bash
hf download Wan-AI/Wan2.1-VACE-1.3B
ln -s ~/.cache/huggingface/hub/models--Wan-AI--Wan2.1-VACE-1.3B/snapshots/*/. Wan2.1-VACE-1.3B
```

Run inference:

```bash
python vace/vace_wan_inference.py \
  --ckpt_dir ./Wan2.1-VACE-1.3B \
  --src_video examples/videos/girl.mp4 \
  --prompt "A girl is walking"
```

### LTX

```bash
docker run --rm --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /shared/.cache/huggingface:/root/.cache/huggingface \
  -it ali-vilab-vace:ltx bash
```

Download model and create symlink:

```bash
hf download ali-vilab/VACE-LTX-Video-0.9
ln -s ~/.cache/huggingface/hub/models--ali-vilab--VACE-LTX-Video-0.9/snapshots/*/. VACE-LTX-Video-0.9
```

Run inference:

```bash
python vace/vace_ltx_inference.py \
  --ckpt_path ./VACE-LTX-Video-0.9/ltx-video-2b-v0.9.safetensors \
  --src_video examples/videos/girl.mp4 \
  --prompt "A girl is walking"
```
