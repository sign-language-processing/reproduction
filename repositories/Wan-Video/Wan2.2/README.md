# Wan2.2 Reproduction

https://github.com/Wan-Video/Wan2.2

## Build

```bash
docker build -t wan2.2:latest -f repositories/Wan-Video/Wan2.2/Dockerfile .
```

## Run

```bash
docker run --rm --gpus all --ipc=host \
  -v /shared/.cache/huggingface:/root/.cache/huggingface \
  -it wan2.2:latest bash
```

## Usage

Download model and create symlink:

```bash
hf download Wan-AI/Wan2.2-TI2V-5B
ln -s ~/.cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/*/. Wan2.2-TI2V-5B
```

Run generation (TI2V-5B, 24GB VRAM):

```bash
python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B \
  --offload_model True --t5_cpu \
  --prompt "A cat walking on the beach"
```
