# GroundingDINO Reproduction

https://github.com/IDEA-Research/GroundingDINO

## Build

```bash
docker build -t groundingdino:latest -f repositories/IDEA-Research/GroundingDINO/Dockerfile .
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
  -it groundingdino:latest bash
```

## Usage

Inside the container, download model and create symlink:

```bash
hf download ShilongLiu/GroundingDINO
ln -s ~/.cache/huggingface/hub/models--ShilongLiu--GroundingDINO/snapshots/*/. weights
```

Run inference on an image:

```bash
python demo/inference_on_a_image.py \
  -c groundingdino/config/GroundingDINO_SwinT_OGC.py \
  -p weights/groundingdino_swint_ogc.pth \
  -i .asset/cat_dog.jpeg \
  -o output \
  -t "cat . dog"
```

## Available Models

| Model | File | Size |
|-------|------|------|
| Swin-T (48.4 AP) | `groundingdino_swint_ogc.pth` | 694 MB |
| Swin-B (56.7 AP) | `groundingdino_swinb_cogcoor.pth` | 938 MB |
