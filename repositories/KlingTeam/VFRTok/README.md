# VFRTok

Variable frame rate video tokenizer using ViT with rotary position embeddings.

## Setup

```bash
# Download model weights on host
hf download KwaiVGI/VFRTok

# Build
docker compose build
```

## Usage

```bash
# Run inference
docker compose run --rm vfrtok \
    deepspeed inference.py -i test.csv -o outputs \
    --config configs/vfrtok-l.yaml --ckpt vfrtok-l.bin --enc_fps 24
```
