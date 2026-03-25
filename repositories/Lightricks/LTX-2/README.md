# LTX-2 Video VAE Reproduction

GitHub: https://github.com/Lightricks/LTX-2

## Goal

Round-trip a real video through the **LTX-2.3 video VAE**:

`mp4 -> decord frames -> center crop -> 256x256 -> VAE encode -> VAE decode -> comparison.mp4`

## Build

```bash
docker build -t ltx-2:latest repositories/Lightricks/LTX-2
```

## Pre-run setup

```bash
# Download the test video
wget -O tmp.mp4 https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4

# Download the VAE checkpoint (1.45 GB)
export HUGGINGFACE_HUB_CACHE=/shared/.cache/huggingface/hub
VAE_PATH=$(huggingface-cli download unsloth/LTX-2.3-GGUF \
  vae/ltx-2.3-22b-distilled_video_vae.safetensors | tail -n 1)
echo "VAE checkpoint: $VAE_PATH"
```

## Run

```bash
VAE_PATH=$(HUGGINGFACE_HUB_CACHE=/shared/.cache/huggingface/hub \
  huggingface-cli download unsloth/LTX-2.3-GGUF \
  vae/ltx-2.3-22b-distilled_video_vae.safetensors | tail -n 1)

export BASE_DIR=$(pwd)/repositories/Lightricks/LTX-2

docker build -t ltx-2:latest repositories/Lightricks/LTX-2 && \
docker run --rm --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /shared/.cache:/shared/.cache \
  -v "$(pwd)/tmp.mp4":/workspace/tmp.mp4:ro \
  -v "$BASE_DIR/output":/workspace/output \
  ltx-2:latest \
  python -m scripts.run_ltx_vae_roundtrip \
    --input tmp.mp4 \
    --checkpoint "$VAE_PATH" \
    --max-frames 33 \
    --size 256 \
    --output output/comparison.mp4
```

## Verified output

```
Encoder params: 318.9M
Decoder params: 407.2M
Tensor shape:   torch.Size([1, 3, 33, 256, 256])
Latent shape:   torch.Size([1, 128, 5, 8, 8])
Latent mean:    0.0083, std: 0.7784
Recon shape:    torch.Size([1, 3, 33, 256, 256])
Encode time:    0.36s
Decode time:    0.29s
Peak VRAM:      3530 MB
```

## Architecture notes

- **LTX-2.3 Video VAE** compresses video with 32x spatial and 8x temporal compression
- Input shape: `(B, 3, F, H, W)` where F must be `1 + 8k` (e.g., 9, 17, 25, 33)
- Latent shape: `(B, 128, F', H', W')` where `F' = 1 + (F-1)/8`, `H' = H/32`, `W' = W/32`
- For 33 frames at 256x256: latent is `(1, 128, 5, 8, 8)`
- VAE config is embedded in safetensors metadata (encoder/decoder block layout)
- VAE checkpoint: `unsloth/LTX-2.3-GGUF` -> `vae/ltx-2.3-22b-distilled_video_vae.safetensors`
- Uses the official `ltx-core` package from the LTX-2 repo for model classes
