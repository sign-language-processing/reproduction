# LTX-2 Video VAE

Video tokenizer using the LTX-2 VAE (8x temporal, 32x spatial compression, 128 latent channels).

## Setup

```bash
# Download model weights on host
hf download Lightricks/LTX-2 --include "vae/*"

# Build
docker compose build
```

## Usage

```bash
# Reconstruct video
docker compose run --rm ltx2 python -m video_tokenizer.bin \
    --tokenizer ltx2 \
    --reconstruct /path/to/video.mp4 \
    --output /path/to/output.mp4 \
    --resolution 256x256
```
