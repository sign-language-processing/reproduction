# Stage 1 Hybrid Tokenizer — 256x256

Stage 1 training of the hybrid FSQ tokenizer on pre-extracted LTX-2.3 VAE latents at 256x256 resolution. Each clip is 65 latent frames × 8×8 spatial × 128 channels (513 pixel frames). See [sign/ltx-vae-quantizer](https://github.com/sign/ltx-vae-quantizer) for the repo and CLI.

## Hybrid first-frame mode

The tokenizer always encodes every frame through FSQ. How frame 0 is *decoded* is a runtime choice:
- **all-discrete** (default at inference): frame 0 is reconstructed from its own code, identical to every other frame.
- **passthrough**: frame 0 output is replaced with the real input latent `z[:, 0]`, projected through a per-position `dec_cond` layer so the decoder can condition on it.

During training, each batch flips a coin (`--passthrough-prob 0.5`) to pick a mode. Both modes share every weight, so one checkpoint serves both at inference. Validation always runs all-discrete.

## Training config

- Model: 20.64M params, `fsq_levels=[4]*7`, `num_codebooks=1`
- Optimizer: AdamW, peak LR 2e-3, 500-step warmup, flat afterwards
- Batch: 1 clip (`[1, 65, 8, 8, 128]`)
- Loss: Huber (latent-space, all frames)
- Train/val split: 184 / 10 clips
- Validation: every 10 min of wall time (`--val-interval-minutes 10`), always all-discrete
- `--passthrough-prob 0.5`
- Hardware: 1 GPU, `ltx-vae-quantizer:latest` docker image

## Results

All numbers use the 513-frame `Elan04.mp4` val clip and the all-discrete path. VAE-only is the lower bound: pixel → VAE encode → VAE decode with no tokenizer.

| Metric                | Value              |
|-----------------------|--------------------|
| Wall time             | ~2h 30m            |
| Step throughput       | ~0.2 s/step        |
| Best step             | 45,841 (epoch 499) |
| Best val huber        | **0.03635**        |
| LPIPS (all-discrete)  | **0.0447**         |
| PSNR (all-discrete)   | 29.51              |
| VAE-only LPIPS        | 0.0088             |
| VAE-only PSNR         | 36.19              |

The run was stopped manually after huber-loss divergence around step 48,000 — `best.pt` holds the pre-divergence checkpoint. See `metrics.json` for the full trajectory and `val_comparison.mp4` for the rendered side-by-side.

## Reproduce

```bash
docker run --rm --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /path/to/latents_256:/workspace/latents:ro \
  -v $(pwd):/workspace/output \
  ltx-vae-quantizer \
  python -m ltx_vae_quantizer.train \
    --latents-dir /workspace/latents \
    --output-dir /workspace/output \
    --passthrough-prob 0.5 \
    --val-interval-minutes 10
```
