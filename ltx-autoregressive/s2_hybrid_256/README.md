# Stage 2 Hybrid Tokenizer — 256x256

Stage 2 fine-tunes the hybrid tokenizer jointly with LoRA adapters on the LTX-2.3 VAE encoder and decoder, using pixel-space losses. Initialized from `s1_hybrid_256/best.pt`. See [sign/ltx-vae-quantizer](https://github.com/sign/ltx-vae-quantizer) for the repo and CLI.

## Hybrid first-frame mode

Per-batch coin flip between all-discrete and passthrough-first-frame decoding (`--passthrough-prob 0.5`). Every frame is encoded through FSQ in both modes; only frame 0's decode differs. Weights are shared, so one checkpoint serves both inference modes. Validation and the final rendered `val_comparison.mp4` always use all-discrete (matches the default inference path).

## Training config

Trainable parameter budget (37.0M total):
- Tokenizer: 20.64M params, fully unfrozen (`--train-tokenizer`)
- VAE encoder LoRA (`--encoder-lora`): 8.65M trainable
- VAE decoder LoRA: 7.69M trainable (of 414.9M total)

Objective:
- MSE ×3 + LPIPS ×3 + KL ×3e-6 + feature distillation ×1e-3
- Feature distillation matches student decoder up_blocks against a teacher running the same LoRA VAE with the tokenizer bypassed; linear 1→0.5 per-layer weights across 9 up_blocks.

Run:
- Optimizer: AdamW, LR 3e-5, 100-step warmup, flat afterwards
- Validation: every 100 steps (pixel-space LPIPS + PSNR, all-discrete)
- Early stopping: 10 checks without LPIPS improvement
- Hardware: 1 GPU, `ltx-vae-quantizer:latest` docker image

## Results

All numbers use the 513-frame `Elan04.mp4` val clip and the all-discrete path.

| Metric                  | Value         |
|-------------------------|---------------|
| Wall time               | ~5h 50m       |
| Step throughput         | ~2.4 s/step   |
| Total steps             | 8,700 (early-stop) |
| Best step (saved)       | 7,700         |
| Best val LPIPS          | **0.0215**    |
| Val PSNR at best LPIPS  | 29.93         |
| Stage 1 baseline LPIPS  | 0.0447        |
| Stage 1 baseline PSNR   | 29.51         |

Stage 2 more than halves LPIPS from the stage 1 init (0.0447 → 0.0215) while holding PSNR ~flat. See `metrics.json` for the full per-step trajectory and `val_comparison.mp4` for the rendered side-by-side.

## Reproduce

```bash
docker run --rm --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /path/to/videos_256:/workspace/videos:ro \
  -v $(pwd):/workspace/output \
  -v /path/to/vae.safetensors:/workspace/vae.safetensors:ro \
  -v /path/to/s1_hybrid_256/best.pt:/workspace/tokenizer.pt:ro \
  ltx-vae-quantizer \
  python -m ltx_vae_quantizer.train_stage2 \
    --vae-checkpoint /workspace/vae.safetensors \
    --tokenizer-checkpoint /workspace/tokenizer.pt \
    --video-dir /workspace/videos \
    --output-dir /workspace/output \
    --encoder-lora \
    --train-tokenizer \
    --passthrough-prob 0.5 \
    --mse-weight 3 --lpips-weight 3 \
    --val-every 100 \
    --early-stopping 10
```
