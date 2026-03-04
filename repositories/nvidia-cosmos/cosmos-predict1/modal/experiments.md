# Training Experiments

Training: Cosmos-Tokenize1-DV8x16x16 at 256p on HD-VILA dataset.
batch_size=9 per GPU, num_workers=8, 600s timeout (debugging).

## num_workers (1x A100-80GB)

| num_workers | Speed (s/it) | Notes |
|-------------|-------------|-------|
| 1 | 2.30 | Baseline |
| 4 | 1.36 | 41% faster |
| 8 | 1.02 | 56% faster |
| 16 | 1.37 | No improvement over 8, data loading no longer bottleneck |

**Conclusion**: num_workers=8 is optimal. Beyond that, GPU compute is the bottleneck.

## GPU scaling (num_workers=8)

Modal pricing: A100-80GB $2.50/h, H100 $3.95/h, H200 $4.54/h, B200 $6.25/h.

| Config | Batch/GPU | Speed (s/it) | Samples/sec | $/h | Samples/$ |
|--------|-----------|-------------|-------------|-----|-----------|
| 1x A100-80GB | 9 | 1.02 | 8.8 | $2.50 | 12,672 |
| 2x A100-80GB | 9 | 1.37 | 13.1 | $5.00 | 9,432 |
| 8x A100-80GB | 9 | TBD | TBD | $20.00 | TBD |
| 1x H100-80GB | 9 | 0.73 | 12.3 | $3.95 | 11,211 |
| 1x H200-141GB | 16 | 1.21 | 13.2 | $4.54 | 10,463 |
| 1x B200-192GB | 20 | 1.09 | 18.3 | $6.25 | 10,541 |

**Observations**:
- **1x A100 is the most cost-efficient** at 12,672 samples/$ — cheapest GPU, good enough throughput
- B200 has highest throughput (18.3 samples/sec) thanks to batch=20, but worst cost-efficiency
- H200 and B200 are similar in cost-efficiency (~10,500 samples/$) despite different batch sizes
- H100 beats H200 on cost-efficiency (11,211 vs 10,463) — the extra VRAM doesn't pay off
- 2x A100 DDP is the worst value at 9,432 samples/$

**Conclusion**: 1x A100 for cost, 1x B200 for speed. Larger VRAM (bigger batches) improves throughput but not cost-efficiency.

## Batch size (1 GPU)

| Batch size | Result |
|-----------|--------|
| 8 | Fits, ~68 GB VRAM |
| 9 | Fits, ~74 GB VRAM |
| 10 | OOM (needs 76.3 + 3.8 GiB, only 79.25 available) |

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduced fragmentation but didn't make batch 10 fit.

## Other notes

- pynvml `nvmlDeviceGetCpuAffinity` crashes on Modal cloud GPUs — patched to skip CPU affinity
- Some Modal A100-80GB nodes have driver 570.86.15 (CUDA 12.8), incompatible with our CUDA 13.1 container — see modal.md
