# Modal Issue: Inconsistent CUDA Driver Versions Across A100-80GB Nodes

## Problem

When running `modal run --detach app.py` with `gpu="A100-80GB"`, we intermittently get assigned nodes
with incompatible NVIDIA driver versions. Our container image is built from
`nvcr.io/nvidia/pytorch:26.02-py3` (CUDA 13.1 toolkit).

## Observed Behavior

| Run | GPU Model | Driver Version | CUDA Version | Result |
|-----|-----------|---------------|--------------|--------|
| 1 | A100-SXM4-40GB | 580.95.05 | 13.0 | Works (Minor Version Compatibility) |
| 2 | A100 80GB PCIe | 570.86.15 | 12.8 | Fails — driver too old |
| 3 | A100-SXM4-80GB | 580.95.05 | 13.0 | Works |
| 4 | A100 80GB PCIe | 570.86.15 | 12.8 | Fails — driver too old |

## Error Message

```
ERROR: This container was built for NVIDIA Driver Release 590.48 or later, but
       version 570.86.15 was detected and compatibility mode is UNAVAILABLE.
```

And from PyTorch:
```
RuntimeError: The NVIDIA driver on your system is too old (found version 12080).
```

## Expected Behavior

Per [Modal CUDA docs](https://modal.com/docs/guide/cuda), the driver version should be **580.95.05**
with CUDA 13.0 across all GPU nodes. CUDA 12.* and 13.* toolkits should be compatible.

## Ask

1. Are there A100-80GB nodes still running driver 570.86.15 (CUDA 12.8)?
2. Can these nodes be updated to 580.95.05+ to match the documented version?
3. Is there a way to request a minimum driver version to avoid being assigned to incompatible nodes?
