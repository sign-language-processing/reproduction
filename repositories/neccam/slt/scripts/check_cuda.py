#!/usr/bin/env python3
"""Exercise the exact training framework on a real CUDA forward/backward path."""

import torch
from torchtext import data


x = torch.randn(64, 64, device="cuda", requires_grad=True)
loss = (x @ x).sum()
loss.backward()
print(
    torch.__version__,
    torch.version.cuda,
    type(data.Field).__name__,
    float(loss),
    float(x.grad.norm()),
)
