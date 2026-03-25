"""Video preprocessing: center crop, resize, normalize."""

import numpy as np
import torch
import torchvision.transforms.functional as TF


def preprocess_frames(
    frames: np.ndarray, size: int = 256
) -> tuple[np.ndarray, torch.Tensor]:
    """Center crop and resize video frames, returning both display and VAE-ready versions.

    Args:
        frames: (F, H, W, 3) uint8 numpy array
        size: target spatial resolution (square)

    Returns:
        orig_resized: (F, H, W, 3) uint8 numpy array, center-cropped and resized
        vae_input: (1, 3, F, size, size) float32 tensor in [-1, 1], ready for VAE
    """
    processed = []
    for frame in frames:
        t = torch.from_numpy(frame).permute(2, 0, 1)  # (3, H, W)
        # Center crop to square
        _, h, w = t.shape
        crop_size = min(h, w)
        t = TF.center_crop(t, [crop_size, crop_size])
        # Resize
        t = TF.resize(t, [size, size], antialias=True)
        processed.append(t)

    stacked = torch.stack(processed)  # (F, 3, H, W)

    # Display version: uint8 numpy
    orig_resized = stacked.permute(0, 2, 3, 1).numpy().astype(np.uint8)

    # VAE input: float32 in [-1, 1], shape (1, 3, F, H, W)
    vae_input = stacked.float() / 255.0 * 2.0 - 1.0  # [0,255] -> [-1,1]
    vae_input = vae_input.permute(1, 0, 2, 3).unsqueeze(0)  # (1, 3, F, H, W)

    return orig_resized, vae_input
