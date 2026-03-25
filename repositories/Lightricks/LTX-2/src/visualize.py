"""Side-by-side comparison video with labels."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _add_label(frame: np.ndarray, text: str) -> np.ndarray:
    """Overlay a small text label at the top-left of a frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    # Draw text with dark background for visibility
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([4, 4, 4 + tw + 8, 4 + th + 8], fill=(0, 0, 0, 180))
    draw.text((8, 6), text, fill=(255, 255, 255), font=font)
    return np.array(img)


def postprocess_decoded(decoded_tensor, size: int = 512) -> np.ndarray:
    """Convert decoded VAE output tensor to uint8 frames resized to display size.

    Args:
        decoded_tensor: (1, 3, F, H, W) float tensor in ~[-1, 1]
        size: target display resolution

    Returns:
        frames: (F, size, size, 3) uint8 numpy array
    """
    import torch
    import torchvision.transforms.functional as TF

    video = decoded_tensor[0]  # (3, F, H, W)
    video = video.clamp(-1, 1)
    video = (video + 1.0) / 2.0 * 255.0  # [-1,1] -> [0,255]
    video = video.to(torch.uint8)

    frames = []
    for i in range(video.shape[1]):
        frame = video[:, i]  # (3, H, W)
        frame = TF.resize(frame, [size, size], antialias=True)
        frames.append(frame.permute(1, 2, 0).cpu().numpy())

    return np.stack(frames)


def resize_frames(frames: np.ndarray, size: int = 512) -> np.ndarray:
    """Resize uint8 frames to a target size."""
    import torch
    import torchvision.transforms.functional as TF

    out = []
    for frame in frames:
        t = torch.from_numpy(frame).permute(2, 0, 1)  # (3, H, W)
        t = TF.resize(t, [size, size], antialias=True)
        out.append(t.permute(1, 2, 0).numpy())
    return np.stack(out)


def make_comparison_video(
    orig_frames: np.ndarray,
    recon_frames: np.ndarray,
    output_path: str,
    fps: float,
    display_size: int = 512,
):
    """Create a side-by-side comparison video.

    Args:
        orig_frames: (F, H, W, 3) uint8, preprocessed originals
        recon_frames: (F, H, W, 3) uint8, VAE reconstructions
        output_path: where to write the comparison mp4
        fps: output frame rate
        display_size: resize both sides to this resolution before concatenating
    """
    from src.video_io import save_video

    orig_up = resize_frames(orig_frames, display_size)
    recon_up = resize_frames(recon_frames, display_size)

    n = min(len(orig_up), len(recon_up))
    comparison = []
    for i in range(n):
        left = _add_label(orig_up[i], "original")
        right = _add_label(recon_up[i], "ltx vae recon")
        combined = np.concatenate([left, right], axis=1)  # (H, 2*W, 3)
        comparison.append(combined)

    comparison = np.stack(comparison)
    save_video(output_path, comparison, fps)
