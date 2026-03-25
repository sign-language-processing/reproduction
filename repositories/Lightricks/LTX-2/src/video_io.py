"""Video I/O using decord for reading and imageio for writing."""

import numpy as np


def load_video(path: str, max_frames: int | None = None) -> tuple[np.ndarray, float]:
    """Load video frames using decord.

    Args:
        path: path to video file
        max_frames: if set, only load this many frames from the start

    Returns:
        frames: (F, H, W, 3) uint8 numpy array
        fps: frames per second of the source video
    """
    from decord import VideoReader, cpu

    vr = VideoReader(path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    total = len(vr)

    n = min(total, max_frames) if max_frames else total
    indices = list(range(n))
    frames = vr.get_batch(indices).asnumpy()  # (F, H, W, 3) uint8

    return frames, fps


def save_video(path: str, frames: np.ndarray, fps: float):
    """Save frames to an mp4 file using imageio.

    Args:
        path: output file path
        frames: (F, H, W, 3) uint8 numpy array
        fps: output frame rate
    """
    import imageio.v3 as iio

    iio.imwrite(path, frames, fps=fps, codec="libx264", plugin="pyav")
    print(f"  Saved {path} ({len(frames)} frames, {fps:.1f} fps)")
