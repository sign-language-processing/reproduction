"""
Run EVATok encode->decode on full videos, chunked into 16-frame windows.
Saves a side-by-side GT|Reconstruction MP4 for each input video.

Usage (inside the container):
    python run_full_video.py \
        --input-dir  /input_videos \
        --output-dir results/full_video \
        --model-config configs/vq/VQ_SB_final_with_router_w_lpips_1.2.yaml \
        --vq-ckpt    ckpts/VQ_SB_final_with_router_ucf_k600_1000k.pt \
        [--display-size 512] \
        [--fps 25]
"""

import argparse
import os
import subprocess
import numpy as np
import torch
import yaml
from pathlib import Path
from torchvision import transforms

import decord
decord.bridge.set_bridge("native")
from decord import VideoReader, cpu

import cv2


def load_model(config_path, ckpt_path, device):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    from utils.model_init import load_model_from_config
    model = load_model_from_config(config)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("ema") or ckpt.get("model") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    init = config["model"]["init_args"]
    return model, init["frame_num"], init["input_size"], init["temporal_patch_size"], init["spatial_token_num_choices"]


@torch.no_grad()
def reconstruct_chunk(model, frames_chw, device, temporal_patch_size, token_choices):
    """
    frames_chw: (C, T, H, W) float [0, 1]
    Returns: (C, T, H, W) float [0, 1]
    """
    x = (frames_chw - 0.5).unsqueeze(0).to(device)
    T_segments = frames_chw.shape[1] // temporal_patch_size
    num_latent_tokens = torch.tensor(
        [[max(token_choices)] * T_segments], dtype=torch.long, device=device
    )

    latent, _, [_, _, indices] = model.encode(x, num_latent_tokens=num_latent_tokens)
    recon = model.decode_code(indices, num_latent_tokens=num_latent_tokens, shape=latent.shape)
    return torch.clamp(recon.squeeze(0) + 0.5, 0.0, 1.0).cpu()


def tensor_to_uint8(tensor_chw):
    """(C, H, W) float [0,1] -> (H, W, C) uint8 numpy."""
    return (tensor_chw.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def process_video(video_path, model, frame_num, input_size, temporal_patch_size,
                  token_choices, display_size, out_fps, output_dir, device):
    vr = VideoReader(str(video_path), ctx=cpu(0))
    total_frames = len(vr)
    fps = out_fps or float(vr.get_avg_fps())

    all_frames = vr.get_batch(list(range(total_frames))).asnumpy()  # (T, H, W, C) uint8

    # Pad to at least one full chunk
    if total_frames < frame_num:
        pad = frame_num - total_frames
        all_frames = np.concatenate([all_frames, np.repeat(all_frames[[-1]], pad, axis=0)])

    resize = transforms.Resize((input_size, input_size), antialias=True)
    to_tensor = transforms.ToTensor()
    n_chunks = max(1, total_frames // frame_num)
    interp = cv2.INTER_LANCZOS4

    gt_frames = []
    rec_frames = []

    for i in range(n_chunks):
        chunk = all_frames[i * frame_num : (i + 1) * frame_num]  # (16, H, W, C) uint8

        model_input = torch.stack([resize(to_tensor(f)) for f in chunk], dim=1)  # (C, T, H, W)
        recon = reconstruct_chunk(model, model_input, device, temporal_patch_size, token_choices)

        for t in range(frame_num):
            gt_f = cv2.resize(chunk[t], (display_size, display_size), interpolation=interp)
            rec_f = cv2.resize(tensor_to_uint8(recon[:, t]), (display_size, display_size), interpolation=interp)
            gt_frames.append(gt_f)
            rec_frames.append(rec_f)

    # Write side-by-side MP4
    out_path = os.path.join(output_dir, f"{Path(video_path).stem}_sbs.mp4")
    tmp_path = out_path.replace(".mp4", "_tmp.mp4")

    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (display_size * 2, display_size))
    for gt_f, rec_f in zip(gt_frames, rec_frames):
        writer.write(cv2.cvtColor(np.concatenate([gt_f, rec_f], axis=1), cv2.COLOR_RGB2BGR))
    writer.release()

    # Re-encode for broad compatibility
    subprocess.run(
        ["ffmpeg", "-y", "-i", tmp_path, "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    os.remove(tmp_path)
    print(f"Saved: {out_path}  ({n_chunks} chunks x {frame_num} frames)")


def main():
    parser = argparse.ArgumentParser(description="EVATok full-video encode/decode with side-by-side output")
    parser.add_argument("--input-dir",    required=True)
    parser.add_argument("--output-dir",   default="results/full_video")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--vq-ckpt",      required=True)
    parser.add_argument("--display-size", type=int, default=128)
    parser.add_argument("--fps",          type=float, default=None)
    parser.add_argument("--videos",       nargs="*", default=None,
                        help="stems to process (default: all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, frame_num, input_size, temporal_patch_size, token_choices = \
        load_model(args.model_config, args.vq_ckpt, device)
    print(f"frame_num={frame_num}, input_size={input_size}, "
          f"temporal_patch_size={temporal_patch_size}, max_tokens={max(token_choices)}, device={device}")

    videos = sorted(Path(args.input_dir).glob("*.mp4"))
    if args.videos:
        videos = [v for v in videos if v.stem in args.videos]
    print(f"Processing {len(videos)} videos from {args.input_dir}")

    for video_path in videos:
        try:
            process_video(video_path, model, frame_num, input_size, temporal_patch_size,
                          token_choices, args.display_size, args.fps, args.output_dir, device)
        except Exception as e:
            print(f"ERROR on {video_path.name}: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()