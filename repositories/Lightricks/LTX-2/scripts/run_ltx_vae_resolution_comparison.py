#!/usr/bin/env python3
"""LTX-2.3 VAE resolution comparison: encode/decode at multiple resolutions, side-by-side.

Produces a single video with columns: original | recon@256 | recon@224 | ... | recon@64
All columns scaled to the same display size, with per-frame PSNR overlaid.

Usage:
    python -m scripts.run_ltx_vae_resolution_comparison \
        --input tmp2.mp4 \
        --checkpoint /path/to/vae.safetensors \
        --max-frames 145 \
        --output output/comparison_res.mp4
"""

import argparse
import sys
import time

import numpy as np
import torch


def psnr(orig: np.ndarray, recon: np.ndarray) -> float:
    """Compute PSNR between two uint8 frames."""
    mse = np.mean((orig.astype(np.float64) - recon.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0**2 / mse)


def main():
    parser = argparse.ArgumentParser(description="LTX-2.3 VAE multi-resolution comparison")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--checkpoint", required=True, help="Path to VAE safetensors file")
    parser.add_argument("--max-frames", type=int, default=145, help="Max frames (must be 1+8k)")
    parser.add_argument("--display-size", type=int, default=512, help="Display size per column")
    parser.add_argument("--output", default="output/comparison_res.mp4", help="Output path")
    parser.add_argument("--device", default="cuda", help="Device")
    args = parser.parse_args()

    resolutions = [256, 224, 192, 160, 128, 64]

    if (args.max_frames - 1) % 8 != 0:
        valid = [1 + 8 * k for k in range(10)]
        print(f"ERROR: --max-frames must be 1+8k. Valid values: {valid}")
        sys.exit(1)

    # Validate all resolutions are divisible by 32 (spatial compression factor)
    for res in resolutions:
        if res % 32 != 0:
            print(f"ERROR: resolution {res} not divisible by 32")
            sys.exit(1)

    # --- Load video ---
    print(f"\n=== Loading video from {args.input} ===")
    from src.video_io import load_video
    frames, fps = load_video(args.input, max_frames=args.max_frames)
    print(f"  {len(frames)} frames, {fps:.1f} fps, shape={frames.shape}")

    # --- Load VAE (once) ---
    print(f"\n=== Loading VAE ===")
    from src.ltx_vae import load_ltx_vae
    encoder, decoder = load_ltx_vae(args.checkpoint, device=args.device)

    # --- Preprocess original at display size for the first column ---
    from src.preprocess import preprocess_frames
    from src.visualize import _add_label, resize_frames

    orig_display = resize_frames(
        preprocess_frames(frames, size=max(resolutions))[0],
        args.display_size,
    )

    # --- Encode/decode at each resolution ---
    from src.ltx_vae import decode, encode
    from src.visualize import postprocess_decoded

    recon_columns = {}
    psnr_per_res = {}
    for res in resolutions:
        print(f"\n=== Resolution {res}x{res} ===")
        _, vae_input = preprocess_frames(frames, size=res)
        vae_input = vae_input.to(device=args.device, dtype=torch.float32)
        print(f"  Input: {vae_input.shape}")

        t0 = time.time()
        latents = encode(encoder, vae_input)
        torch.cuda.synchronize()
        print(f"  Latent: {latents.shape}  (encode {time.time() - t0:.2f}s)")

        t0 = time.time()
        recon = decode(decoder, latents)
        torch.cuda.synchronize()
        print(f"  Recon:  {recon.shape}  (decode {time.time() - t0:.2f}s)")

        recon_display = postprocess_decoded(recon.cpu(), size=args.display_size)
        recon_columns[res] = recon_display

        # Compute per-frame PSNR
        frame_psnrs = [psnr(orig_display[i], recon_display[i]) for i in range(len(frames))]
        psnr_per_res[res] = frame_psnrs
        avg = np.mean(frame_psnrs)
        print(f"  PSNR:   avg={avg:.2f} dB, min={min(frame_psnrs):.2f}, max={max(frame_psnrs):.2f}")

        del vae_input, latents, recon
        torch.cuda.empty_cache()

    # --- Compose multi-column video ---
    print(f"\n=== Composing comparison video ===")
    n = len(frames)
    all_frames = []
    for i in range(n):
        panels = [_add_label(orig_display[i], "original")]
        for res in resolutions:
            p = psnr_per_res[res][i]
            panels.append(_add_label(recon_columns[res][i], f"{res}  {p:.1f}dB"))
        all_frames.append(np.concatenate(panels, axis=1))

    all_frames = np.stack(all_frames)
    print(f"  Output shape: {all_frames.shape}")

    from src.video_io import save_video
    save_video(args.output, all_frames, fps)

    # --- Summary ---
    print(f"\n=== Summary ===")
    for res in resolutions:
        avg = np.mean(psnr_per_res[res])
        print(f"  {res}x{res}: avg PSNR = {avg:.2f} dB")

    print(f"\n  Output: {args.output}")
    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"  Peak VRAM: {peak_mb:.0f} MB")


if __name__ == "__main__":
    main()
