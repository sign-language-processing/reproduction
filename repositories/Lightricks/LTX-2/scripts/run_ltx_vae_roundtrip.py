#!/usr/bin/env python3
"""LTX-2.3 Video VAE round-trip: mp4 -> encode -> decode -> comparison.mp4

Usage:
    python -m scripts.run_ltx_vae_roundtrip \
        --input tmp.mp4 \
        --checkpoint /root/.cache/huggingface/hub/.../ltx-2.3-22b-distilled_video_vae.safetensors \
        --max-frames 33 \
        --size 256
"""

import argparse
import sys
import time

import torch


def main():
    parser = argparse.ArgumentParser(description="LTX-2.3 VAE round-trip test")
    parser.add_argument("--input", default="tmp.mp4", help="Input video path")
    parser.add_argument("--checkpoint", required=True, help="Path to VAE safetensors file")
    parser.add_argument("--max-frames", type=int, default=33, help="Max frames to process (must be 1+8k)")
    parser.add_argument("--size", type=int, default=256, help="Spatial resolution for VAE input")
    parser.add_argument("--display-size", type=int, default=512, help="Display resolution for comparison video")
    parser.add_argument("--output", default="comparison.mp4", help="Output comparison video path")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--use-tiling", action="store_true", help="Enable VAE tiling for decode")
    args = parser.parse_args()

    # Validate frame count: must be 1 + 8k for LTX temporal compression
    if (args.max_frames - 1) % 8 != 0:
        valid = [1 + 8 * k for k in range(10)]
        print(f"ERROR: --max-frames must be 1+8k. Valid values: {valid}")
        sys.exit(1)

    # --- Step 1: Load video ---
    print(f"\n=== Step 1: Loading video from {args.input} ===")
    from src.video_io import load_video
    frames, fps = load_video(args.input, max_frames=args.max_frames)
    print(f"  Loaded {len(frames)} frames, {fps:.1f} fps, shape={frames.shape}")

    # --- Step 2: Preprocess ---
    print(f"\n=== Step 2: Preprocessing to {args.size}x{args.size} ===")
    from src.preprocess import preprocess_frames
    orig_resized, vae_input = preprocess_frames(frames, size=args.size)
    print(f"  orig_resized: shape={orig_resized.shape}, dtype={orig_resized.dtype}")
    print(f"  vae_input:    shape={vae_input.shape}, dtype={vae_input.dtype}")
    print(f"  vae_input:    min={vae_input.min():.3f}, max={vae_input.max():.3f}")

    # --- Step 3: Load VAE ---
    print(f"\n=== Step 3: Loading LTX-2.3 VAE from {args.checkpoint} ===")
    from src.ltx_vae import load_ltx_vae
    encoder, decoder = load_ltx_vae(args.checkpoint, device=args.device)

    enc_params = sum(p.numel() for p in encoder.parameters())
    dec_params = sum(p.numel() for p in decoder.parameters())
    print(f"  Encoder params: {enc_params / 1e6:.1f}M")
    print(f"  Decoder params: {dec_params / 1e6:.1f}M")

    # --- Step 4: Tensor layout sanity ---
    print("\n=== Step 4: Tensor layout check ===")
    vae_input = vae_input.to(device=args.device, dtype=torch.float32)
    print(f"  Frame count:    {vae_input.shape[2]}")
    print(f"  Tensor shape:   {vae_input.shape}")
    print(f"  dtype:          {vae_input.dtype}")
    print(f"  min/max:        {vae_input.min():.4f} / {vae_input.max():.4f}")
    vram_mb = vae_input.nelement() * vae_input.element_size() / 1024 / 1024
    print(f"  Input VRAM:     {vram_mb:.1f} MB")

    # --- Step 5: Encode ---
    print("\n=== Step 5: Encoding ===")
    torch.cuda.synchronize()
    t0 = time.time()
    from src.ltx_vae import encode
    latents = encode(encoder, vae_input)
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"  Latent shape:   {latents.shape}")
    print(f"  Latent dtype:   {latents.dtype}")
    print(f"  Latent mean:    {latents.mean():.4f}")
    print(f"  Latent std:     {latents.std():.4f}")
    print(f"  Latent min/max: {latents.min():.4f} / {latents.max():.4f}")
    print(f"  Encode time:    {t1 - t0:.2f}s")

    # --- Step 6: Decode ---
    print("\n=== Step 6: Decoding ===")
    torch.cuda.synchronize()
    t0 = time.time()
    from src.ltx_vae import decode

    if args.use_tiling and hasattr(decoder, "tiled_decode"):
        print("  Using tiled decoding")
        recon = decoder.tiled_decode(latents)
    else:
        recon = decode(decoder, latents)
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"  Recon shape:    {recon.shape}")
    print(f"  Recon dtype:    {recon.dtype}")
    print(f"  Recon min/max:  {recon.min():.4f} / {recon.max():.4f}")
    print(f"  Decode time:    {t1 - t0:.2f}s")

    # --- Step 7: Postprocess ---
    print("\n=== Step 7: Postprocessing ===")
    from src.visualize import postprocess_decoded
    recon_frames = postprocess_decoded(recon.cpu(), size=args.display_size)
    print(f"  Recon frames:   shape={recon_frames.shape}, dtype={recon_frames.dtype}")

    # --- Step 8: Comparison video ---
    print(f"\n=== Step 8: Writing comparison video to {args.output} ===")
    from src.visualize import make_comparison_video
    make_comparison_video(orig_resized, recon_frames, args.output, fps, display_size=args.display_size)

    print("\n=== Done! ===")
    print(f"  Input:      {args.input}")
    print(f"  Comparison: {args.output}")
    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"  Peak VRAM:  {peak_mb:.0f} MB")


if __name__ == "__main__":
    main()
