"""Thin wrapper around the LTX-2 Video VAE encoder and decoder."""

import json

import torch
from safetensors import safe_open
from safetensors.torch import load_file


def _get_vae_config(checkpoint_path: str) -> dict:
    """Extract VAE config from safetensors metadata."""
    with safe_open(checkpoint_path, framework="pt") as f:
        meta = f.metadata()
    if meta and "config" in meta:
        return json.loads(meta["config"])
    return {}


def load_ltx_vae(checkpoint_path: str, device: str = "cuda"):
    """Load the LTX-2.3 Video VAE encoder and decoder from a safetensors checkpoint.

    Reads the architecture config from the safetensors metadata, instantiates
    VideoEncoder/VideoDecoder with the correct block layout, then loads weights.

    Returns (encoder, decoder) both on the specified device in eval mode.
    """
    from ltx_core.model.video_vae.enums import LogVarianceType, NormLayerType, PaddingModeType
    from ltx_core.model.video_vae.video_vae import VideoDecoder, VideoEncoder

    device = torch.device(device)

    # Read config from safetensors metadata
    config = _get_vae_config(checkpoint_path)
    vae_cfg = config.get("vae", {})
    print(f"  VAE config from metadata: latent_channels={vae_cfg.get('latent_channels')}, "
          f"patch_size={vae_cfg.get('patch_size')}, "
          f"encoder_blocks={len(vae_cfg.get('encoder_blocks', []))} blocks, "
          f"decoder_blocks={len(vae_cfg.get('decoder_blocks', []))} blocks")

    # Instantiate encoder with config from metadata
    encoder = VideoEncoder(
        convolution_dimensions=vae_cfg.get("dims", 3),
        in_channels=vae_cfg.get("in_channels", 3),
        out_channels=vae_cfg.get("latent_channels", 128),
        encoder_blocks=vae_cfg.get("encoder_blocks", []),
        patch_size=vae_cfg.get("patch_size", 4),
        norm_layer=NormLayerType(vae_cfg.get("norm_layer", "pixel_norm")),
        latent_log_var=LogVarianceType(vae_cfg.get("latent_log_var", "uniform")),
        encoder_spatial_padding_mode=PaddingModeType(vae_cfg.get("spatial_padding_mode", "zeros")),
    )

    # Instantiate decoder with config from metadata
    decoder = VideoDecoder(
        convolution_dimensions=vae_cfg.get("dims", 3),
        in_channels=vae_cfg.get("latent_channels", 128),
        out_channels=vae_cfg.get("out_channels", 3),
        decoder_blocks=vae_cfg.get("decoder_blocks", []),
        patch_size=vae_cfg.get("patch_size", 4),
        norm_layer=NormLayerType(vae_cfg.get("norm_layer", "pixel_norm")),
        causal=vae_cfg.get("causal_decoder", False),
        timestep_conditioning=vae_cfg.get("timestep_conditioning", False),
        decoder_spatial_padding_mode=PaddingModeType(vae_cfg.get("spatial_padding_mode", "zeros")),
        base_channels=vae_cfg.get("decoder_base_channels", 128),
    )

    # Load full state dict and split by prefix
    state_dict = load_file(checkpoint_path)
    encoder_sd = {}
    decoder_sd = {}
    for k, v in state_dict.items():
        if k.startswith("encoder."):
            encoder_sd[k[len("encoder."):]] = v
        elif k.startswith("decoder."):
            decoder_sd[k[len("decoder."):]] = v
        elif k.startswith("per_channel_statistics."):
            # Both encoder and decoder have per_channel_statistics
            encoder_sd[k] = v
            decoder_sd[k] = v

    # Load weights
    enc_result = encoder.load_state_dict(encoder_sd, strict=True)
    print(f"  Encoder load: {enc_result}")
    dec_result = decoder.load_state_dict(decoder_sd, strict=True)
    print(f"  Decoder load: {dec_result}")

    encoder = encoder.to(device=device, dtype=torch.float32).eval()
    decoder = decoder.to(device=device, dtype=torch.float32).eval()

    return encoder, decoder


@torch.no_grad()
def encode(encoder, video_tensor: torch.Tensor) -> torch.Tensor:
    """Encode a video tensor to normalized latents.

    The encoder internally normalizes latents using per_channel_statistics.

    Args:
        encoder: LTX VideoEncoder
        video_tensor: (B, C, F, H, W) float tensor in [-1, 1]

    Returns:
        latents: (B, 128, F', H', W') normalized tensor
    """
    return encoder(video_tensor)


@torch.no_grad()
def decode(decoder, latents: torch.Tensor) -> torch.Tensor:
    """Decode latents back to video.

    Args:
        decoder: LTX VideoDecoder
        latents: (B, 128, F', H', W') tensor

    Returns:
        video: (B, C, F, H, W) float tensor in ~[-1, 1]
    """
    return decoder(latents)
