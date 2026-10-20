# Copyright (c) 2026 Beijing Horizon Information Technology Company Limited.
# Author: Ego4WAM Team.
# SPDX-License-Identifier: MIT
"""Diffusers ``AutoencoderKLWan`` adapter.

Wraps `diffusers.AutoencoderKLWan` so it conforms to the
the VAE interface expected by the FastWAM framework.

The adapter handles three differences between our hand-written VAE and the
diffusers implementation:

1. **Interface**: ``encode(videos, device, tiled, tile_size, tile_stride)``
   and ``decode(latents, device, tiled, tile_size, tile_stride)`` are mapped
   to diffusers ``_encode`` / ``_decode`` (bypassing the
   ``DiagonalGaussianDistribution`` wrapper).

2. **Scaling**: diffusers VAE does not apply ``latents_mean`` / ``latents_std``
   scaling internally (it is done in the pipeline).  The adapter applies it
   so that the output is identical to the hand-written VAE.

3. **Checkpoint loading**: the adapter loads from HuggingFace Hub
   (``from_pretrained``), a local diffusers pipeline directory containing
   ``vae/``, or a standalone diffusers VAE directory.
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


def _is_wan_vae_config(config_path: str) -> bool:
    """Return whether ``config_path`` identifies a standalone Wan VAE."""
    if not os.path.isfile(config_path):
        return False
    try:
        with open(config_path, encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read VAE config {config_path!r}: {exc}") from exc
    return config.get("_class_name") == "AutoencoderKLWan"


class DiffusersWanVAE(nn.Module):
    """diffusers ``AutoencoderKLWan`` wrapped for the FastWAM framework VAE interface.

    Parameters
    ----------
    pretrained_path :
        Local directory containing diffusers-format VAE safetensors, or a
        HuggingFace model id (e.g. ``"Wan-AI/Wan2.2-TI2V-5B-Diffusers"``).
    torch_dtype :
        Data type for the VAE weights.
    """

    def __init__(
        self,
        pretrained_path: str,
        torch_dtype: torch.dtype = torch.float32,
        init_from_config_only: bool = False,
        **kwargs: object,  # absorb z_dim / dim from build_vae factory
    ) -> None:
        super().__init__()
        from diffusers import AutoencoderKLWan

        if os.path.isdir(pretrained_path):
            pipeline_vae_config = os.path.join(pretrained_path, "vae", "config.json")
            component_config = os.path.join(pretrained_path, "config.json")
            if os.path.isfile(pipeline_vae_config):
                load_kwargs = {"subfolder": "vae"}
            elif _is_wan_vae_config(component_config):
                load_kwargs = {}
            else:
                raise FileNotFoundError(
                    f"Local path {pretrained_path!r} is neither a diffusers pipeline "
                    "containing vae/config.json nor a standalone AutoencoderKLWan "
                    "directory. Configure world_model.vae_pretrained_path separately."
                )
        else:
            load_kwargs = {"subfolder": "vae"}

        if init_from_config_only:
            config = AutoencoderKLWan.load_config(pretrained_path, **load_kwargs)
            with torch.device("meta"):
                self._vae = AutoencoderKLWan.from_config(config)
            self._vae = self._vae.to(dtype=torch_dtype)
            logger.info("Built diffusers Wan VAE from config: %s", pretrained_path)
        else:
            self._vae = AutoencoderKLWan.from_pretrained(
                pretrained_path,
                torch_dtype=torch_dtype,
                **load_kwargs,
            )
            logger.info("Loaded diffusers Wan VAE weights from: %s", pretrained_path)

        meta_parameters = [name for name, parameter in self._vae.named_parameters() if parameter.is_meta]
        if meta_parameters and not init_from_config_only:
            preview = ", ".join(meta_parameters[:5])
            raise RuntimeError(
                f"Wan VAE loaded from {pretrained_path!r} still has {len(meta_parameters)} meta "
                f"parameters (for example: {preview}). The checkpoint is incomplete or uses the "
                "wrong format; refusing to materialize uninitialized weights with to_empty()."
            )

        # --- metadata ---
        self.upsampling_factor: int = int(self._vae.config.scale_factor_spatial)
        self.temporal_downsample_factor: int = int(self._vae.config.scale_factor_temporal)
        self.z_dim: int = int(self._vae.config.z_dim)

        # --- scaling buffers ---
        # latents_mean / latents_std come from the checkpoint config.json.
        _mean = torch.tensor(self._vae.config.latents_mean, dtype=torch.float32).view(1, -1, 1, 1, 1)
        _std = torch.tensor(self._vae.config.latents_std, dtype=torch.float32).view(1, -1, 1, 1, 1)
        self.register_buffer("_mean", _mean)
        self.register_buffer("_inv_std", 1.0 / _std)

    # ------------------------------------------------------------------
    # VAE interface
    # ------------------------------------------------------------------

    def encode(
        self,
        videos: torch.Tensor | list[torch.Tensor],
        device: torch.device | str,
        tiled: bool = False,
        tile_size: tuple[int, int] = (34, 34),
        tile_stride: tuple[int, int] = (18, 16),
    ) -> torch.Tensor:
        """Encode video(s) → latents ``[B, z_dim, T_lat, H_lat, W_lat]``.

        *videos* can be a ``[B, C, T, H, W]`` tensor or a list of ``[C, T, H, W]``
        tensors (matching the original ``WanVideoVAE`` API).  Batched tensors are
        passed directly to the diffusers VAE instead of looping over samples.
        """
        if isinstance(videos, list):
            videos = torch.stack([v.unsqueeze(0) for v in videos])

        videos = videos.to(device)
        self._vae.use_tiling = tiled
        if tiled:
            self._configure_tiles(tile_size, tile_stride)

        posterior = self._vae.encode(videos)
        latent = posterior.latent_dist.mode()
        # Apply scaling in FP32 for low-precision VAEs, then restore the VAE
        # output dtype.  Leaving the FP32 scaling buffers to promote ``latent``
        # makes downstream BF16 Conv3d depend on an ambient autocast context.
        output_dtype = latent.dtype
        scaling_dtype = torch.float32 if output_dtype in (torch.float16, torch.bfloat16) else output_dtype
        latent = latent.to(scaling_dtype)
        latent = (latent - self._mean.to(device=device, dtype=scaling_dtype)) * self._inv_std.to(
            device=device,
            dtype=scaling_dtype,
        )
        return latent.to(output_dtype)

    def decode(
        self,
        hidden_states: torch.Tensor,
        device: torch.device | str,
        tiled: bool = False,
        tile_size: tuple[int, int] = (34, 34),
        tile_stride: tuple[int, int] = (18, 16),
    ) -> torch.Tensor:
        """Decode latents ``[B, z_dim, T_lat, H_lat, W_lat]`` → video ``[B, 3, T, H, W]``."""
        self._vae.use_tiling = tiled
        if tiled:
            self._configure_tiles(tile_size, tile_stride)

        z = hidden_states.to(device)
        # Apply inverse scaling: z * std + mean
        vae_dtype = next(self._vae.parameters()).dtype
        scaling_dtype = torch.float32 if z.dtype in (torch.float16, torch.bfloat16) else z.dtype
        z = z.to(scaling_dtype)
        z = z / self._inv_std.to(device=device, dtype=scaling_dtype) + self._mean.to(
            device=device,
            dtype=scaling_dtype,
        )
        z = z.to(vae_dtype)
        result = self._vae._decode(z, return_dict=True)
        return result.sample  # [B, 3, T, H, W]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _configure_tiles(
        self,
        tile_size: tuple[int, int],
        tile_stride: tuple[int, int],
    ) -> None:
        """Map our latent-space tile sizes to diffusers pixel-space tile params."""
        sf = self.upsampling_factor
        self._vae.tile_sample_min_height = tile_size[0] * sf
        self._vae.tile_sample_min_width = tile_size[1] * sf
        self._vae.tile_sample_stride_height = tile_stride[0] * sf
        self._vae.tile_sample_stride_width = tile_stride[1] * sf


__all__ = ["DiffusersWanVAE"]
