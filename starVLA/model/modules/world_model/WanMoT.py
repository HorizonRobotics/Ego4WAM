# Copyright (c) 2026 The FastWAM Authors.
# Modifications Copyright (c) 2026 Ego4WAM Authors.
# SPDX-License-Identifier: MIT
"""Wan2.2-TI2V video expert + backbone + loaders for the FastWAM MoT framework.

``WanVideoDiT`` is a plain ``nn.Module`` that directly implements the shared
:class:`~starVLA.model.modules.mot_model.mot.MoTExpertProtocol`.  It owns all Wan
weights (patch_embedding, blocks, head, text/time embeddings, 3D RoPE freqs)
and exposes ``prepare`` / ``mot_layer_qkv`` / ``mot_layer_post_attention`` /
``mot_finalize`` so the backbone-agnostic :class:`~starVLA.model.modules.mot_model.mot.MoT`
can couple it with the action expert without ever touching Wan block attributes.

``WanMoTBackbone`` owns the Wan-specific components (VAE, text encoder,
``WanVideoDiT``) and provides VAE encode/decode, text encode and first-frame
latent handling.  It is the sole owner of ``video_dit``; the framework accesses
it via ``backbone.video_dit``.

``load_wan22_components`` is the top-level entry point that loads the Wan2.2
diffusers transformer state dict, converts it to ``WanVideoDiT``, loads the
Wan VAE and (optionally) the UMT5 text encoder + tokenizer, and assembles a
:class:`WanMoTBackbone`.

Numeric parity target: the per-layer math mirrors the reference
``FastWAM/src/fastwam/models/wan22/wan_video_dit.py`` ``DiTBlock.forward`` /
``mot.py`` ``_build_expert_attention_io`` / ``_apply_expert_post_block``, split
into the two halves around the MoT mixed SDPA.  The only intentional reshape is
``[B, S, H*Dh]`` -> ``[B, H, S, Dh]`` at the protocol boundary (pure view, done
after norm+RoPE).
"""

from __future__ import annotations

import glob
import os
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from safetensors.torch import load_file as safetensors_load_file

from starVLA.model.modules.mot_model.dit import (
    DiTBlock,
    Head,
    modulate,
    precompute_freqs_cis_3d,
    rope_apply,
    sinusoidal_embedding_1d,
)
from starVLA.model.modules.mot_model.mot import (
    MoTExpertState,
    MoTLayerIO,
    MoTTokenLayout,
)
from starVLA.model.modules.text_embedding.transformers_t5 import (
    TransformersT5Encoder,
    TransformersTokenizer,
)
from starVLA.model.modules.vae.diffusers_wan_vae import DiffusersWanVAE
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


# ============================================================================
# State-dict I/O helpers
# ============================================================================


def load_state_dict(
    file_path: str | list[str], torch_dtype: Optional[torch.dtype] = None, device: str = "cpu"
) -> dict[str, torch.Tensor]:
    """Load a state dict from a safetensors / .bin / list-of-files checkpoint."""
    if isinstance(file_path, list):
        state_dict: dict[str, torch.Tensor] = {}
        for fp in file_path:
            state_dict.update(load_state_dict(fp, torch_dtype=torch_dtype, device=device))
        return state_dict
    if str(file_path).endswith(".safetensors"):
        return safetensors_load_file(file_path, device=device)
    state = torch.load(file_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if torch_dtype is not None:
        state = {k: v.to(torch_dtype) for k, v in state.items()}
    return state


# ============================================================================
# diffusers WanTransformer3DModel -> WanVideoDiT key mapping
# (verbatim from reference state_dict_converters.wan_video_dit_from_diffusers)
# ============================================================================


def wan_video_dit_from_diffusers(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    rename_dict = {
        "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
        "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
        "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
        "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
        "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
        "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
        "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
        "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
        "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
        "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
        "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
        "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
        "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
        "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
        "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
        "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
        "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
        "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
        "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
        "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
        "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
        "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
        "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
        "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
        "blocks.0.norm2.bias": "blocks.0.norm3.bias",
        "blocks.0.norm2.weight": "blocks.0.norm3.weight",
        "blocks.0.scale_shift_table": "blocks.0.modulation",
        "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
        "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
        "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
        "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
        "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
        "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
        "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
        "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
        "condition_embedder.time_proj.bias": "time_projection.1.bias",
        "condition_embedder.time_proj.weight": "time_projection.1.weight",
        "patch_embedding.bias": "patch_embedding.bias",
        "patch_embedding.weight": "patch_embedding.weight",
        "scale_shift_table": "head.modulation",
        "proj_out.bias": "head.head.bias",
        "proj_out.weight": "head.head.weight",
    }
    converted: dict[str, torch.Tensor] = {}
    for name, value in state_dict.items():
        if name in rename_dict:
            converted[rename_dict[name]] = value
        else:
            probe_name = ".".join([*name.split(".")[:1], "0", *name.split(".")[2:]])
            if probe_name in rename_dict:
                mapped = rename_dict[probe_name]
                mapped = ".".join([*mapped.split(".")[:1], name.split(".")[1], *mapped.split(".")[2:]])
                converted[mapped] = value
    return converted


def load_wan_video_dit_from_state_dict(
    video_dit,
    state_dict: dict[str, torch.Tensor],
    *,
    strict: bool = True,
    logger=None,
) -> dict[str, Any]:
    """Map a diffusers Wan state dict into ``video_dit`` (``WanVideoDiT``).

    Returns a report ``{loaded, missing, unexpected, source_format}``.
    The diffusers keys that have no WanVideoDiT counterpart (e.g. image-embedder
    for TI2V) are dropped silently — they are not part of the FastWAM MoT video
    expert path.
    """
    converted = wan_video_dit_from_diffusers(state_dict)
    current_keys = set(video_dit.state_dict().keys())
    converted_keys = set(converted.keys())

    # Only keep keys that actually exist on the target model; drop diffusers-only
    # keys (image_embedder / extra heads) with an informational note.
    applicable = {k: v for k, v in converted.items() if k in current_keys}
    dropped = sorted(converted_keys - current_keys)

    missing = sorted(current_keys - set(applicable.keys()))
    unexpected = sorted(set(applicable.keys()) - current_keys)  # always empty by construction

    video_dit.load_state_dict(applicable, strict=False)

    report = {
        "loaded": len(applicable),
        "missing": missing,
        "unexpected": unexpected,
        "dropped_diffusers_only": dropped,
        "source_format": "diffusers_wan_transformer_3d",
    }
    if logger is not None:
        logger.info(
            "Loaded Wan video DiT from diffusers state dict: loaded=%d missing=%d dropped=%d",
            report["loaded"],
            len(missing),
            len(dropped),
        )
    return report


class WanVideoDiT(nn.Module):
    """Wan2.2 video DiT expert implementing the MoT expert protocol.

    The block structure is identical to the reference ``WanVideoDiT``: a
    Conv3d patch embedding, a stack of shared :class:`DiTBlock`, a modulation
    ``Head`` + unpatchify.  Per-token (separated) timesteps are used: the first
    latent frame always carries ``t=0`` (clean) and the remaining frames carry
    the sampled timestep.
    """

    def __init__(
        self,
        hidden_dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        video_attention_mask_mode: str = "first_frame_causal",
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}")

        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.ffn_dim = ffn_dim
        self.out_dim = out_dim
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.eps = eps
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.num_layers = num_layers
        self.video_attention_mask_mode = str(video_attention_mask_mode)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

        self.attn_hidden_dim = self.num_heads * self.attn_head_dim

        self.patch_embedding = nn.Conv3d(in_dim, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, attn_head_dim, num_heads, ffn_dim, eps) for _ in range(num_layers)]
        )
        self.head = Head(hidden_dim, out_dim, patch_size, eps)
        self.freqs = precompute_freqs_cis_3d(attn_head_dim)

    # ------------------------------------------------------------------
    # MoTExpertProtocol: capability properties
    # ------------------------------------------------------------------
    @property
    def mot_num_layers(self) -> int:
        return self.num_layers

    @property
    def mot_num_heads(self) -> int:
        return self.num_heads

    @property
    def mot_head_dim(self) -> int:
        return self.attn_head_dim

    # ------------------------------------------------------------------
    # patchify / unpatchify (verbatim reference)
    # ------------------------------------------------------------------
    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        return self.patch_embedding(x)

    def unpatchify(self, x: torch.Tensor, grid_size: Tuple[int, int, int]) -> torch.Tensor:
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

    def build_video_to_video_mask(
        self,
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Per-frame video self-attention mask (matches reference)."""
        if video_seq_len <= 0:
            raise ValueError(f"`video_seq_len` must be positive, got {video_seq_len}")
        if video_tokens_per_frame <= 0:
            raise ValueError(f"`video_tokens_per_frame` must be positive, got {video_tokens_per_frame}")

        if self.video_attention_mask_mode == "bidirectional":
            return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)

        if self.video_attention_mask_mode == "per_frame_causal":
            if video_seq_len % video_tokens_per_frame != 0:
                raise ValueError(
                    "`video_seq_len` must be divisible by `video_tokens_per_frame` in "
                    f"`per_frame_causal` mode, got {video_seq_len} and {video_tokens_per_frame}"
                )
            num_video_frames = video_seq_len // video_tokens_per_frame
            frame_causal = torch.tril(torch.ones((num_video_frames, num_video_frames), dtype=torch.bool, device=device))
            return frame_causal.repeat_interleave(video_tokens_per_frame, dim=0).repeat_interleave(
                video_tokens_per_frame, dim=1
            )

        if self.video_attention_mask_mode == "first_frame_causal":
            video_mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
            first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
            video_mask[:first_frame_tokens, first_frame_tokens:] = False
            return video_mask

        raise ValueError(f"Unsupported video attention mask mode: {self.video_attention_mask_mode}")

    # ------------------------------------------------------------------
    # MoTExpertProtocol: prepare
    # ------------------------------------------------------------------
    def prepare(
        self,
        video_latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
        valid_mask: torch.Tensor,
        frame_ids: torch.Tensor,
    ) -> MoTExpertState:
        """Tokenize video latents into a MoT expert state.

        Args:
            video_latents: ``[B, in_dim, T, H, W]`` VAE latents.
            timestep: ``[B]`` (per-sample).  First latent frame is forced to
                ``t=0`` at the token level regardless (clean first-frame).
            context: text/proprio context ``[B, L, text_dim]``.
            context_mask: ``[B, L]`` bool.
            valid_mask: ``[B, S_v]`` bool over video tokens (excludes padding).
            frame_ids: ``[B, S_v]`` int64 latent-frame id per token (``0..T_lat-1``).
        """
        if video_latents.ndim != 5:
            raise ValueError(f"`video_latents` must be 5D [B, C, T, H, W], got {tuple(video_latents.shape)}")
        batch_size = video_latents.shape[0]
        patch_h = int(self.patch_size[1])
        patch_w = int(self.patch_size[2])
        if video_latents.shape[3] % patch_h != 0 or video_latents.shape[4] % patch_w != 0:
            raise ValueError(
                "Latent spatial shape must be divisible by DiT patch size, "
                f"got HxW=({video_latents.shape[3]}, {video_latents.shape[4]}), "
                f"patch=({patch_h}, {patch_w})"
            )
        tokens_per_frame = (video_latents.shape[3] // patch_h) * (video_latents.shape[4] // patch_w)

        x = self.patchify(video_latents)
        f, h, w = x.shape[2:]
        # Per-token (separated) timestep: first latent frame t=0, others = timestep.
        token_timesteps = torch.ones(
            (batch_size, x.shape[2], tokens_per_frame),
            dtype=timestep.dtype,
            device=timestep.device,
        ) * timestep.view(batch_size, 1, 1)
        token_timesteps[:, 0, :] = 0
        token_timesteps = token_timesteps.reshape(batch_size, -1)
        token_t_emb = sinusoidal_embedding_1d(self.freq_dim, token_timesteps.reshape(-1))
        t = self.time_embedding(token_t_emb).reshape(batch_size, -1, self.hidden_dim)  # [B,S,D]
        t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))  # [B,S,6,D]

        x_tokens = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        freqs = (
            torch.cat(
                [
                    self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
            .to(x_tokens.device)
        )

        if context_mask is None:
            context_mask = torch.ones((context.shape[0], context.shape[1]), dtype=torch.bool, device=context.device)
        # Cross-attention context mask expanded over the video token sequence.
        context_attn_mask = context_mask.unsqueeze(1).expand(-1, x_tokens.shape[1], -1)

        context_emb = self.text_embedding(context)
        layout = MoTTokenLayout(
            groups={"video": valid_mask.to(x_tokens.device)},
            frame_ids=frame_ids.to(x_tokens.device),
            valid_mask=valid_mask.to(x_tokens.device),
        )
        payload = {
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_attn_mask,
            "freqs": freqs,
            "grid_size": (f, h, w),
            "head_t_mod": t,  # [B,S,D] for Head.forward 3D path
            "tokens_per_frame": tokens_per_frame,
        }
        return MoTExpertState(tokens=x_tokens, layout=layout, payload=payload)

    # ------------------------------------------------------------------
    # MoTExpertProtocol: mot_layer_qkv
    # ------------------------------------------------------------------
    def mot_layer_qkv(self, layer_idx: int, state: MoTExpertState) -> MoTLayerIO:
        block = self.blocks[layer_idx]
        x = state.tokens
        t_mod = state.payload["t_mod"]
        freqs = state.payload["freqs"]

        # Mirror reference _build_expert_attention_io / DiTBlock.forward.
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            block.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            shift_mlp = shift_mlp.squeeze(2)
            scale_mlp = scale_mlp.squeeze(2)
            gate_mlp = gate_mlp.squeeze(2)
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)

        q = block.self_attn.norm_q(block.self_attn.q(attn_input))  # [B,S,H*Dh]
        k = block.self_attn.norm_k(block.self_attn.k(attn_input))
        v = block.self_attn.v(attn_input)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        # Reshape to [B,H,S,Dh] at the protocol boundary (pure view after RoPE).
        B, S = q.shape[0], q.shape[1]
        q = q.view(B, S, self.num_heads, self.attn_head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.view(B, S, self.num_heads, self.attn_head_dim).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, S, self.num_heads, self.attn_head_dim).permute(0, 2, 1, 3).contiguous()

        layer_payload = {
            "gate_msa": gate_msa,
            "shift_mlp": shift_mlp,
            "scale_mlp": scale_mlp,
            "gate_mlp": gate_mlp,
            "context": state.payload["context"],
            "context_mask": state.payload["context_mask"],
        }
        return MoTLayerIO(q=q, k=k, v=v, residual=x, payload=layer_payload)

    # ------------------------------------------------------------------
    # MoTExpertProtocol: mot_layer_post_attention
    # ------------------------------------------------------------------
    def mot_layer_post_attention(
        self,
        layer_idx: int,
        attention_output: torch.Tensor,
        layer_io: MoTLayerIO,
        state: MoTExpertState,
    ) -> MoTExpertState:
        block = self.blocks[layer_idx]
        # attention_output: [B,H,S,Dh] -> [B,S,H*Dh]
        B, H, S, Dh = attention_output.shape
        mixed_flat = attention_output.permute(0, 2, 1, 3).reshape(B, S, H * Dh)

        p = layer_io.payload
        x = layer_io.residual + p["gate_msa"] * block.self_attn.o(mixed_flat)
        # Unconditional cross-attention (P2): reference runs it whenever context
        # is present, which is always for FastWAM.
        x = x + block.cross_attn(block.norm3(x), p["context"], ctx_mask=p["context_mask"])
        mlp_input = modulate(block.norm2(x), p["shift_mlp"], p["scale_mlp"])
        x = x + p["gate_mlp"] * block.ffn(mlp_input)
        # Rebind tokens; layout and payload are read-only (kept by reference).
        return MoTExpertState(tokens=x, layout=state.layout, payload=state.payload)

    # ------------------------------------------------------------------
    # MoTExpertProtocol: mot_finalize
    # ------------------------------------------------------------------
    def mot_finalize(self, state: MoTExpertState) -> torch.Tensor:
        grid_size = state.payload["grid_size"]
        x = self.head(state.tokens, state.payload["head_t_mod"])
        x = self.unpatchify(x, grid_size)
        return x

    # ------------------------------------------------------------------
    # Standalone forward (kept for parity testing against the reference path;
    # the MoT does NOT call this — it calls the protocol methods above).
    # ------------------------------------------------------------------
    def forward(
        self,
        video_latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        frame_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if valid_mask is None:
            B, _, T, H, W = video_latents.shape
            patch_h, patch_w = int(self.patch_size[1]), int(self.patch_size[2])
            tokens_per_frame = (H // patch_h) * (W // patch_w)
            S = (T) * tokens_per_frame
            valid_mask = torch.ones((B, S), dtype=torch.bool, device=video_latents.device)
        if frame_ids is None:
            B, _, T, H, W = video_latents.shape
            patch_h, patch_w = int(self.patch_size[1]), int(self.patch_size[2])
            tokens_per_frame = (H // patch_h) * (W // patch_w)
            # latent-frame id per token: repeat each frame id across its spatial tokens.
            frame_ids = (
                torch.arange(T, device=video_latents.device).view(1, T, 1).expand(B, T, tokens_per_frame).reshape(B, -1)
            )
        state = self.prepare(video_latents, timestep, context, context_mask, valid_mask, frame_ids)
        self_attn_mask = (
            self.build_video_to_video_mask(
                video_seq_len=state.tokens.shape[1],
                video_tokens_per_frame=int(state.payload["tokens_per_frame"]),
                device=state.tokens.device,
            )
            if self.video_attention_mask_mode != "bidirectional"
            else None
        )
        for layer_idx in range(self.num_layers):
            io = self.mot_layer_qkv(layer_idx, state)
            from starVLA.model.modules.mot_model.dit import scaled_dot_product_attention

            mixed = scaled_dot_product_attention(io.q, io.k, io.v, num_heads=self.num_heads, ctx_mask=self_attn_mask)
            state = self.mot_layer_post_attention(layer_idx, mixed, io, state)
        return self.mot_finalize(state)


# ============================================================================
# WanMoTBackbone
# ============================================================================


class WanMoTBackbone(nn.Module):
    """Wan-specific components owned by the FastWAM framework.

    Holds the VAE (see :class:`~starVLA.model.modules.vae.build_vae`),
    the text encoder / tokenizer (see
    :func:`~starVLA.model.modules.text_embedding.build_text_encoder` /
    :func:`~starVLA.model.modules.text_embedding.build_tokenizer`, either of
    which may be ``None`` when precomputed context is supplied), and the
    :class:`WanVideoDiT` video expert.  Provides VAE encode/decode, text encode
    and first-frame latent handling.  It does NOT own the action scheduler or
    action loss.
    """

    def __init__(
        self,
        video_dit: WanVideoDiT,
        vae: Any,
        text_encoder: Any = None,
        tokenizer: Any = None,
    ):
        super().__init__()
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.video_dit = video_dit

        self.vae_scale_factor_spatial = self._require_vae_metadata(vae, "upsampling_factor")
        self.vae_scale_factor_temporal = self._require_vae_metadata(vae, "temporal_downsample_factor")
        self.vae_z_dim = self._require_vae_metadata(vae, "z_dim")
        if self.text_encoder is not None and self.tokenizer is None:
            raise ValueError("`tokenizer` is required when `text_encoder` is provided.")

    @staticmethod
    def _require_vae_metadata(vae: Any, name: str) -> int:
        if not hasattr(vae, name):
            raise TypeError(f"VAE {type(vae).__name__} must define integer `{name}` metadata.")
        value = int(getattr(vae, name))
        if value <= 0:
            raise ValueError(f"VAE `{name}` must be positive, got {value}.")
        return value

    @property
    def text_dim(self) -> int:
        return self.video_dit.text_dim

    @torch.no_grad()
    def encode_video_latents(
        self,
        video_tensor: torch.Tensor,
        device: torch.device,
        tiled: bool = False,
        tile_size: Tuple[int, int] = (30, 52),
        tile_stride: Tuple[int, int] = (15, 26),
    ) -> torch.Tensor:
        return self.vae.encode(
            video_tensor,
            device=device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )

    @torch.no_grad()
    def encode_first_frame_latents(
        self,
        input_image: torch.Tensor,
        device: torch.device,
        tiled: bool = False,
        tile_size: Tuple[int, int] = (30, 52),
        tile_stride: Tuple[int, int] = (15, 26),
    ) -> torch.Tensor:
        """Encode a single conditioning image ``[1,3,H,W]`` (or ``[3,H,W]``).

        Returns ``[1, z_dim, 1, H/16, W/16]`` (a single latent frame).
        """
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        image = input_image.to(device)[0].unsqueeze(1)  # [3,1,H,W]
        z = self.vae.encode([image], device=device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    @torch.no_grad()
    def decode_latents(
        self,
        latents: torch.Tensor,
        device: torch.device,
        tiled: bool = False,
        tile_size: Tuple[int, int] = (30, 52),
        tile_stride: Tuple[int, int] = (15, 26),
    ) -> torch.Tensor:
        return self.vae.decode(latents, device=device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)

    @torch.no_grad()
    def encode_prompt(self, prompt, device: torch.device, dtype: torch.dtype):
        """Encode a text prompt into context ``[B, L, text_dim]`` + mask ``[B, L]``."""
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set text.source=encoder or provide precomputed context/context_mask."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(device)
        mask = mask.to(device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        # Match reference: zero out padding positions so cross-attn sees clean context.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=device, dtype=dtype), mask

    def latent_grid(self, num_frames: int, height: int, width: int) -> Tuple[int, int, int]:
        """Return ``(latent_t, latent_h, latent_w)`` for a pixel video shape."""
        latent_t = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_h = height // self.vae_scale_factor_spatial
        latent_w = width // self.vae_scale_factor_spatial
        return latent_t, latent_h, latent_w


# ============================================================================
# Top-level component loader
# ============================================================================


def _resolve_file(local_dir: str, patterns: list[str]) -> Optional[str | list[str]]:
    """Find a file under ``local_dir`` matching any of ``patterns`` (glob)."""
    matches: list[str] = []
    for pat in patterns:
        matches.extend(glob.glob(os.path.join(local_dir, pat)))
    matches = sorted(set(matches))
    if len(matches) == 0:
        return None
    if len(matches) == 1:
        return matches[0]
    return matches


def _maybe_download(model_id: Optional[str], allow_patterns: list[str]) -> Optional[str]:
    """Best-effort HF Hub download; returns local dir or None."""
    if model_id is None:
        return None
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.warning(
            "huggingface_hub not installed; cannot download %s. Provide a local pretrained_path.",
            model_id,
        )
        return None
    try:
        local_dir = snapshot_download(repo_id=model_id, allow_patterns=allow_patterns)
        return local_dir
    except Exception as e:
        logger.warning("HF Hub download failed for %s: %s", model_id, e)
        return None


def load_wan22_components(
    *,
    video_dit_config: dict[str, Any],
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    device: str = "cpu",
    torch_dtype: torch.dtype = torch.float32,
    init_from_config_only: bool = False,
) -> WanMoTBackbone:
    """Build the fixed Wan2.2 components used by Ego4WAM.

    ``model_id`` is an operational asset location, not an architecture option.
    The video expert, VAE, text encoder and tokenizer families are fixed.
    """
    if video_dit_config is None:
        raise ValueError("video_dit_config is required")

    if init_from_config_only:
        with torch.device("meta"):
            video_dit = WanVideoDiT(**video_dit_config)
        video_dit = video_dit.to(dtype=torch_dtype)
    else:
        video_dit = WanVideoDiT(**video_dit_config).to(device=device, dtype=torch_dtype)
        base_dir = _maybe_download(
            model_id,
            allow_patterns=[
                "transformer/config.json",
                "transformer/diffusion_pytorch_model*.safetensors",
                "transformer/diffusion_pytorch_model*.bin",
            ],
        )
        if base_dir is None:
            raise FileNotFoundError(f"Could not resolve Ego4WAM Wan assets from {model_id!r}.")
        dit_files = _resolve_file(
            base_dir,
            [
                "transformer/diffusion_pytorch_model*.safetensors",
                "transformer/diffusion_pytorch_model*.bin",
                "diffusion_pytorch_model*.safetensors",
                "diffusion_pytorch_model*.bin",
            ],
        )
        if dit_files is None:
            raise FileNotFoundError(f"Could not find the Wan transformer under {base_dir!r}.")
        state_dict = load_state_dict(dit_files, torch_dtype=torch_dtype, device="cpu")
        checkpoint_keys = set(state_dict)
        native_format = any("self_attn." in key for key in checkpoint_keys) and any(
            key.endswith(".modulation") for key in checkpoint_keys
        )
        if native_format:
            missing, unexpected = video_dit.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    "Wan video expert checkpoint mismatch: "
                    f"missing={list(missing)[:10]}, unexpected={list(unexpected)[:10]}"
                )
        else:
            report = load_wan_video_dit_from_state_dict(
                video_dit, state_dict, strict=True, logger=logger
            )
            if report["missing"] or report["unexpected"]:
                raise RuntimeError(
                    "Converted Wan video expert checkpoint mismatch: "
                    f"missing={report['missing'][:10]}, unexpected={report['unexpected'][:10]}"
                )
        video_dit = video_dit.to(device=device, dtype=torch_dtype)

    vae = DiffusersWanVAE(
        z_dim=int(video_dit_config["in_dim"]),
        dim=160,
        pretrained_path=model_id,
        torch_dtype=torch_dtype,
        init_from_config_only=init_from_config_only,
    )
    text_encoder = TransformersT5Encoder(
        pretrained_model_name_or_path=model_id,
        subfolder="text_encoder",
        torch_dtype=torch_dtype,
        device=device,
        init_from_config_only=init_from_config_only,
    )
    tokenizer = TransformersTokenizer(
        pretrained_model_name_or_path=model_id,
        subfolder="tokenizer",
        seq_len=128,
    )
    if init_from_config_only:
        vae = vae.to(dtype=torch_dtype)
    else:
        vae = vae.to(device=device, dtype=torch_dtype)
    vae = vae.eval().requires_grad_(False)

    return WanMoTBackbone(
        video_dit=video_dit,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )


__all__ = [
    "WanMoTBackbone",
    "WanVideoDiT",
    "load_state_dict",
    "load_wan22_components",
    "load_wan_video_dit_from_state_dict",
    "wan_video_dit_from_diffusers",
]
