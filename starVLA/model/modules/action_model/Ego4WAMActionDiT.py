# Copyright (c) 2026 The FastWAM Authors.
# Modifications Copyright (c) 2026 Beijing Horizon Information Technology Company Limited.
# Author: Ego4WAM Team.
# SPDX-License-Identifier: Apache-2.0 AND MIT
# Ego4WAM additions/modifications: Apache-2.0; upstream FastWAM portions: MIT.
# See LICENSES/FastWAM-MIT.txt for the retained upstream license.
"""Ego4WAM action expert (ActionDiT) + init-payload loader.

``Ego4WAMActionDiT`` is a plain ``nn.Module`` that implements the shared
:class:`~starVLA.model.modules.mot_model.mot.MoTExpertProtocol`, so the backbone-agnostic
MoT can couple it with the Wan video expert without touching action-specific
attributes.  Its block structure is identical to the reference ``ActionDiT``: an
``action_encoder`` Linear, a ``text_embedding`` (Linear-GELU-Linear), time
embedding / time-projection, a stack of shared :class:`DiTBlock`, and a plain
``Linear`` head (the reference's ``ActionHead`` modulation head is unused in the
runtime path and not ported).

``build_ego4wam_action_dit`` is the convenience constructor that assembles an expert
and optionally loads an offline Wan→ActionDiT init payload.

Numeric parity target: per-layer math mirrors the reference
``FastWAM/src/fastwam/models/wan22/action_dit.py`` ``DiTBlock.forward`` / the
``mot.py`` QKV / post-attention helpers.  The action branch uses 1D RoPE and a
3D ``t_mod`` (``has_seq=False``), so the modulation chunking is the
``chunk_dim=1`` path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from starVLA.model.modules.mot_model.dit import (
    DiTBlock,
    modulate,
    precompute_freqs_cis,
    rope_apply,
    sinusoidal_embedding_1d,
)
from starVLA.model.modules.mot_model.mot import MoTExpertState, MoTLayerIO, MoTTokenLayout
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# Keys that are randomly initialised (NOT transferred from the Wan video expert)
# and therefore excluded from the offline Wan->ActionDiT init payload.
ACTION_BACKBONE_SKIP_PREFIXES = ("action_encoder.", "head.")
ACTION_BACKBONE_META_KEYS = (
    "hidden_dim",
    "ffn_dim",
    "num_layers",
    "num_heads",
    "attn_head_dim",
    "text_dim",
    "freq_dim",
    "eps",
)


class Ego4WAMActionDiT(nn.Module):
    """Action diffusion transformer expert implementing the MoT expert protocol."""

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        ffn_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.ffn_dim = ffn_dim
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.num_layers = num_layers
        self.eps = eps
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}")

        self.attn_hidden_dim = self.num_heads * self.attn_head_dim

        self.action_encoder = nn.Linear(action_dim, hidden_dim)
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
        self.head = nn.Linear(hidden_dim, action_dim)
        self.freqs = precompute_freqs_cis(attn_head_dim, end=1024)

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

    @classmethod
    def backbone_key_set(cls, keys) -> set[str]:
        """Keys that are transferred from the Wan video expert (excludes
        ``action_encoder.*`` / ``head.*``)."""
        return {key for key in keys if not any(key.startswith(prefix) for prefix in ACTION_BACKBONE_SKIP_PREFIXES)}

    # ------------------------------------------------------------------
    # MoTExpertProtocol: prepare
    # ------------------------------------------------------------------
    def prepare(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None,
    ) -> MoTExpertState:
        """Tokenize raw action tokens into a MoT expert state.

        Args:
            action_tokens: ``[B, S, action_dim]`` (noised action latents).
            timestep: ``[B]`` per-sample timestep.
            context: cross-attention context ``[B, L, text_dim]`` (text/proprio).
            context_mask: ``[B, L]`` bool.
            valid_mask: ``[B, S]`` bool over action tokens (excludes padding).
        """
        if action_tokens.ndim != 3:
            raise ValueError(f"`action_tokens` must be 3D [B, S, action_dim], got {tuple(action_tokens.shape)}")
        if action_tokens.shape[2] != self.action_dim:
            raise ValueError(f"`action_tokens` last dim must be {self.action_dim}, got {action_tokens.shape[2]}")
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B], got {tuple(timestep.shape)}")
        if context.ndim != 3:
            raise ValueError(f"`context` must be 3D [B, L, D], got {tuple(context.shape)}")
        batch_size = action_tokens.shape[0]
        if context.shape[0] != batch_size:
            raise ValueError(f"Batch mismatch action vs context: {batch_size} vs {context.shape[0]}")
        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}")
        if timestep.shape[0] == 1 and batch_size > 1:
            if self.training:
                raise ValueError("During training, action timestep length must match batch_size.")
            timestep = timestep.expand(batch_size)

        if context_mask is None:
            context_mask = torch.ones((batch_size, context.shape[1]), dtype=torch.bool, device=context.device)
        else:
            if context_mask.ndim != 2 or context_mask.shape != (batch_size, context.shape[1]):
                got = tuple(context_mask.shape)
                raise ValueError(
                    f"`context_mask` must be [B, L] matching context, got {got} " f"vs {(batch_size, context.shape[1])}"
                )

        seq_len = action_tokens.shape[1]
        if seq_len > self.freqs.shape[0]:
            raise ValueError(f"Action token length {seq_len} exceeds RoPE cache {self.freqs.shape[0]}.")

        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))  # [B, D]
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))  # [B, 6, D]
        tokens = self.action_encoder(action_tokens)  # [B, S, D]
        context_emb = self.text_embedding(context)  # [B, L, D]
        context_attn_mask = context_mask.unsqueeze(1).expand(-1, seq_len, -1)  # [B, S, L]
        freqs = self.freqs[:seq_len].view(seq_len, 1, -1).to(tokens.device)  # [S, 1, rope_dim]

        if valid_mask is None:
            valid_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=tokens.device)
        layout = MoTTokenLayout(
            groups={"action": valid_mask.to(tokens.device)},
            frame_ids=None,
            valid_mask=valid_mask.to(tokens.device),
        )
        payload = {
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_attn_mask,
            "freqs": freqs,
        }
        return MoTExpertState(tokens=tokens, layout=layout, payload=payload)

    # ------------------------------------------------------------------
    # MoTExpertProtocol: mot_layer_qkv
    # ------------------------------------------------------------------
    def mot_layer_qkv(self, layer_idx: int, state: MoTExpertState) -> MoTLayerIO:
        block = self.blocks[layer_idx]
        x = state.tokens
        t_mod = state.payload["t_mod"]
        freqs = state.payload["freqs"]

        # 3D t_mod -> has_seq=False -> chunk_dim=1 (matches reference ActionDiT).
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            block.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=1)
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)

        q = block.self_attn.norm_q(block.self_attn.q(attn_input))  # [B,S,H*Dh]
        k = block.self_attn.norm_k(block.self_attn.k(attn_input))
        v = block.self_attn.v(attn_input)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
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
        B, H, S, Dh = attention_output.shape
        mixed_flat = attention_output.permute(0, 2, 1, 3).reshape(B, S, H * Dh)

        p = layer_io.payload
        x = layer_io.residual + p["gate_msa"] * block.self_attn.o(mixed_flat)
        x = x + block.cross_attn(block.norm3(x), p["context"], ctx_mask=p["context_mask"])
        mlp_input = modulate(block.norm2(x), p["shift_mlp"], p["scale_mlp"])
        x = x + p["gate_mlp"] * block.ffn(mlp_input)
        return MoTExpertState(tokens=x, layout=state.layout, payload=state.payload)

    # ------------------------------------------------------------------
    # MoTExpertProtocol: mot_finalize
    # ------------------------------------------------------------------
    def mot_finalize(self, state: MoTExpertState) -> torch.Tensor:
        return self.head(state.tokens)  # [B, S, action_dim]

    # ------------------------------------------------------------------
    # Standalone forward (kept for parity testing; MoT does NOT call this).
    # ------------------------------------------------------------------
    def forward(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        state = self.prepare(action_tokens, timestep, context, context_mask, valid_mask)
        for layer_idx in range(self.num_layers):
            io = self.mot_layer_qkv(layer_idx, state)
            from starVLA.model.modules.mot_model.dit import scaled_dot_product_attention

            mixed = scaled_dot_product_attention(io.q, io.k, io.v, num_heads=self.num_heads, ctx_mask=None)
            state = self.mot_layer_post_attention(layer_idx, mixed, io, state)
        return self.mot_finalize(state)

    # ------------------------------------------------------------------
    # Init payload loader (offline Wan->ActionDiT interpolation, strict load)
    # ------------------------------------------------------------------
    def load_init_payload(self, payload_path: str) -> Dict[str, Any]:
        """Strictly load an offline ActionDiT init payload into this expert.

        The payload produced by the checkpoint preparation tools carries the
        interpolated Wan-video-expert backbone weights for the
        action DiT; ``action_encoder.*`` / ``head.*`` are left random.  This
        performs NO interpolation at runtime — it only validates metadata and
        strictly loads, mirroring the reference ``ActionDiT.from_pretrained``.
        """
        p = Path(payload_path)
        if not p.is_absolute():
            p = Path.cwd() / p
        if not p.is_file():
            raise FileNotFoundError(f"ActionDiT init payload does not exist: {p}")

        # Payload is a trusted local artifact containing only tensors + plain
        # dicts, so weights_only=True is safe and silences the torch.load
        # FutureWarning.
        payload = torch.load(str(p), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid ActionDiT payload type from {p}: {type(payload)}")

        meta = payload.get("meta")
        if not isinstance(meta, dict):
            raise ValueError(f"ActionDiT payload `meta` must be a dict in {p}")
        expected_meta = {
            "hidden_dim": int(self.hidden_dim),
            "ffn_dim": int(self.ffn_dim),
            "num_layers": int(self.num_layers),
            "num_heads": int(self.num_heads),
            "attn_head_dim": int(self.attn_head_dim),
            "text_dim": int(self.text_dim),
            "freq_dim": int(self.freq_dim),
            "eps": float(self.eps),
        }
        for key in ACTION_BACKBONE_META_KEYS:
            if key not in meta:
                raise ValueError(f"`meta.{key}` missing in {p}")
            expected_value = expected_meta[key]
            got_value = meta[key]
            if key == "eps":
                if abs(float(got_value) - float(expected_value)) > 1e-12:
                    raise ValueError(f"`meta.{key}` mismatch in {p}: expected {expected_value}, got {got_value}")
            elif int(got_value) != int(expected_value):
                raise ValueError(f"`meta.{key}` mismatch in {p}: expected {expected_value}, got {got_value}")

        # Reject payloads prepared from a different backbone family.
        source_family = payload.get("source_backbone_family")
        if source_family is not None and source_family != "wan22":
            raise ValueError(
                f"ActionDiT payload source_backbone_family={source_family!r} in {p} is not "
                "'wan22'; this Wan2.2-paired ActionDiT only accepts wan22 payloads."
            )

        backbone_state_dict = payload.get("backbone_state_dict")
        if not isinstance(backbone_state_dict, dict):
            raise ValueError(f"`backbone_state_dict` must be a dict in {p}, got {type(backbone_state_dict)}")

        current_state = self.state_dict()
        expected_backbone_keys = self.backbone_key_set(current_state.keys())
        provided_keys = set(backbone_state_dict.keys())
        missing_keys = sorted(expected_backbone_keys - provided_keys)
        unexpected_keys = sorted(provided_keys - expected_backbone_keys)
        if missing_keys or unexpected_keys:
            raise ValueError(
                "ActionDiT backbone key mismatch in payload. "
                f"missing={missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}, "
                f"unexpected={unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}"
            )

        merged_state = dict(current_state)
        for key in expected_backbone_keys:
            value = backbone_state_dict[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"`backbone_state_dict[{key}]` must be torch.Tensor in {p}, got {type(value)}")
            target = merged_state[key]
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"Shape mismatch for `{key}` in {p}: expected {tuple(target.shape)}, " f"got {tuple(value.shape)}"
                )
            merged_state[key] = value.to(device=target.device, dtype=target.dtype)

        self.load_state_dict(merged_state, strict=True)
        return {
            "policy": payload.get("policy", {}),
            "meta": meta,
            "source_backbone_family": source_family,
            "loaded_keys": len(expected_backbone_keys),
        }


__all__ = [
    "ACTION_BACKBONE_META_KEYS",
    "ACTION_BACKBONE_SKIP_PREFIXES",
    "Ego4WAMActionDiT",
    "build_ego4wam_action_dit",
]


# ============================================================================
# Convenience constructor
# ============================================================================


def build_ego4wam_action_dit(
    *,
    action_dit_config: dict[str, Any],
    init_payload_path: str | None = None,
    device: str = "cpu",
    torch_dtype: torch.dtype = torch.float32,
    init_from_config_only: bool = False,
) -> Ego4WAMActionDiT:
    """Construct an action expert, optionally loading an init payload.

    Args:
        action_dit_config: kwargs for :class:`Ego4WAMActionDiT`.
        init_payload_path: optional path to the offline Wan->ActionDiT init
            payload produced by the checkpoint preparation tools. If None, the
            expert is randomly initialised and may be overridden by an Ego4WAM
            checkpoint later.
    """
    # Build on CPU and load the payload on CPU first, then move to the target
    # device: loading a payload when the model is already on GPU doubles peak
    # VRAM (model + payload tensors both on device), which OOMs alongside a
    # resident 5B video expert. Loading on CPU then `.to(device)` is a single
    # residency move.
    if init_from_config_only:
        if init_payload_path is not None:
            raise ValueError("Config-only ActionDiT construction cannot load an init payload.")
        with torch.device("meta"):
            expert = Ego4WAMActionDiT(**action_dit_config)
        return expert.to(dtype=torch_dtype)

    expert = Ego4WAMActionDiT(**action_dit_config)
    if init_payload_path is None:
        logger.info("No init payload; initialising Ego4WAMActionDiT with random weights.")
        return expert.to(device=device, dtype=torch_dtype)
    report = expert.load_init_payload(init_payload_path)
    logger.info(
        "Loaded ActionDiT init payload from %s (loaded_keys=%d, source_backbone_family=%s).",
        init_payload_path,
        report["loaded_keys"],
        report.get("source_backbone_family"),
    )
    return expert.to(device=device, dtype=torch_dtype)
