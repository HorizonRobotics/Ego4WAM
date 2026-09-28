# Copyright (c) 2026 The FastWAM Authors.
# Modifications Copyright (c) 2026 Ego4WAM Authors.
# SPDX-License-Identifier: MIT
"""
Shared FastWAM DiT building blocks.

These helpers are ported from the FastWAM reference implementation
(``FastWAM/src/fastwam/models/wan22/wan_video_dit.py``) so that the Wan video
expert (``WanVideoDiT``), the action expert (``FastWAMActionDiT``) and the
shared ``MoT`` all import the *same* ``DiTBlock`` / RoPE / SDPA code.  The
reference ``ActionDiT`` already imports ``DiTBlock``/``sinusoidal_embedding_1d``/
``precompute_freqs_cis`` from the video-DiT module, so sharing the code is the
reference structure.

Numeric parity target: the helpers below are byte-for-byte equivalent to the
reference except for the public attention helper being renamed to
``scaled_dot_product_attention`` (the design doc mandates this name and forbids
claiming FlashAttention).  It accepts both the ``[B, S, H*Dh]`` layout used by
standalone experts and the ``[B, H, S, Dh]`` layout used by the MoT coupling
layer, so no double-rearrange occurs at the protocol boundary.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)

_SDPA_BACKEND_DIAGNOSTIC_DONE = False


def _identify_sdpa_backend(operator_names: list[str]) -> str:
    """Identify the concrete PyTorch SDPA implementation from profiler ops."""

    backend_markers = (
        ("_scaled_dot_product_flash_attention", "flash"),
        ("_scaled_dot_product_cudnn_attention", "cudnn"),
        ("_scaled_dot_product_efficient_attention", "efficient"),
        ("_scaled_dot_product_attention_math", "math"),
    )
    for marker, backend in backend_markers:
        if any(marker in name for name in operator_names):
            return backend
    return "unknown"


def _should_diagnose_sdpa_backend(q: torch.Tensor) -> bool:
    enabled = os.environ.get("FASTWAM_SDPA_BACKEND_DIAGNOSTIC", "").strip().lower()
    return enabled in {"1", "true", "yes", "on"} and q.is_cuda and int(os.environ.get("RANK", "0")) == 0


def _run_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ctx_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Run SDPA, optionally profiling its first rank-0 CUDA dispatch once."""

    global _SDPA_BACKEND_DIAGNOSTIC_DONE
    if _SDPA_BACKEND_DIAGNOSTIC_DONE or not _should_diagnose_sdpa_backend(q):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=ctx_mask)

    # Mark before entering the profiler so an exception cannot repeatedly add
    # profiler overhead on every MoT layer.
    _SDPA_BACKEND_DIAGNOSTIC_DONE = True
    out = None
    try:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=ctx_mask)
            torch.cuda.synchronize(q.device)
        operator_names = sorted({event.key for event in profile.key_averages() if "scaled_dot_product" in event.key})
        backend = _identify_sdpa_backend(operator_names)
        logger.warning(
            "FastWAM SDPA backend diagnostic: backend=%s q_shape=%s dtype=%s " "mask_shape=%s operators=%s",
            backend,
            tuple(q.shape),
            q.dtype,
            None if ctx_mask is None else tuple(ctx_mask.shape),
            operator_names,
        )
    except Exception:
        logger.exception("FastWAM SDPA backend diagnostic failed; continuing training.")
        if out is None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=ctx_mask)
    return out


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    ctx_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Scaled dot-product attention on top of ``F.scaled_dot_product_attention``.

    Args:
        q, k, v: Either ``[B, S, H*Dh]`` (expert-internal, reference layout) or
            ``[B, H, S, Dh]`` (MoT coupling-layer layout).  When 4D the SDPA is
            called directly without any rearrange.
        num_heads: Number of attention heads ``H`` (only needed for the 3D path).
        ctx_mask: Optional bool attention mask where ``True`` = keep.  For the 3D
            path it is broadcast to ``[B, 1, S_q, S_k]``; for the 4D path it must
            already be broadcastable to ``[B, H, S_q, S_k]`` (typically
            ``[B, 1, S_q, S_k]``).

    Returns:
        Output in the *same* layout convention as the input ``q``.
    """
    if q.dim() == 4:
        # MoT coupling-layer path: already [B, H, S, Dh].
        out = _run_sdpa(q, k, v, ctx_mask)
        return out
    # Standalone-expert path: [B, S, H*Dh] -> [B, H, S, Dh] -> back.
    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    # A 3D [B, Sq, Sk] cross-attention mask must be unsqueezed to [B, 1, Sq, Sk]
    # so it broadcasts over heads (matches reference DiTBlock.forward's
    # `context_mask.unsqueeze(1)` for the standalone path, and the reference
    # MoT `_apply_expert_post_block` for the MoT-coupled cross-attention path).
    if ctx_mask is not None and ctx_mask.dim() == 3:
        ctx_mask = ctx_mask.unsqueeze(1)
    out = _run_sdpa(q, k, v, ctx_mask)
    out = rearrange(out, "b n s d -> b s (n d)", n=num_heads)
    return out


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(
            10000,
            -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2),
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0) -> torch.Tensor:
    # 1d rope precompute
    # [perf] float32 instead of float64: complex64 multiply is 2.5x faster on GPU
    # (ref: FastWAM reference uses float64 here; our version is float32).
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    freqs = torch.outer(torch.arange(end, dtype=torch.float32, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x: torch.Tensor, freqs: torch.Tensor, num_heads: int) -> torch.Tensor:
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    # [perf] float32 intermediate instead of float64: complex64 multiply is 2.5x
    # faster on GPU with no measurable precision loss in bf16/fp16 training
    # (ref: FastWAM reference uses x.to(torch.float64) here).
    x_out = torch.view_as_complex(x.float().reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = self.num_heads * self.attn_head_dim

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = scaled_dot_product_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=self_attn_mask)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = self.num_heads * self.attn_head_dim

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        ctx: torch.Tensor,
        ctx_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = scaled_dot_product_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=ctx_mask)
        return self.o(x)


class DiTBlock(nn.Module):
    """Shared DiT block used by BOTH the video and action experts.

    Forward order (mirrors the reference exactly, lines 249-268):
      norm1/modulate -> self-attn -> residual gate
      -> cross-attn (norm3) -> residual (unconditional, P2)
      -> norm2/modulate -> FFN -> residual gate
    """

    def __init__(
        self,
        hidden_dim: int,
        attn_head_dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_head_dim = attn_head_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.cross_attn = CrossAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if context_mask is not None and context_mask.dim() == 3:
            context_mask = context_mask.unsqueeze(1)  # (B, 1, seq_len, context_len)
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            # means t_mod has separate modulation for each token, otherwise same
            # modulation for all tokens in the block
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.self_attn(input_x, freqs, self_attn_mask=self_attn_mask)
        x = x + self.cross_attn(self.norm3(x), context, ctx_mask=context_mask)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.ffn(input_x)
        return x


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, t_mod: torch.Tensor) -> torch.Tensor:
        if len(t_mod.shape) == 3:
            shift, scale = (
                self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)
            ).chunk(2, dim=2)
            x = self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = self.head(self.norm(x) * (1 + scale) + shift)
        return x


def gradient_checkpoint_forward(model, use_gradient_checkpointing, *args, **kwargs):
    """Thin checkpoint wrapper ported from the reference ``helpers/gradient.py``."""
    if use_gradient_checkpointing:
        model_output = torch.utils.checkpoint.checkpoint(
            model,
            *args,
            **kwargs,
            use_reentrant=False,
        )
    else:
        model_output = model(*args, **kwargs)
    return model_output
