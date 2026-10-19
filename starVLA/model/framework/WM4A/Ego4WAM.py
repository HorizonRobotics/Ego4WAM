# Copyright (c) 2026 The FastWAM Authors.
# Modifications Copyright (c) 2026 Beijing Horizon Information Technology Company Limited.
# Author: Ego4WAM Team.
# SPDX-License-Identifier: MIT
# Ego4WAM additions/modifications: MIT; upstream FastWAM portions: MIT.
# See LICENSES/FastWAM-MIT.txt for the retained upstream license.
"""Ego4WAM framework — state-conditioned action expert with a shared MoT.

The framework couples a Wan video expert and an action expert layer by layer,
while exposing proprioceptive state only to the action expert. The video expert
is conditioned on text alone.

Design doc (§4-§8) constraints honoured here:
  - Single inheritance chain ``PreTrainedModel -> baseframework -> Ego4WAMFramework``.
  - Framework is the sole owner of ``backbone`` / ``action_model`` / ``mot``;
    MoT holds no trainable weights and is not a duplicate owner of experts.
  - Base cache and Joint synchronous inference are separate methods
    (``_predict_action_base`` / ``_predict_action_joint``); Joint never uses cache.
  - ``forward`` returns a single scalar ``action_loss`` (the weighted total) plus
    ``loss_metrics`` (floats, filtered by ``baseframework.compute_loss``).
  - ``predict_action`` uses the checkpoint's fixed 20-step Euler sampler.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.Ego4WAMActionDiT import build_ego4wam_action_dit
from starVLA.model.modules.mot_model.mot import (
    InteractionRule,
    MoT,
    MoTBranchBinding,
    compile_attention_mask,
)
from starVLA.model.modules.scheduler.wan_scheduler import WanContinuousFlowMatchScheduler
from starVLA.model.modules.world_model.WanMoT import load_wan22_components
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


# ---------------------------------------------------------------------------
# Fixed model contract
# ---------------------------------------------------------------------------

_WAN_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
_ACTION_DIM = 32
_ACTION_HORIZON = 32
_STATE_DIM = 32
_NUM_VIDEO_FRAMES = 9
_VIDEO_STRIDE = 4
_NUM_INFERENCE_STEPS = 20
_NUM_TRAIN_TIMESTEPS = 1000
_FLOW_SHIFT = 5.0
_PROMPT_TEMPLATE = (
    "An ARX X5A robot is executing the following instruction: {lang}. "
    "The video is composed of three camera views. The upper view is captured "
    "by a head camera, while the two lower views are provided by the wrist cameras."
)

_VIDEO_DIT_CONFIG = {
    "hidden_dim": 3072,
    "in_dim": 48,
    "ffn_dim": 14336,
    "out_dim": 48,
    "text_dim": 4096,
    "freq_dim": 256,
    "eps": 1.0e-6,
    "patch_size": (1, 2, 2),
    "num_heads": 24,
    "attn_head_dim": 128,
    "num_layers": 30,
    "video_attention_mask_mode": "first_frame_causal",
    "use_gradient_checkpointing": False,
}

_BASE_INTERACTION: tuple[InteractionRule, ...] = (
    InteractionRule("video.video", "video.video", "first_frame_causal"),
    InteractionRule("action.action", "video.video", "key_first_frame"),
    InteractionRule("action.action", "action.action", "full"),
)

_JOINT_INTERACTION: tuple[InteractionRule, ...] = (
    InteractionRule("video.video", "video.video", "first_frame_causal"),
    InteractionRule("action.action", "video.video", "full"),
    InteractionRule("action.action", "action.action", "full"),
)


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass
class _MotOutput:
    """Intermediate tensors from a single MoT forward pass.

    ``pred_video`` and ``target_video`` are future-only slices when
    ``first_frame_latents`` is not None (the Base interaction path).
    """
    input_latents: torch.Tensor
    first_frame_latents: Optional[torch.Tensor]
    pred_video: torch.Tensor
    pred_action: torch.Tensor
    target_video: torch.Tensor
    target_action: torch.Tensor
    t_video: torch.Tensor
    t_action: torch.Tensor
    latent_frame_pad: Optional[torch.Tensor] = None


def _resolve_interaction_policy(interaction_mode: str) -> list[InteractionRule]:
    if interaction_mode == "base":
        return list(_BASE_INTERACTION)
    if interaction_mode == "joint":
        return list(_JOINT_INTERACTION)
    raise ValueError(
        "framework.interaction_mode must be 'base' or 'joint', "
        f"got {interaction_mode!r}"
    )


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------


@dataclass
class Ego4WAMDefaultConfig:
    """The complete public Ego4WAM model configuration."""

    name: str = "Ego4WAM"
    interaction_mode: Optional[str] = None


@FRAMEWORK_REGISTRY.register("Ego4WAM")
class Ego4WAMFramework(baseframework):
    """Ego4WAM with state conditioning isolated to the action expert."""

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__()
        incoming_framework = getattr(config, "framework", {})
        incoming_keys = set(incoming_framework.keys()) if hasattr(incoming_framework, "keys") else set()
        unsupported = incoming_keys - {"name", "interaction_mode"}
        if unsupported:
            raise ValueError(
                "Ego4WAM only accepts framework.name and framework.interaction_mode; "
                f"unsupported keys: {sorted(unsupported)}"
            )
        self.config = merge_framework_config(Ego4WAMDefaultConfig, config)
        fw = self.config.framework

        checkpoint_init = self.config.get("_checkpoint_init", {})
        checkpoint_asset_root = None
        if checkpoint_init:
            artifact_dir = Path(str(checkpoint_init["artifact_dir"]))
            candidate = artifact_dir / "model_assets"
            if not candidate.is_dir():
                raise FileNotFoundError(
                    "Ego4WAM checkpoint is missing bundled model assets: "
                    f"{candidate}"
                )
            checkpoint_asset_root = str(candidate)
        init_from_config_only = checkpoint_asset_root is not None

        self.interaction_mode = str(fw.interaction_mode)
        self.mot_rules = _resolve_interaction_policy(self.interaction_mode)

        # These are part of the released architecture, not configuration axes.
        self.action_dim = _ACTION_DIM
        self.action_horizon = _ACTION_HORIZON
        self.num_video_frames = _NUM_VIDEO_FRAMES
        self.video_stride = _VIDEO_STRIDE
        self.history_delta_indices = ()
        self.num_history_frames = 0
        self.prompt_template = _PROMPT_TEMPLATE

        self.backbone = load_wan22_components(
            video_dit_config=dict(_VIDEO_DIT_CONFIG),
            model_id=checkpoint_asset_root or os.environ.get("EGO4WAM_WAN_DIR", _WAN_MODEL_ID),
            device="cpu",
            torch_dtype=torch.float32,
            init_from_config_only=init_from_config_only,
        )
        video_dit = self.backbone.video_dit

        action_dit_config = {
            "hidden_dim": 1024,
            "action_dim": _ACTION_DIM,
            "ffn_dim": 4096,
            "text_dim": int(video_dit.text_dim),
            "freq_dim": int(video_dit.freq_dim),
            "eps": float(video_dit.eps),
            "num_heads": int(video_dit.num_heads),
            "attn_head_dim": int(video_dit.attn_head_dim),
            "num_layers": int(video_dit.num_layers),
        }
        self.action_model = build_ego4wam_action_dit(
            action_dit_config=action_dit_config,
            init_payload_path=None,
            device="cpu",
            torch_dtype=torch.float32,
            init_from_config_only=init_from_config_only,
        )
        self.mot = MoT(gradient_checkpoint=False)
        problems = self.mot.validate_expert(video_dit) + self.mot.validate_expert(self.action_model)
        if problems:
            raise ValueError(f"MoT expert validation failed: {problems}")

        self.video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=_NUM_TRAIN_TIMESTEPS, shift=_FLOW_SHIFT
        )
        self.action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=_NUM_TRAIN_TIMESTEPS, shift=_FLOW_SHIFT
        )
        self.video_num_infer_steps = _NUM_INFERENCE_STEPS
        self.action_num_infer_steps = _NUM_INFERENCE_STEPS
        self.loss_video_weight = 1.0
        self.loss_action_weight = 1.0
        self.video_trainable = True
        self.force_eval_when_frozen = True
        self.proprio_enabled = True
        self.proprio_dim = _STATE_DIM
        if init_from_config_only:
            with torch.device("meta"):
                self.proprio_encoder = nn.Linear(_STATE_DIM, int(video_dit.text_dim))
        else:
            self.proprio_encoder = nn.Linear(_STATE_DIM, int(video_dit.text_dim))
        self.temporal_downsample_factor = int(self.backbone.vae_scale_factor_temporal)
        self._inference_context_cache = None
        self._validate_horizon()

        # --- freeze states ---------------------------------------------
        self._apply_freeze()

    # ------------------------------------------------------------------
    # freeze / train mode
    # ------------------------------------------------------------------
    def _apply_freeze(self) -> None:
        # VAE + text encoder: frozen + eval.
        for p in self.backbone.vae.parameters():
            p.requires_grad_(False)
        self.backbone.vae.eval()
        if self.backbone.text_encoder is not None:
            for p in self.backbone.text_encoder.parameters():
                p.requires_grad_(False)
            self.backbone.text_encoder.eval()
        # video expert
        if not self.video_trainable:
            for p in self.backbone.video_dit.parameters():
                p.requires_grad_(False)
        # action expert + proprio
        for p in self.action_model.parameters():
            p.requires_grad_(True)
        if self.proprio_encoder is not None:
            for p in self.proprio_encoder.parameters():
                p.requires_grad_(True)

    @staticmethod
    def prepare_checkpoint_state_dict(state_dict: dict) -> dict:
        """Restore the tied UMT5 alias omitted by safetensors exporters."""

        alias = "backbone.text_encoder._encoder.shared.weight"
        source = "backbone.text_encoder._encoder.encoder.embed_tokens.weight"
        if alias not in state_dict and source in state_dict:
            state_dict[alias] = state_dict[source]
        return state_dict

    def materialize_runtime_tensors(self) -> None:
        """Rebuild non-persistent RoPE tables after meta checkpoint init."""

        from starVLA.model.modules.mot_model.dit import (
            precompute_freqs_cis,
            precompute_freqs_cis_3d,
        )

        if any(value.is_meta for value in self.backbone.video_dit.freqs):
            self.backbone.video_dit.freqs = precompute_freqs_cis_3d(
                self.backbone.video_dit.attn_head_dim
            )
        if self.action_model.freqs.is_meta:
            self.action_model.freqs = precompute_freqs_cis(
                self.action_model.attn_head_dim,
                end=1024,
            )

    def _validate_horizon(self) -> None:
        expected_frames = self.action_horizon // self.video_stride + 1
        if self.num_video_frames != expected_frames:
            raise RuntimeError(
                "Invalid built-in Ego4WAM temporal contract: "
                f"num_video_frames={self.num_video_frames}, expected={expected_frames}."
            )
        if self.num_video_frames % self.temporal_downsample_factor != 1:
            raise RuntimeError(
                "Invalid built-in Ego4WAM VAE contract: video frame count must "
                "produce one clean first-frame latent followed by future latents."
            )

    def _latent_time_from_video_frames(self) -> int:
        return (self.num_video_frames - 1) // self.temporal_downsample_factor + 1

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._inference_context_cache = None
        # Restore frozen + force_eval modules to eval even after model.train().
        if self.force_eval_when_frozen:
            self.backbone.vae.eval()
            if self.backbone.text_encoder is not None:
                self.backbone.text_encoder.eval()
            if not self.video_trainable:
                self.backbone.video_dit.eval()
        return self

    def _run_mot_forward(
        self,
        input_latents: torch.Tensor,
        first_frame_latents: Optional[torch.Tensor],
        video_context: torch.Tensor,
        video_context_mask: torch.Tensor,
        action_context: torch.Tensor,
        action_context_mask: torch.Tensor,
        action: torch.Tensor,
        image_is_pad: Optional[torch.Tensor] = None,
    ) -> _MotOutput:
        """Sample noise, run MoT, return all intermediate tensors.

        Does NOT compute loss or vis — those are handled by
        :meth:`_compute_losses` and :meth:`_decode_video_vis`.
        """
        device = input_latents.device
        dtype = input_latents.dtype
        B = input_latents.shape[0]

        # --- video branch ---
        noise_video = torch.randn_like(input_latents)
        t_video = self.video_scheduler.sample_training_t(B, device, dtype)
        latents = self.video_scheduler.add_noise(input_latents, noise_video, t_video)
        target_video = self.video_scheduler.training_target(input_latents, noise_video, t_video)
        if first_frame_latents is not None:
            latents[:, :, 0:1] = first_frame_latents

        # --- action branch ---
        noise_action = torch.randn_like(action)
        t_action = self.action_scheduler.sample_training_t(B, device, dtype)
        noisy_action = self.action_scheduler.add_noise(action, noise_action, t_action)
        target_action = self.action_scheduler.training_target(action, noise_action, t_action)

        # --- build states + bindings ---
        latent_frame_pad = None
        if image_is_pad is not None:
            temporal = self.temporal_downsample_factor
            tail_pad = image_is_pad[:, 1:]
            latent_tail_pad = tail_pad.view(B, -1, temporal).all(dim=2)
            latent_frame_pad = torch.cat([image_is_pad[:, :1], latent_tail_pad], dim=1)
        video_valid, frame_ids = self._build_video_token_layout(input_latents, latent_frame_pad)

        v_state = self.backbone.video_dit.prepare(
            latents,
            t_video,
            video_context,
            video_context_mask,
            video_valid,
            frame_ids,
        )
        a_state = self.action_model.prepare(
            noisy_action,
            t_action,
            action_context,
            action_context_mask,
        )
        vb = MoTBranchBinding("video", self.backbone.video_dit, v_state)
        ab = MoTBranchBinding("action", self.action_model, a_state)

        mask = compile_attention_mask(query_bindings=[vb, ab], key_bindings=[vb, ab], rules=self.mot_rules)
        out = self.mot.forward([vb, ab], mask)
        pred_video = self.backbone.video_dit.mot_finalize(out["video"])
        pred_action = self.action_model.mot_finalize(out["action"])

        if first_frame_latents is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        return _MotOutput(
            input_latents=input_latents,
            first_frame_latents=first_frame_latents,
            pred_video=pred_video,
            pred_action=pred_action,
            target_video=target_video,
            target_action=target_action,
            t_video=t_video,
            t_action=t_action,
            latent_frame_pad=latent_frame_pad,
        )

    def _compute_losses(
        self,
        mot_out: _MotOutput,
        action_is_pad: Optional[torch.Tensor] = None,
        image_is_pad: Optional[torch.Tensor] = None,
    ) -> dict:
        """Compute weighted video + action loss from a :class:`_MotOutput`."""
        include_initial = mot_out.first_frame_latents is None
        loss_video = self._video_loss(
            mot_out.pred_video, mot_out.target_video,
            image_is_pad, include_initial, mot_out.t_video,
            latent_frame_pad=mot_out.latent_frame_pad,
        )
        loss_action = self._action_loss(
            mot_out.pred_action, mot_out.target_action,
            action_is_pad, mot_out.t_action,
        )
        loss_total = self.loss_video_weight * loss_video + self.loss_action_weight * loss_action
        return {
            "action_loss": loss_total,
            "loss_metrics": {
                "video": float(self.loss_video_weight * loss_video.detach().item()),
                "action_denoise": float(self.loss_action_weight * loss_action.detach().item()),
            },
        }

    def _compute_training_loss(
        self,
        input_latents: torch.Tensor,
        first_frame_latents: Optional[torch.Tensor],
        video_context: torch.Tensor,
        video_context_mask: torch.Tensor,
        action_context: torch.Tensor,
        action_context_mask: torch.Tensor,
        action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor] = None,
        image_is_pad: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """Thin wrapper for tests: run forward + loss, ignore vis."""
        mot_out = self._run_mot_forward(
            input_latents,
            first_frame_latents,
            video_context,
            video_context_mask,
            action_context,
            action_context_mask,
            action,
            image_is_pad,
        )
        return self._compute_losses(mot_out, action_is_pad, image_is_pad)

    # Loss helpers.
    def _video_loss(self, pred_video, target_video, image_is_pad, include_initial, t_video,
                     latent_frame_pad=None):
        loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            loss_per_sample = loss_token.mean(dim=1)
        else:
            if latent_frame_pad is None:
                temporal = self.temporal_downsample_factor
                tail = image_is_pad[:, 1:]
                latent_tail = tail.view(image_is_pad.shape[0], -1, temporal).all(dim=2)
                latent_frame_pad = latent_tail if not include_initial else torch.cat(
                    [image_is_pad[:, :1], latent_tail], dim=1,
                )
            video_is_pad = latent_frame_pad[:, 1:] if not include_initial else latent_frame_pad
            valid = (~video_is_pad).to(loss_token.dtype)
            loss_per_sample = (loss_token * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        w = self.video_scheduler.training_weight(t_video).to(loss_per_sample.dtype)
        return (loss_per_sample * w).mean()

    def _action_loss(self, pred_action, target_action, action_is_pad, t_action):
        loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
        if action_is_pad is None:
            loss_per_sample = loss_token.mean(dim=1)
        else:
            valid = (~action_is_pad).to(loss_token.dtype)
            loss_per_sample = (loss_token * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        w = self.action_scheduler.training_weight(t_action).to(loss_per_sample.dtype)
        return (loss_per_sample * w).mean()

    def _build_video_token_layout(
        self,
        latents: torch.Tensor,
        latent_frame_pad: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build token validity and frame IDs for ``[B,C,T,H,W]`` latents."""
        if latents.ndim != 5:
            raise ValueError(f"Video latents must be [B,C,T,H,W], got {tuple(latents.shape)}.")
        batch_size, _channels, latent_frames, latent_height, latent_width = latents.shape
        patch_t, patch_h, patch_w = (int(value) for value in self.backbone.video_dit.patch_size)
        if patch_t != 1:
            raise ValueError(f"Ego4WAM token layout requires temporal patch size 1, got {patch_t}.")
        if latent_height % patch_h != 0 or latent_width % patch_w != 0:
            raise ValueError(
                "Latent spatial dimensions must be divisible by the video patch size, "
                f"got {(latent_height, latent_width)} and {(patch_h, patch_w)}."
            )

        tokens_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
        if latent_frame_pad is None:
            frame_valid = torch.ones(
                batch_size,
                latent_frames,
                dtype=torch.bool,
                device=latents.device,
            )
        else:
            expected_shape = (batch_size, latent_frames)
            if tuple(latent_frame_pad.shape) != expected_shape:
                raise ValueError(
                    f"Latent frame pad mask must have shape {expected_shape}, got {tuple(latent_frame_pad.shape)}."
                )
            frame_valid = ~latent_frame_pad.to(device=latents.device, dtype=torch.bool)

        valid_mask = frame_valid.unsqueeze(-1).expand(-1, -1, tokens_per_frame).reshape(batch_size, -1)
        frame_ids = (
            torch.arange(latent_frames, device=latents.device)
            .view(1, latent_frames, 1)
            .expand(batch_size, -1, tokens_per_frame)
            .reshape(batch_size, -1)
        )
        return valid_mask, frame_ids

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    def _resolve_inactive_action_mask(
        self,
        examples: List[dict],
        *,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Resolve per-sample inactive dimensions for analytic-path pinning.

        Requests without ``action_dim_mask`` take the all-active fast path.
        Data adapters may provide the mask for padded dimensions.
        """

        active_masks = []
        expected_shape = (self.action_horizon, self.action_dim)
        for example in examples:
            value = example.get("action_dim_mask")
            if value is None:
                active = np.ones(expected_shape, dtype=bool)
            else:
                active = np.asarray(_to_numpy(value), dtype=bool)
                if active.shape != expected_shape:
                    raise ValueError(
                        "action_dim_mask must have shape "
                        f"{expected_shape}, got {active.shape}."
                    )
            active_masks.append(active)

        inactive = ~torch.as_tensor(
            np.stack(active_masks), dtype=torch.bool, device=device
        )
        return inactive if bool(inactive.any()) else None

    def _pin_inactive_action_latents(
        self,
        action_latents: torch.Tensor,
        initial_action_noise: torch.Tensor,
        inactive_action_mask: torch.Tensor | None,
        timestep: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """Keep inactive dimensions on ``epsilon * sigma_next`` exactly."""

        if inactive_action_mask is None:
            return action_latents
        sigma_next = timestep.to(
            device=action_latents.device, dtype=action_latents.dtype
        ) / float(self.action_scheduler.num_train_timesteps)
        sigma_next = sigma_next + delta.to(
            device=action_latents.device, dtype=action_latents.dtype
        )
        if sigma_next.ndim != 0:
            sigma_next = sigma_next.view(
                -1, *([1] * (action_latents.ndim - 1))
            )
        analytic_latents = initial_action_noise * sigma_next
        return torch.where(
            inactive_action_mask.to(device=action_latents.device),
            analytic_latents,
            action_latents,
        )

    def _predict_action_base(
        self,
        first_frame_latents: torch.Tensor,
        video_context: torch.Tensor,
        video_context_mask: torch.Tensor,
        action_context: torch.Tensor,
        action_context_mask: torch.Tensor,
        inactive_action_mask: torch.Tensor | None = None,
        num_inference_steps: Optional[int] = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        device = first_frame_latents.device
        dtype = first_frame_latents.dtype
        B = first_frame_latents.shape[0]

        t_video = torch.zeros((B,), dtype=dtype, device=device)
        video_valid, frame_ids = self._build_video_token_layout(first_frame_latents)
        v_state = self.backbone.video_dit.prepare(
            first_frame_latents,
            t_video,
            video_context,
            video_context_mask,
            video_valid,
            frame_ids,
        )
        vb = MoTBranchBinding("video", self.backbone.video_dit, v_state)
        self_mask = compile_attention_mask(
            query_bindings=[vb], key_bindings=[vb], rules=[r for r in self.mot_rules if r.query.startswith("video.")]
        )
        cache = self.mot.prefill_branch_cache(vb, self_mask, export_groups=("video",))

        latents_action = torch.randn(
            B,
            self.action_horizon,
            self.action_model.action_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        initial_action_noise = latents_action.detach().clone()
        steps = num_inference_steps or self.action_num_infer_steps
        timesteps, deltas = self.action_scheduler.build_inference_schedule(steps, device, dtype)

        # Pre-compute attention mask (action + cache layouts are invariant across steps).
        t_a = timesteps[0].reshape(1).to(device=device, dtype=dtype).expand(B)
        a_state = self.action_model.prepare(
            latents_action,
            t_a,
            action_context,
            action_context_mask,
        )
        ab = MoTBranchBinding("action", self.action_model, a_state)
        mask = compile_attention_mask(query_bindings=[ab], key_bindings=[cache, ab], rules=self.mot_rules)

        for t, d in zip(timesteps, deltas, strict=True):
            t_a = t.reshape(1).to(device=device, dtype=dtype).expand(B)
            a_state = self.action_model.prepare(
                latents_action,
                t_a,
                action_context,
                action_context_mask,
            )
            ab = MoTBranchBinding("action", self.action_model, a_state)
            out = self.mot.forward_with_cache([ab], [cache], mask)
            pred = self.action_model.mot_finalize(out["action"])
            latents_action = self.action_scheduler.step(pred, d, latents_action)
            latents_action = self._pin_inactive_action_latents(
                latents_action,
                initial_action_noise,
                inactive_action_mask,
                t,
                d,
            )
        return latents_action

    def _predict_action_joint(
        self,
        first_frame_latents: torch.Tensor,
        video_context: torch.Tensor,
        video_context_mask: torch.Tensor,
        action_context: torch.Tensor,
        action_context_mask: torch.Tensor,
        inactive_action_mask: torch.Tensor | None = None,
        num_inference_steps: Optional[int] = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        device = first_frame_latents.device
        dtype = first_frame_latents.dtype
        B = first_frame_latents.shape[0]

        latent_t = self._latent_time_from_video_frames()
        z_dim = self.backbone.vae_z_dim
        latent_h = first_frame_latents.shape[3]
        latent_w = first_frame_latents.shape[4]
        latents_video = torch.randn(
            B,
            z_dim,
            latent_t,
            latent_h,
            latent_w,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        latents_video[:, :, 0:1] = first_frame_latents
        latents_action = torch.randn(
            B,
            self.action_horizon,
            self.action_model.action_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        initial_action_noise = latents_action.detach().clone()

        steps = num_inference_steps or self.video_num_infer_steps
        v_ts, v_ds = self.video_scheduler.build_inference_schedule(steps, device, dtype)
        a_ts, a_ds = self.action_scheduler.build_inference_schedule(steps, device, dtype)
        if len(v_ts) != len(a_ts):
            raise ValueError("Joint inference requires equal video/action step counts (enforced at build).")

        # Pre-compute attention mask and video token layout (invariant across steps).
        valid, frame_ids = self._build_video_token_layout(latents_video)
        t_v = v_ts[0].reshape(1).to(device=device, dtype=dtype).expand(B)
        t_a = a_ts[0].reshape(1).to(device=device, dtype=dtype).expand(B)
        v_state = self.backbone.video_dit.prepare(
            latents_video,
            t_v,
            video_context,
            video_context_mask,
            valid,
            frame_ids,
        )
        a_state = self.action_model.prepare(
            latents_action,
            t_a,
            action_context,
            action_context_mask,
        )
        vb = MoTBranchBinding("video", self.backbone.video_dit, v_state)
        ab = MoTBranchBinding("action", self.action_model, a_state)
        mask = compile_attention_mask(query_bindings=[vb, ab], key_bindings=[vb, ab], rules=self.mot_rules)

        for vt, vd, at, ad in zip(v_ts, v_ds, a_ts, a_ds, strict=True):
            t_v = vt.reshape(1).to(device=device, dtype=dtype).expand(B)
            t_a = at.reshape(1).to(device=device, dtype=dtype).expand(B)
            v_state = self.backbone.video_dit.prepare(
                latents_video,
                t_v,
                video_context,
                video_context_mask,
                valid,
                frame_ids,
            )
            a_state = self.action_model.prepare(
                latents_action,
                t_a,
                action_context,
                action_context_mask,
            )
            vb = MoTBranchBinding("video", self.backbone.video_dit, v_state)
            ab = MoTBranchBinding("action", self.action_model, a_state)
            out = self.mot.forward([vb, ab], mask)
            pred_v = self.backbone.video_dit.mot_finalize(out["video"])
            pred_a = self.action_model.mot_finalize(out["action"])
            latents_video = self.video_scheduler.step(pred_v, vd, latents_video)
            latents_action = self.action_scheduler.step(pred_a, ad, latents_action)
            latents_action = self._pin_inactive_action_latents(
                latents_action,
                initial_action_noise,
                inactive_action_mask,
                at,
                ad,
            )
            latents_video[:, :, 0:1] = first_frame_latents
        return latents_action

    def forward(self, examples: List[dict], **kwargs) -> dict:
        if kwargs:
            raise ValueError(f"Unknown forward kwargs: {sorted(kwargs)}")
        if not isinstance(examples, list) or not examples:
            raise ValueError("forward(examples) expects a non-empty List[dict].")

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        video_context, video_context_mask = self._encode_context(
            [example["lang"] for example in examples], device, dtype
        )

        videos = []
        image_padding = []
        actions = []
        action_padding = []
        for example in examples:
            video, video_is_pad = self._prepare_training_video(example, device, dtype)
            videos.append(video)
            image_padding.append(video_is_pad)

            action = torch.as_tensor(example["action"], dtype=dtype, device=device)
            if action.shape != (self.action_horizon, self.action_dim):
                raise ValueError(
                    f"action must be [{self.action_horizon},{self.action_dim}], "
                    f"got {tuple(action.shape)}"
                )
            actions.append(action)

            action_is_pad = example.get("action_is_pad")
            if action_is_pad is None:
                pad = torch.zeros(self.action_horizon, dtype=torch.bool, device=device)
            else:
                pad = torch.as_tensor(action_is_pad, dtype=torch.bool, device=device)
                if pad.shape != (self.action_horizon,):
                    raise ValueError(
                        f"action_is_pad must have shape ({self.action_horizon},), "
                        f"got {tuple(pad.shape)}"
                    )
            action_padding.append(pad)

        video = torch.cat(videos, dim=0)
        image_is_pad = torch.stack(image_padding, dim=0)
        input_latents = self.backbone.encode_video_latents(video, device=device)
        first_frame_latents = input_latents[:, :, :1]
        action_context, action_context_mask = self._merge_proprio(
            examples, video_context, video_context_mask, device, dtype
        )
        mot_output = self._run_mot_forward(
            input_latents=input_latents,
            first_frame_latents=first_frame_latents,
            video_context=video_context,
            video_context_mask=video_context_mask,
            action_context=action_context,
            action_context_mask=action_context_mask,
            action=torch.stack(actions, dim=0),
            image_is_pad=image_is_pad,
        )
        return self._compute_losses(
            mot_output,
            action_is_pad=torch.stack(action_padding, dim=0),
            image_is_pad=image_is_pad,
        )

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        sampler_seed = kwargs.pop("sampler_seed", None)
        if sampler_seed is not None and (
            isinstance(sampler_seed, bool) or not isinstance(sampler_seed, Integral)
        ):
            raise ValueError("sampler_seed must be an integer or None")
        if kwargs:
            raise ValueError(f"Unknown predict_action kwargs: {sorted(kwargs)}")

        if not isinstance(examples, list):
            examples = [examples]
        if not examples:
            raise ValueError("predict_action(examples) expects a non-empty List[dict].")

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        generator = None
        if sampler_seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(sampler_seed))

        video_context, video_context_mask = self._encode_context(
            [example["lang"] for example in examples], device, dtype
        )
        current_video = torch.cat(
            [self._prepare_current_video(example, device, dtype) for example in examples],
            dim=0,
        )
        first_frame_latents = self.backbone.encode_video_latents(
            current_video[:, :, :1], device=device
        )
        action_context, action_context_mask = self._merge_proprio(
            examples, video_context, video_context_mask, device, dtype
        )
        inactive_action_mask = self._resolve_inactive_action_mask(examples, device=device)

        predict = (
            self._predict_action_base
            if self.interaction_mode == "base"
            else self._predict_action_joint
        )
        actions = predict(
            first_frame_latents,
            video_context,
            video_context_mask,
            action_context,
            action_context_mask,
            inactive_action_mask,
            _NUM_INFERENCE_STEPS,
            generator=generator,
        )
        return {
            "normalized_actions": actions.detach().cpu().float().numpy(),
            "sampler": {
                "algorithm": "euler_flow_matching",
                "version": "ego4wam_euler_v1",
                "seed": None if sampler_seed is None else int(sampler_seed),
                "num_inference_timesteps": _NUM_INFERENCE_STEPS,
            },
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _encode_context(self, prompts: List[str], device, dtype):
        if self.backbone.text_encoder is None:
            raise ValueError(
                "Ego4WAM forward/predict_action needs its bundled text encoder."
            )
        formatted_prompts = [self.prompt_template.format(lang=prompt) for prompt in prompts]
        cache_key = (tuple(formatted_prompts), torch.device(device), dtype)
        if not self.training and self._inference_context_cache is not None:
            cached_key, cached_context, cached_mask = self._inference_context_cache
            if cached_key == cache_key:
                return cached_context, cached_mask

        context, context_mask = self.backbone.encode_prompt(
            formatted_prompts,
            device=device,
            dtype=dtype,
        )
        if not self.training:
            self._inference_context_cache = (cache_key, context, context_mask)
        return context, context_mask

    def _prepare_training_video(self, example: dict, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Consume a pre-composed ``[C,T,H,W]`` training video tensor."""

        if "video_tensor" not in example:
            raise KeyError("Ego4WAM expects a pre-composed `video_tensor`; preprocessing belongs to the data adapter.")
        video = torch.as_tensor(example["video_tensor"], dtype=dtype, device=device)
        if video.ndim != 4 or video.shape[0] != 3 or video.shape[1] != self.num_video_frames:
            raise ValueError(
                f"video_tensor must be [3,{self.num_video_frames},H,W], got {tuple(video.shape)}"
            )
        video_is_pad = example.get("video_is_pad")
        if video_is_pad is None:
            padding = torch.zeros(self.num_video_frames, dtype=torch.bool, device=device)
        else:
            padding = torch.as_tensor(video_is_pad, dtype=torch.bool, device=device)
            if padding.shape != (self.num_video_frames,):
                raise ValueError(
                    f"video_is_pad must have shape ({self.num_video_frames},), got {tuple(padding.shape)}"
                )
        return video.unsqueeze(0), padding

    def _prepare_current_video(self, example: dict, device, dtype) -> torch.Tensor:
        """Consume a pre-composed ``[3,T,H,W]`` inference video tensor."""

        if "video_tensor" not in example:
            raise KeyError("Ego4WAM expects a pre-composed `video_tensor`; preprocessing belongs to the data adapter.")
        video = torch.as_tensor(example["video_tensor"], dtype=dtype, device=device)
        if video.ndim != 4 or video.shape[0] != 3 or video.shape[1] < 1:
            raise ValueError(f"video_tensor must be [3,T,H,W] with T>=1, got {tuple(video.shape)}")
        return video.unsqueeze(0)

    def _merge_proprio(self, examples, context, context_mask, device, dtype):
        if not self.proprio_enabled:
            return context, context_mask
        B = context.shape[0]
        proprio_list = []
        for ex in examples:
            p = self._prepare_proprio(ex.get("state"), device, dtype)
            if p is None:
                # Pad with zeros so the batch dimension stays B.
                p = torch.zeros(1, self.proprio_dim, device=device, dtype=dtype)
            proprio_list.append(p)
        if not proprio_list:
            return context, context_mask
        proprio = torch.cat(proprio_list, dim=0)
        if proprio.shape[0] != B:
            raise RuntimeError(
                f"proprio batch size mismatch: expected {B}, got {proprio.shape[0]}"
            )
        proprio_token = self.proprio_encoder(proprio.unsqueeze(1)).to(dtype)
        proprio_mask = torch.ones((B, 1), dtype=torch.bool, device=device)
        context = torch.cat([context, proprio_token], dim=1)
        context_mask = torch.cat([context_mask, proprio_mask], dim=1)
        return context, context_mask

    def _prepare_proprio(self, state: Optional[np.ndarray], device, dtype) -> Optional[torch.Tensor]:
        if self.proprio_encoder is None or state is None:
            return None
        state = np.asarray(state)
        if state.ndim == 2:
            state = state[-1:]  # take current state (last entry of hist_state)
        elif state.ndim == 1:
            state = state[None, :]
        else:
            raise ValueError(
                f"proprio state must be 1D or 2D, got ndim={state.ndim} shape={state.shape}"
            )
        if state.shape[1] != self.proprio_dim:
            raise ValueError(f"proprio last dim {state.shape[1]} != proprio_dim {self.proprio_dim}")
        return torch.from_numpy(state).float().to(device=device, dtype=dtype)
