# Copyright (c) 2026 Ego4WAM Authors.
# SPDX-License-Identifier: MIT
"""CUDA smoke tests for Ego4WAM's coupled experts and Base cache path."""

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from starVLA.model.modules.action_model.Ego4WAMActionDiT import Ego4WAMActionDiT  # noqa: E402
from starVLA.model.modules.mot_model import (
    InteractionRule,
    MoT,
    MoTBranchBinding,
    compile_attention_mask,
)  # noqa: E402
from starVLA.model.modules.world_model.WanMoT import WanVideoDiT  # noqa: E402


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


BASE_RULES = (
    InteractionRule("video.video", "video.video", "first_frame_causal"),
    InteractionRule("action.action", "video.video", "key_first_frame"),
    InteractionRule("action.action", "action.action", "full"),
)
JOINT_RULES = (
    InteractionRule("video.video", "video.video", "first_frame_causal"),
    InteractionRule("action.action", "video.video", "full"),
    InteractionRule("action.action", "action.action", "full"),
)


def _tiny_experts():
    video = WanVideoDiT(
        hidden_dim=16,
        in_dim=4,
        ffn_dim=32,
        out_dim=4,
        text_dim=10,
        freq_dim=8,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=2,
        attn_head_dim=8,
        num_layers=2,
        video_attention_mask_mode="first_frame_causal",
        use_gradient_checkpointing=False,
    ).cuda()
    action = Ego4WAMActionDiT(
        hidden_dim=12,
        action_dim=6,
        ffn_dim=24,
        text_dim=10,
        freq_dim=8,
        eps=1e-6,
        num_heads=2,
        attn_head_dim=8,
        num_layers=2,
    ).cuda()
    return video, action, MoT().cuda()


def _inputs(batch_size: int = 2):
    device = torch.device("cuda")
    context = torch.randn(batch_size, 5, 10, device=device)
    context_mask = torch.ones(batch_size, 5, device=device, dtype=torch.bool)
    video_latents = torch.randn(batch_size, 4, 3, 4, 4, device=device)
    video_timestep = torch.rand(batch_size, device=device)
    video_valid = torch.ones(batch_size, 12, device=device, dtype=torch.bool)
    frame_ids = (
        torch.arange(3, device=device)
        .repeat_interleave(4)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    action_tokens = torch.randn(batch_size, 7, 6, device=device)
    action_timestep = torch.rand(batch_size, device=device)
    return (
        context,
        context_mask,
        video_latents,
        video_timestep,
        video_valid,
        frame_ids,
        action_tokens,
        action_timestep,
    )


@pytest.mark.parametrize("rules", [BASE_RULES, JOINT_RULES], ids=["base", "joint"])
def test_coupled_video_action_forward_backward(rules) -> None:
    torch.manual_seed(0)
    video, action, mot = _tiny_experts()
    context, context_mask, video_latents, vt, valid, frame_ids, actions, at = _inputs()

    video_state = video.prepare(video_latents, vt, context, context_mask, valid, frame_ids)
    action_state = action.prepare(actions, at, context, context_mask)
    video_branch = MoTBranchBinding("video", video, video_state)
    action_branch = MoTBranchBinding("action", action, action_state)
    branches = [video_branch, action_branch]
    mask = compile_attention_mask(
        query_bindings=branches,
        key_bindings=branches,
        rules=rules,
    )

    states = mot(branches, mask)
    video_output = video.mot_finalize(states["video"])
    action_output = action.mot_finalize(states["action"])
    loss = video_output.square().mean() + action_output.square().mean()
    loss.backward()

    assert video_output.shape == video_latents.shape
    assert action_output.shape == actions.shape
    assert torch.isfinite(video_output).all()
    assert torch.isfinite(action_output).all()
    grads = [
        parameter.grad
        for parameter in (*video.parameters(), *action.parameters())
        if parameter.grad is not None
    ]
    assert grads
    assert all(torch.isfinite(gradient).all() for gradient in grads)


def test_base_cached_action_forward() -> None:
    torch.manual_seed(0)
    video, action, mot = _tiny_experts()
    context, context_mask, video_latents, vt, _, _, actions, at = _inputs()
    batch_size = video_latents.shape[0]

    first_frame = video_latents[:, :, :1]
    valid = torch.ones(batch_size, 4, device="cuda", dtype=torch.bool)
    frame_ids = torch.zeros(batch_size, 4, device="cuda", dtype=torch.long)
    video_state = video.prepare(first_frame, vt, context, context_mask, valid, frame_ids)
    video_branch = MoTBranchBinding("video", video, video_state)
    video_mask = compile_attention_mask(
        query_bindings=[video_branch],
        key_bindings=[video_branch],
        rules=BASE_RULES,
    )
    cache = mot.prefill_branch_cache(video_branch, video_mask, export_groups=("video",))

    action_state = action.prepare(actions, at, context, context_mask)
    action_branch = MoTBranchBinding("action", action, action_state)
    action_mask = compile_attention_mask(
        query_bindings=[action_branch],
        key_bindings=[cache, action_branch],
        rules=BASE_RULES,
    )
    states = mot.forward_with_cache([action_branch], [cache], action_mask)
    output = action.mot_finalize(states["action"])

    assert cache.layers[0].k.shape[2] == 4
    assert output.shape == actions.shape
    assert torch.isfinite(output).all()
