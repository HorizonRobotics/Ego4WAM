# Copyright (c) 2026 Ego4WAM Authors.
# SPDX-License-Identifier: Apache-2.0
"""MoT model shared core (DiT blocks + backbone-agnostic MoT protocol).

These components are shared by the Wan video expert (``WanMoT.py``), the action
expert (``Ego4WAMActionDiT.py``), and the ``Ego4WAM`` framework orchestrator.
The flow-matching scheduler lives in ``starVLA.model.modules.scheduler``.
"""

from .dit import (
    CrossAttention,
    DiTBlock,
    Head,
    RMSNorm,
    SelfAttention,
    modulate,
    precompute_freqs_cis,
    precompute_freqs_cis_3d,
    rope_apply,
    scaled_dot_product_attention,
    sinusoidal_embedding_1d,
)
from .mot import (
    BranchKVCache,
    InteractionRule,
    LayerKVCache,
    MoT,
    MoTBranchBinding,
    MoTExpertState,
    MoTLayerIO,
    MoTTokenLayout,
    compile_attention_mask,
)
