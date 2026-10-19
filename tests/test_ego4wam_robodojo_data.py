# Copyright (c) 2026 Beijing Horizon Information Technology Company Limited.
# Author: Ego4WAM Team.
# SPDX-License-Identifier: MIT

"""Unit tests for the fixed Ego4WAM RoboDojo data contract."""

from __future__ import annotations

import numpy as np

from starVLA.dataloader.ego4wam.robodojo_contract import VIDEO_OFFSETS
from starVLA.dataloader.ego4wam.robodojo_contract import STATS_PATH
from starVLA.dataloader.ego4wam.robodojo_dataset import (
    build_action_joint_targets,
    build_video_pad_mask,
)
from starVLA.dataloader.ego4wam.robodojo_video import compose_robodojo_video


def test_public_stats_asset_does_not_expose_experiment_config_name() -> None:
    import json

    assert STATS_PATH.name == "robodojo_normalization_stats.json"
    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    assert stats["version"] == "ego4wam_robodojo_normalization_v1"
    assert not {
        "parts_aggregated",
        "parts_expected",
        "parts_missing",
        "per_dim_count",
        "rows",
        "sample_failures",
        "skipped_non_finite",
    }.intersection(stats)


def test_action_targets_preserve_historical_gripper_offset() -> None:
    released = np.arange(40 * 14, dtype=np.float32).reshape(40, 14)
    targets, is_pad = build_action_joint_targets(released, base_index=0)
    np.testing.assert_array_equal(targets[:, :6], released[:32, :6])
    np.testing.assert_array_equal(targets[:, 7:13], released[:32, 7:13])
    np.testing.assert_array_equal(targets[:, 6], released[1:33, 6])
    np.testing.assert_array_equal(targets[:, 13], released[1:33, 13])
    assert not is_pad.any()


def test_action_and_video_terminal_masks_use_different_logical_lengths() -> None:
    released = np.arange(10 * 14, dtype=np.float32).reshape(10, 14)
    targets, action_is_pad = build_action_joint_targets(released, base_index=9)
    assert action_is_pad.tolist() == [False] + [True] * 31
    np.testing.assert_array_equal(targets, np.repeat(released[-1:], 32, axis=0))

    # Official frame 9 is current. Logical frame 10 is the valid duplicated
    # terminal frame, while all later offsets are padding.
    expected = [(9 + offset) > 10 for offset in VIDEO_OFFSETS]
    assert build_video_pad_mask(9, 10).tolist() == expected


def test_video_composition_layout_and_range() -> None:
    left = np.full((9, 480, 640, 3), 32, dtype=np.uint8)
    right = np.full((9, 480, 640, 3), 96, dtype=np.uint8)
    head = np.full((9, 480, 640, 3), 224, dtype=np.uint8)
    video = compose_robodojo_video([left, right, head])
    assert video.shape == (3, 9, 384, 320)
    assert video.dtype == np.float32
    # Antialiased float32 interpolation can overshoot an endpoint by one ULP.
    assert -1.0 - 2e-7 <= float(video.min()) <= float(video.max()) <= 1.0 + 2e-7
    # Ignore padded borders and sample the interior of each view.
    np.testing.assert_allclose(video[:, :, 100, 160], 224 / 127.5 - 1.0, atol=1e-6)
    np.testing.assert_allclose(video[:, :, 320, 80], 32 / 127.5 - 1.0, atol=1e-6)
    np.testing.assert_allclose(video[:, :, 320, 240], 96 / 127.5 - 1.0, atol=1e-6)
