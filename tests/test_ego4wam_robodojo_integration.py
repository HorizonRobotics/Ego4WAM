# Copyright (c) 2026 Beijing Horizon Information Technology Company Limited.
# Author: Ego4WAM Team.
# SPDX-License-Identifier: MIT

"""Optional parity smoke test against a local official RoboDojo download."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


DATASET_ROOT = os.environ.get("EGO4WAM_ROBODOJO_DATASET")
pytestmark = pytest.mark.skipif(not DATASET_ROOT, reason="set EGO4WAM_ROBODOJO_DATASET")


def test_official_episode_zero_matches_checkpoint_contract() -> None:
    from starVLA.dataloader.ego4wam import Ego4WAMRoboDojoDataset
    from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
    from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
    from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform

    modalities = {
        "video": ModalityConfig(
            delta_indices=list(range(0, 33, 4)),
            modality_keys=[
                "video.cam_left_wrist",
                "video.cam_right_wrist",
                "video.cam_high",
            ],
        ),
        "state": ModalityConfig(delta_indices=[0], modality_keys=["state.joints_eef"]),
        "action": ModalityConfig(delta_indices=list(range(32)), modality_keys=["action.joints_eef"]),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    }
    dataset = Ego4WAMRoboDojoDataset(
        dataset_path=Path(DATASET_ROOT),
        modality_configs=modalities,
        transforms=ComposedModalityTransform(transforms=[]),
        embodiment_tag=EmbodimentTag.ARX_X5,
        video_backend="pyav",
        data_cfg={"lerobot_version": "v2.0"},
    )
    sample = dataset.get_step_data(0, 0)
    expected_state_head = np.asarray(
        [
            0.5929635,
            -1.0571153,
            -0.95455086,
            0.52589065,
            -0.22289805,
            -0.00520851,
            0.67946756,
            -0.55145806,
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(sample["state"][0, :8], expected_state_head, atol=2e-6, rtol=0)
    assert sample["video_tensor"].shape == (3, 9, 384, 320)
    assert sample["state"].shape == (1, 32)
    assert sample["action"].shape == (32, 32)
    assert not sample["video_is_pad"].any()
    assert not sample["action_is_pad"].any()
    assert sample["lang"] == (
        "Arrange the numbers from left to right to form the largest possible number, "
        "and place them on the pad"
    )
