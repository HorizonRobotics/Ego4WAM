# Copyright (c) 2026 Beijing Horizon Information Technology Company Limited.
# Author: Ego4WAM Team.
# SPDX-License-Identifier: MIT

"""Contract tests for the online Ego4WAM RoboDojo policy adapter."""

from __future__ import annotations

import numpy as np

from deployment.model_server.ego4wam_robodojo_policy import Ego4WAMRoboDojoPolicy
from starVLA.dataloader.ego4wam.robodojo_contract import BASE_TO_WORLD
from starVLA.dataloader.ego4wam.robodojo_kinematics import RoboDojoJointsEEF32


class _ZeroPolicy:
    def predict_action(self, examples, **kwargs):
        del kwargs
        return {
            "normalized_actions": np.zeros((len(examples), 32, 32), dtype=np.float32)
        }


def _raw_example() -> dict:
    identity = np.eye(4, dtype=np.float32)

    def camera(value: int) -> dict:
        return {
            "image": np.full((480, 640, 3), value, dtype=np.uint8),
            "color_space": "rgb",
            "depth": np.ones((480, 640), dtype=np.float32),
            "depth_unit": "meters",
            "intrinsic": np.eye(3, dtype=np.float32),
            "T_world2cam": identity.copy(),
        }

    return {
        "camera_observations": {
            "head": camera(224),
            "left_wrist": camera(32),
            "right_wrist": camera(96),
        },
        "state": np.zeros(14, dtype=np.float32),
        "env_origin": np.zeros(3, dtype=np.float32),
        "lang": "build a tower",
    }


def _policy_without_loading_weights() -> Ego4WAMRoboDojoPolicy:
    policy = Ego4WAMRoboDojoPolicy.__new__(Ego4WAMRoboDojoPolicy)
    policy.encoder = RoboDojoJointsEEF32()
    policy.base_to_world = np.asarray(BASE_TO_WORLD, dtype=np.float32)
    policy.sampler_steps = 20
    policy.model = _ZeroPolicy()
    policy.artifact_dir = "test-artifact"
    return policy


def test_live_observation_builds_model_contract() -> None:
    sample = _policy_without_loading_weights().build_sample(_raw_example())
    assert sample["video_tensor"].shape == (3, 1, 384, 320)
    assert sample["video_tensor"].dtype == np.float32
    assert sample["state"].shape == (1, 32)
    assert sample["lang"] == "build a tower"
    np.testing.assert_allclose(
        sample["video_tensor"][:, 0, 100, 160],
        224 / 127.5 - 1.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        sample["video_tensor"][:, 0, 320, 80],
        32 / 127.5 - 1.0,
        atol=1e-6,
    )


def test_live_extrinsic_changes_eef_but_not_joint_state() -> None:
    policy = _policy_without_loading_weights()
    first = policy.build_sample(_raw_example())["state"]
    shifted_example = _raw_example()
    shifted_example["camera_observations"]["head"]["T_world2cam"][0, 3] = 0.25
    shifted = policy.build_sample(shifted_example)["state"]
    np.testing.assert_array_equal(first[:, :14], shifted[:, :14])
    assert not np.array_equal(first[:, 14:], shifted[:, 14:])


def test_policy_returns_14d_absolute_joint_chunks() -> None:
    policy = _policy_without_loading_weights()
    result = policy.predict_action([_raw_example()], num_ddim_steps=20)
    assert result["actions"].shape == (1, 32, 14)
    assert result["actions"].dtype == np.float32
    expected = policy.encoder.unnormalize(
        np.zeros((1, 32, 32), dtype=np.float32)
    )[..., :14]
    np.testing.assert_array_equal(result["actions"], expected)
