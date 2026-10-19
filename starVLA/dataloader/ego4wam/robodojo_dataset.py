# Copyright (c) 2026 Beijing Horizon Information Technology Company Limited.
# Author: Ego4WAM Team.
# SPDX-License-Identifier: MIT

"""RoboDojo LeRobot-v2.1 adapter for Ego4WAM."""

from __future__ import annotations

import json

import numpy as np

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.schema import DatasetMetadata, LeRobotModalityMetadata

from .robodojo_contract import (
    ACTION_HORIZON,
    CAMERA_KEYS,
    PROMPTS_PATH,
    STATS_PATH,
    VIDEO_OFFSETS,
)
from .robodojo_kinematics import RoboDojoJointsEEF32
from .robodojo_video import compose_robodojo_video


def build_action_joint_targets(released_action: np.ndarray, base_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the historical arm/gripper timing and return its pad mask."""

    released_action = np.asarray(released_action, dtype=np.float32)
    if released_action.ndim != 2 or released_action.shape[1] != 14 or len(released_action) == 0:
        raise ValueError(f"released RoboDojo action must be non-empty [T,14], got {released_action.shape}")
    length = len(released_action)
    requested = base_index + np.arange(ACTION_HORIZON, dtype=np.int64)
    action_indices = np.minimum(requested, length - 1)
    gripper_indices = np.minimum(action_indices + 1, length - 1)
    targets = released_action[action_indices].copy()
    targets[:, (6, 13)] = released_action[gripper_indices][:, (6, 13)]
    return targets, requested >= length


def build_video_pad_mask(base_index: int, episode_length: int) -> np.ndarray:
    """Mark offsets beyond the duplicated logical terminal frame as padding."""

    if episode_length <= 0:
        raise ValueError("episode_length must be positive")
    return base_index + np.asarray(VIDEO_OFFSETS) > episode_length


class Ego4WAMRoboDojoDataset(LeRobotSingleDataset):
    """Adapt the official 14-D RoboDojo release to Ego4WAM's fixed contract."""

    def __init__(self, *args, **kwargs) -> None:
        self._ego4wam_encoder = None
        self._legacy_prompts = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
        super().__init__(*args, **kwargs)

    @property
    def encoder(self) -> RoboDojoJointsEEF32:
        # Construct lazily inside each DataLoader worker; kinematic chains are
        # expensive to pickle and are read-only once built.
        if self._ego4wam_encoder is None:
            self._ego4wam_encoder = RoboDojoJointsEEF32()
        return self._ego4wam_encoder

    def _get_lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        return LeRobotModalityMetadata.model_validate(
            {
                "state": {
                    "joints_eef": {
                        "start": 0,
                        "end": 14,
                        "dtype": "float32",
                        "original_key": "observation.state",
                    }
                },
                "action": {
                    "joints_eef": {
                        "start": 0,
                        "end": 14,
                        "dtype": "float32",
                        "original_key": "action",
                    }
                },
                "video": {
                    "cam_left_wrist": {"original_key": "observation.images.cam_left_wrist"},
                    "cam_right_wrist": {"original_key": "observation.images.cam_right_wrist"},
                    "cam_high": {"original_key": "observation.images.cam_high"},
                },
                "annotation": {
                    "human.action.task_description": {"original_key": "task_index"}
                },
            }
        )

    def _get_metadata(self, embodiment_tag) -> DatasetMetadata:
        stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
        values = {
            "max": stats["maximum"],
            "min": stats["minimum"],
            "mean": stats["mean"],
            "std": stats["std"],
            # The adapter performs its own versioned z-score transform. These
            # fields only satisfy StarVLA's mixture metadata contract.
            "q01": stats["minimum"],
            "q99": stats["maximum"],
        }
        video = {
            key.removeprefix("video."): {
                "resolution": (640, 480),
                "channels": 3,
                "fps": 25.0,
            }
            for key in CAMERA_KEYS
        }
        return DatasetMetadata.model_validate(
            {
                "statistics": {
                    "state": {"joints_eef": values},
                    "action": {"joints_eef": values},
                },
                "modalities": {
                    "video": video,
                    "state": {
                        "joints_eef": {
                            "absolute": True,
                            "shape": (32,),
                            "continuous": True,
                        }
                    },
                    "action": {
                        "joints_eef": {
                            "absolute": True,
                            "shape": (32,),
                            "continuous": True,
                        }
                    },
                },
                "embodiment_tag": embodiment_tag,
            }
        )

    def _get_all_steps(self) -> list[tuple[int, int]]:
        # The generic implementation writes a cache into meta/. Official
        # benchmark data remains immutable in the Ego4WAM path.
        return [
            (int(trajectory_id), step)
            for trajectory_id, length in zip(self.trajectory_ids, self.trajectory_lengths, strict=True)
            for step in range(int(length))
        ]

    def _check_integrity(self) -> None:
        required = {
            "observation.state",
            "action",
            "timestamp",
            "task_index",
        }
        info_features = set(self.lerobot_info_meta.get("features", {}))
        missing = required - info_features
        if missing:
            raise ValueError(f"RoboDojo LeRobot data is missing required features: {sorted(missing)}")

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        frame = self.get_trajectory_data(trajectory_id)
        self.curr_traj_id = trajectory_id
        self.curr_traj_data = frame
        length = len(frame)
        if not 0 <= base_index < length:
            raise IndexError(f"RoboDojo frame {base_index} is outside episode length {length}")

        observation = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
        released_action = np.stack(frame["action"].to_numpy()).astype(np.float32)

        action_joints, action_is_pad = build_action_joint_targets(released_action, base_index)

        state = self.encoder.normalize(self.encoder.encode(observation[base_index]))
        action = self.encoder.normalize(self.encoder.encode(action_joints))
        camera_frames = [self.get_video(trajectory_id, key, base_index) for key in CAMERA_KEYS]
        video_tensor = compose_robodojo_video(camera_frames)
        # The official export omits one terminal frame that was an exact copy
        # of its predecessor. Treat that logical frame as valid while reusing
        # the last physical video frame.
        video_is_pad = build_video_pad_mask(base_index, length)

        task_index = int(frame["task_index"].iloc[base_index])
        task = self._legacy_prompts[str(task_index)].rstrip(".")
        return {
            "video_tensor": video_tensor,
            "video_is_pad": video_is_pad.astype(bool),
            "state": state,
            "action": action,
            "action_is_pad": action_is_pad.astype(bool),
            # The framework owns the model prompt template. Keep the adapter's
            # output to the raw historical task text so it is formatted once.
            "lang": task,
            "robot_tag": self.tag,
            "episode_index": int(trajectory_id),
            "frame_index": int(base_index),
        }

    def _pack_sample(self, data: dict) -> dict:
        return data
