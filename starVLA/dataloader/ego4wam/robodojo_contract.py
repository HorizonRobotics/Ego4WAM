# Copyright (c) 2026 Beijing Horizon Information Technology Company Limited.
# Author: Ego4WAM Team.
# SPDX-License-Identifier: MIT

"""The single RoboDojo training contract supported by Ego4WAM."""

from pathlib import Path

ACTION_DIM = 32
ACTION_HORIZON = 32
VIDEO_OFFSETS = tuple(range(0, 33, 4))
VIDEO_FRAMES = len(VIDEO_OFFSETS)
CAMERA_KEYS = (
    "video.cam_left_wrist",
    "video.cam_right_wrist",
    "video.cam_high",
)
ROT6D_DIMS = tuple(range(17, 23)) + tuple(range(26, 32))

ASSET_ROOT = Path(__file__).with_name("assets")
URDF_PATH = ASSET_ROOT / "robotwin2_dual_arm_arx_x5a.pruned.urdf"
STATS_PATH = ASSET_ROOT / "robodojo_normalization_stats.json"
PROMPTS_PATH = ASSET_ROOT / "robodojo_task_prompts.json"

# Fixed transforms used to express ARX TCP poses in the head-camera frame.
BASE_TO_WORLD = (
    (0.0, -1.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, -0.45),
    (0.0, 0.0, 1.0, 0.765),
    (0.0, 0.0, 0.0, 1.0),
)
WORLD_TO_HEAD_CAMERA = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, -0.86602534, -0.5000001, 0.29892973),
    (0.0, 0.5000001, -0.86602534, 1.33776116),
    (0.0, 0.0, 0.0, 1.0),
)
EEF_FRAME_FROM_FK = (
    (0.0, 0.0, 1.0),
    (0.0, -1.0, 0.0),
    (1.0, 0.0, 0.0),
)
ROLL_X_90 = (
    (1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 1.0, 0.0),
)
