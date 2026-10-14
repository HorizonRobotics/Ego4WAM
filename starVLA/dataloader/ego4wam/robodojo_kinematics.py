# Copyright (c) 2026 Beijing Horizon Information Technology Company Limited.
# Author: Ego4WAM Team.
# SPDX-License-Identifier: Apache-2.0

"""ARX X5A joint/TCP encoding for Ego4WAM's fixed 32-D action space."""

from __future__ import annotations

import json

import numpy as np
import torch

from .robodojo_contract import (
    ACTION_DIM,
    BASE_TO_WORLD,
    EEF_FRAME_FROM_FK,
    ROLL_X_90,
    ROT6D_DIMS,
    STATS_PATH,
    URDF_PATH,
    WORLD_TO_HEAD_CAMERA,
)


class RoboDojoJointsEEF32:
    """Encode absolute ARX joints and two camera-frame TCP poses as 32-D rows."""

    def __init__(self) -> None:
        try:
            import pytorch_kinematics as pk
        except ImportError as exc:
            raise ImportError("Ego4WAM RoboDojo loading requires pytorch-kinematics") from exc

        self.chain = pk.build_chain_from_urdf(URDF_PATH.read_bytes()).to(dtype=torch.float32)
        self.base_to_ego = torch.tensor(WORLD_TO_HEAD_CAMERA, dtype=torch.float32) @ torch.tensor(
            BASE_TO_WORLD, dtype=torch.float32
        )
        self.eef_rotation = torch.tensor(EEF_FRAME_FROM_FK, dtype=torch.float32) @ torch.tensor(
            ROLL_X_90, dtype=torch.float32
        )

        stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
        if (
            stats.get("version") != "ego4wam_robodojo_normalization_v1"
            or stats.get("action_dim") != ACTION_DIM
        ):
            raise ValueError(f"Unsupported Ego4WAM RoboDojo statistics: {STATS_PATH}")
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        selected = np.ones(ACTION_DIM, dtype=bool)
        selected[list(ROT6D_DIMS)] = False
        selected &= std >= 1e-4
        self.mean = np.where(selected, mean, 0.0).astype(np.float32)
        self.scale = np.where(selected, 1.0 / np.maximum(std, 1e-4), 1.0).astype(np.float32)
        self.unscale = np.where(selected, std, 1.0).astype(np.float32)

    def encode(
        self,
        joints: np.ndarray,
        *,
        base_to_ego: np.ndarray | None = None,
    ) -> np.ndarray:
        joints = np.asarray(joints, dtype=np.float32)
        if joints.ndim == 1:
            joints = joints[None]
        if joints.ndim != 2 or joints.shape[1] != 14:
            raise ValueError(f"RoboDojo joints must have shape [T,14], got {joints.shape}")

        # msgpack may expose a read-only NumPy view. PyTorch warns that such a
        # tensor has undefined write behavior, so materialize one owned C-order
        # array at the protocol boundary.
        q = torch.from_numpy(np.array(joints, dtype=np.float32, order="C", copy=True))
        chain_q = torch.zeros((q.shape[0], len(self.chain.get_joints())), dtype=torch.float32)
        chain_q[:, 0:6] = q[:, 0:6]
        chain_q[:, 8:14] = q[:, 7:13]
        poses = self.chain.forward_kinematics(chain_q)
        tcp = torch.stack(
            (
                poses["left_link6_tcp"].get_matrix(),
                poses["right_link6_tcp"].get_matrix(),
            ),
            dim=1,
        )
        if base_to_ego is None:
            transform = self.base_to_ego
        else:
            transform = torch.as_tensor(base_to_ego, dtype=torch.float32)
            if transform.shape != (4, 4) or not torch.isfinite(transform).all():
                raise ValueError(
                    "base_to_ego must be one finite [4,4] transform, "
                    f"got {tuple(transform.shape)}"
                )
        tcp = transform[None, None] @ tcp
        rotation = tcp[..., :3, :3] @ self.eef_rotation
        rotation_6d = torch.cat((rotation[..., :, 0], rotation[..., :, 1]), dim=-1)
        eef = torch.cat((tcp[..., :3, 3], rotation_6d), dim=-1).reshape(q.shape[0], 18)
        return torch.cat((q, eef), dim=-1).numpy().astype(np.float32, copy=False)

    def normalize(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.shape[-1] != ACTION_DIM:
            raise ValueError(f"Ego4WAM state/action must end in {ACTION_DIM}, got {values.shape}")
        return np.ascontiguousarray((values - self.mean) * self.scale)

    def unnormalize(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.shape[-1] != ACTION_DIM:
            raise ValueError(f"Ego4WAM state/action must end in {ACTION_DIM}, got {values.shape}")
        # Match the historical serving normalizer's multiply-by-std operation
        # order exactly (division by reciprocal has different fp32 rounding).
        return np.ascontiguousarray(values * self.unscale + self.mean)
