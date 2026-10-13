# Copyright (c) 2026 Ego4WAM Authors.
# SPDX-License-Identifier: Apache-2.0
# Derived from the historical RoboDojo XPolicyLab starVLA adapter.
# Modified for the independent Ego4WAM policy and per-environment geometry.

from __future__ import annotations

from typing import Any

import numpy as np
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)

from .robodojo_client_contract import (
    RAW_OBSERVATION_SCHEMA,
    build_raw_observation,
    build_world_to_camera_by_role,
)


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        self.action_type = self.model_cfg.get("action_type", "joint")
        if self.action_type != "joint":
            raise ValueError("Ego4WAM currently supports action_type='joint' first")

        self.env_cfg_type = self.model_cfg.get("env_cfg_type")
        if self.env_cfg_type is None:
            raise ValueError("Ego4WAM requires env_cfg_type")
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.action_dim = sum(self.robot_action_dim_info["arm_dim"]) + sum(self.robot_action_dim_info["ee_dim"])

        from .transport import RawPolicyClient

        self.client = RawPolicyClient(
            self.model_cfg.get("ego4wam_server_host", "127.0.0.1"),
            int(self.model_cfg.get("ego4wam_server_port", 5694)),
        )
        server_meta = self.client.get_server_metadata()
        if server_meta.get("server_preprocessing") is not True:
            raise ValueError("RoboDojo raw observations require server_preprocessing=true")
        if server_meta.get("raw_observation_schema") != RAW_OBSERVATION_SCHEMA:
            raise ValueError(
                "RoboDojo raw observation schema mismatch: "
                f"expected {RAW_OBSERVATION_SCHEMA!r}, got {server_meta.get('raw_observation_schema')!r}"
            )
        if server_meta.get("proprio_enabled") is False:
            raise ValueError("RoboDojo raw observations always include state, but server proprioception is disabled")

        self.action_chunk_size = int(server_meta["action_chunk_size"])
        self.execute_horizon = int(self.model_cfg.get("execute_horizon", self.action_chunk_size))
        if self.execute_horizon < 1 or self.execute_horizon > self.action_chunk_size:
            raise ValueError(
                f"execute_horizon={self.execute_horizon} must satisfy "
                f"1 <= execute_horizon <= action_chunk_size={self.action_chunk_size}"
            )

        self.unnorm_key = self.model_cfg.get("unnorm_key", "robodojo_arx_x5")

        self.obs_by_env: dict[int, dict[str, Any]] = {}
        self.previous_world_to_camera_by_env: dict[int, dict[str, np.ndarray]] = {}
        self.action_chunks_by_env: dict[int, np.ndarray] = {}
        self.cursor_by_env: dict[int, int] = {}
        self._latest_env_idx_list = [0]

        print(
            f"[Ego4WAM] connected to Ego4WAM server, action_dim={self.action_dim}, "
            f"chunk={self.action_chunk_size}, execute_horizon={self.execute_horizon}, "
            f"raw_schema={RAW_OBSERVATION_SCHEMA}, action_order=xpolicy, metadata={server_meta}"
        )

    def _convert_obs(self, observation: dict[str, Any]) -> dict[str, Any]:
        state = pack_robot_state(
            observation,
            self.action_type,
            self.robot_action_dim_info,
            source_type="obs",
        )
        converted_obs = build_raw_observation(
            observation,
            state=state,
            fallback_instruction=str(self.model_cfg.get("task_name", "")),
        )
        origin = np.asarray(observation.get("env_origin"), dtype=np.float32).reshape(-1)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("Ego4WAM requires a finite per-environment origin")
        converted_obs["env_origin"] = np.ascontiguousarray(origin)
        return converted_obs

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = []
        for obs in obs_list:
            env_idx = int(obs.get("env_idx", 0))
            self._latest_env_idx_list.append(env_idx)
            converted_obs = self._convert_obs(obs)
            camera_observations = converted_obs["camera_observations"]
            current_world_to_camera = {
                camera_name: np.asarray(camera["T_world2cam"]).copy()
                for camera_name, camera in camera_observations.items()
            }
            previous_world_to_camera = self.previous_world_to_camera_by_env.get(env_idx)
            if previous_world_to_camera is not None:
                if previous_world_to_camera.keys() != current_world_to_camera.keys():
                    raise ValueError(
                        "camera set changed between observations: "
                        f"previous={sorted(previous_world_to_camera)}, "
                        f"current={sorted(current_world_to_camera)}"
                    )
                for camera_name, world_to_camera in previous_world_to_camera.items():
                    camera_observations[camera_name]["T_world2cam"] = world_to_camera
            self.previous_world_to_camera_by_env[env_idx] = current_world_to_camera
            self.obs_by_env[env_idx] = converted_obs

    def update_camera_pose_batch(self, obs_list):
        for obs in obs_list:
            env_idx = int(obs.get("env_idx", 0))
            if env_idx not in self.obs_by_env:
                raise AssertionError(f"update_obs must be called before update_camera_pose_batch for env_idx={env_idx}")
            world_to_camera = build_world_to_camera_by_role(obs)
            previous_world_to_camera = self.previous_world_to_camera_by_env.get(env_idx)
            if previous_world_to_camera is not None and previous_world_to_camera.keys() != world_to_camera.keys():
                raise ValueError(
                    "camera set changed during pose update: "
                    f"previous={sorted(previous_world_to_camera)}, current={sorted(world_to_camera)}"
                )
            self.previous_world_to_camera_by_env[env_idx] = {
                camera_name: np.asarray(matrix).copy() for camera_name, matrix in world_to_camera.items()
            }

    def _infer_chunks(self, env_idx_list: list[int]) -> None:
        missing_envs = [env_idx for env_idx in env_idx_list if env_idx not in self.obs_by_env]
        if missing_envs:
            raise AssertionError(f"update_obs must be called before get_action for envs={missing_envs}")

        vla_input = {
            "examples": [self.obs_by_env[env_idx] for env_idx in env_idx_list],
            "unnorm_key": self.unnorm_key,
        }
        response = self.client.predict_action(vla_input)
        if not response.get("ok", False):
            raise RuntimeError(f"Ego4WAM inference failed: {response.get('error', response)}")
        chunks = np.asarray(response["data"]["actions"], dtype=np.float32)
        if chunks.ndim != 3:
            raise ValueError(f"expected batched action chunks [B, T, {self.action_dim}], got {chunks.shape}")
        if chunks.shape[0] != len(env_idx_list):
            raise ValueError(f"expected {len(env_idx_list)} action chunks, got {chunks.shape[0]}")
        if chunks.shape[-1] != self.action_dim:
            raise ValueError(f"expected action dim {self.action_dim}, got {chunks.shape[-1]}")
        if chunks.shape[1] < self.execute_horizon:
            raise ValueError(
                f"action chunk length {chunks.shape[1]} is shorter than " f"execute_horizon={self.execute_horizon}"
            )
        if not np.isfinite(chunks).all():
            raise ValueError("action chunks contain NaN or infinity")

        for env_idx, chunk in zip(env_idx_list, chunks, strict=True):
            self.action_chunks_by_env[env_idx] = chunk
            self.cursor_by_env[env_idx] = 0

    def _consume_action_vectors(self, env_idx: int) -> np.ndarray:
        chunk = self.action_chunks_by_env.get(env_idx)
        cursor = self.cursor_by_env.get(env_idx, self.execute_horizon)
        if chunk is None or cursor >= self.execute_horizon:
            raise AssertionError(f"action chunk for env_idx={env_idx} was not refreshed")

        actions = np.asarray(chunk[cursor : self.execute_horizon], dtype=np.float32)
        self.cursor_by_env[env_idx] = self.execute_horizon
        return actions

    def get_action(self):
        return self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]])[0]

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = self._latest_env_idx_list
        env_idx_list = [int(env_idx) for env_idx in env_idx_list]
        envs_needing_chunk = [
            env_idx
            for env_idx in env_idx_list
            if self.action_chunks_by_env.get(env_idx) is None
            or self.cursor_by_env.get(env_idx, self.execute_horizon) >= self.execute_horizon
        ]
        if envs_needing_chunk:
            self._infer_chunks(envs_needing_chunk)
        action_vectors_by_env = [self._consume_action_vectors(env_idx) for env_idx in env_idx_list]
        chunk_lengths = {len(action_vectors) for action_vectors in action_vectors_by_env}
        if len(chunk_lengths) != 1:
            raise ValueError(f"active environments have mismatched action chunk lengths: {sorted(chunk_lengths)}")
        return [
            [
                unpack_robot_state(
                    action_vector,
                    self.action_type,
                    self.robot_action_dim_info,
                    source_type="obs",
                )
                for action_vector in action_vectors
            ]
            for action_vectors in action_vectors_by_env
        ]

    def reset(self):
        self.obs_by_env.clear()
        self.previous_world_to_camera_by_env.clear()
        self.action_chunks_by_env.clear()
        self.cursor_by_env.clear()
        self._latest_env_idx_list = [0]
