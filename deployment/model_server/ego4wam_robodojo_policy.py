# Copyright (c) 2026 Ego4WAM Authors.
# SPDX-License-Identifier: MIT

"""RoboDojo raw-observation policy wrapper for Ego4WAM artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from starVLA.dataloader.ego4wam.robodojo_contract import BASE_TO_WORLD
from starVLA.dataloader.ego4wam.robodojo_kinematics import RoboDojoJointsEEF32
from starVLA.dataloader.ego4wam.robodojo_video import compose_robodojo_live_video
from starVLA.model.framework.base_framework import baseframework


CAMERA_ROLES = ("head", "left_wrist", "right_wrist")
ACTION_KEYS = [
    *[f"action.left_joint_{index}" for index in range(1, 7)],
    "action.left_gripper",
    *[f"action.right_joint_{index}" for index in range(1, 7)],
    "action.right_gripper",
]
STATE_KEYS = [key.replace("action.", "state.", 1) for key in ACTION_KEYS]


def _resolve_devices(device: str) -> tuple[torch.device, torch.device]:
    names = [name.strip() for name in device.split(",")]
    if not all(names) or len(names) > 2:
        raise ValueError("Ego4WAM device must be one device or a comma-separated pair")
    if names == ["cuda"] and torch.cuda.device_count() >= 2:
        names = ["cuda:0", "cuda:1"]
    devices = tuple(torch.device(name) for name in names)
    return (devices[0], devices[-1])


def _dispatch_model(model, devices: tuple[torch.device, torch.device]):
    main_device, text_device = devices
    target_dtype = torch.bfloat16 if main_device.type == "cuda" else torch.float32
    if main_device == text_device:
        return model.to(device=main_device, dtype=target_dtype)

    if any(device.type != "cuda" or device.index is None for device in devices):
        raise ValueError("two-device Ego4WAM inference requires explicit CUDA devices")
    from accelerate import dispatch_model

    model.to(dtype=target_dtype)
    return dispatch_model(
        model,
        device_map={
            "backbone.vae": main_device,
            "backbone.video_dit": main_device,
            "backbone.text_encoder": text_device,
            "action_model": main_device,
            "proprio_encoder": main_device,
        },
        main_device=main_device,
    )


class Ego4WAMRoboDojoPolicy:
    """Serve one Ego4WAM artifact through RoboDojo's 14-D joint contract."""

    def __init__(self, artifact_dir: str | Path, *, device: str = "cuda") -> None:
        self.artifact_dir = Path(artifact_dir).expanduser().resolve(strict=True)
        weights_path = self.artifact_dir / "model.safetensors"
        config_path = self.artifact_dir / "inference_config.yaml"
        assets_path = self.artifact_dir / "model_assets"
        if not weights_path.is_file() or not config_path.is_file() or not assets_path.is_dir():
            raise FileNotFoundError(
                "Ego4WAM artifact directory requires model.safetensors, "
                "inference_config.yaml, and model_assets/"
            )

        devices = _resolve_devices(device)
        if devices[0] == devices[1] and devices[0].type == "cuda":
            weight_bytes = weights_path.stat().st_size
            total_memory = torch.cuda.get_device_properties(devices[0]).total_memory
            if weight_bytes >= total_memory:
                raise RuntimeError(
                    "Ego4WAM weights do not fit on one selected GPU; expose two GPUs "
                    "and use device='cuda' or 'cuda:0,cuda:1'"
                )

        self.model = baseframework.from_pretrained(str(weights_path))
        if getattr(self.model, "interaction_mode", None) != "joint":
            raise ValueError("RoboDojo release policy requires Ego4WAM interaction_mode='joint'")
        if int(getattr(self.model, "action_dim", -1)) != 32:
            raise ValueError("RoboDojo Ego4WAM model must predict 32-D actions")
        if int(getattr(self.model, "action_horizon", -1)) != 32:
            raise ValueError("RoboDojo Ego4WAM model must use a 32-step horizon")
        self.sampler_steps = int(getattr(self.model, "action_num_infer_steps", -1))
        if self.sampler_steps != 20:
            raise ValueError(
                f"Ego4WAM requires the fixed 20-step sampler, got {self.sampler_steps}"
            )
        self.model = _dispatch_model(self.model, devices).eval()
        self.encoder = RoboDojoJointsEEF32()
        self.base_to_world = np.asarray(BASE_TO_WORLD, dtype=np.float32)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "env": "starvla_policy_server",
            "ckpt_path": str(self.artifact_dir),
            "action_chunk_size": 32,
            "available_unnorm_keys": ["robodojo_arx_x5"],
            "default_unnorm_key": "robodojo_arx_x5",
            "action_format": "piper_joint_qpos",
            "joint_action_encoding": "joints_eef_32_abs",
            "action_keys": list(ACTION_KEYS),
            "state_keys": list(STATE_KEYS),
            "normalize_state_action": True,
            "proprio_enabled": True,
            "observation_image_size": [320, 256],
            "image_geometry_mode": "wo_virtual_camera",
            "camera_layout": "piper_mosaic",
            "action_num_inference_steps": 20,
            "requires_t_world2head": True,
            "provides_eef_ee_poses": False,
            "raw_observation_schema": "camera_observations_v1",
            "server_preprocessing": True,
            "camera_roles": list(CAMERA_ROLES),
            "requires_env_origin": True,
            "camera_type": "wo_virtual_camera",
            "target_aspect": [5.0, 4.0],
        }

    @staticmethod
    def _prepare_raw(example: Mapping[str, Any]) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, str]:
        cameras = example.get("camera_observations")
        if not isinstance(cameras, Mapping):
            raise TypeError("Ego4WAM requires raw camera_observations")
        if set(cameras) != set(CAMERA_ROLES):
            raise ValueError(
                f"camera roles must be exactly {CAMERA_ROLES}, got {sorted(cameras)}"
            )

        images = []
        head_world_to_camera = None
        for role in CAMERA_ROLES:
            entry = cameras[role]
            if not isinstance(entry, Mapping):
                raise TypeError(f"camera_observations[{role!r}] must be a mapping")
            required = {
                "image", "color_space", "depth", "depth_unit", "intrinsic", "T_world2cam"
            }
            missing = sorted(required - set(entry))
            if missing:
                raise KeyError(f"camera_observations[{role!r}] is missing fields {missing}")
            image = np.asarray(entry.get("image"))
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError(f"camera {role!r} image must be uint8 HWC RGB")
            if str(entry.get("color_space", "")).lower() != "rgb":
                raise ValueError(f"camera {role!r} color_space must be 'rgb'")
            depth = np.asarray(entry["depth"])
            if depth.shape != image.shape[:2] or not np.issubdtype(depth.dtype, np.floating):
                raise ValueError(f"camera {role!r} depth must be floating HxW matching image")
            if entry["depth_unit"] != "meters":
                raise ValueError(f"camera {role!r} depth_unit must be 'meters'")
            intrinsic = np.asarray(entry["intrinsic"], dtype=np.float64)
            if intrinsic.shape not in ((3, 3), (4, 4)) or not np.isfinite(intrinsic).all():
                raise ValueError(f"camera {role!r} intrinsic must be finite 3x3 or 4x4")
            images.append(np.ascontiguousarray(image))
            if role == "head":
                head_world_to_camera = np.asarray(entry.get("T_world2cam"), dtype=np.float64)

        assert head_world_to_camera is not None
        if head_world_to_camera.shape != (4, 4) or not np.isfinite(head_world_to_camera).all():
            raise ValueError("head T_world2cam must be one finite [4,4] matrix")
        origin = np.asarray(example.get("env_origin"), dtype=np.float64).reshape(-1)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("RoboDojo observation requires one finite env_origin 3-vector")
        local_to_global = np.eye(4, dtype=np.float64)
        local_to_global[:3, 3] = origin
        world_to_head = head_world_to_camera @ local_to_global

        state = np.asarray(example.get("state"), dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("RoboDojo state must be one finite 14-vector")
        return images, state, world_to_head.astype(np.float32), str(example.get("lang", ""))

    def build_sample(self, example: Mapping[str, Any]) -> dict[str, Any]:
        images, joints, world_to_head, lang = self._prepare_raw(example)
        head, left, right = images
        video = compose_robodojo_live_video([left, right, head])
        base_to_ego = world_to_head @ self.base_to_world
        state = self.encoder.normalize(
            self.encoder.encode(joints, base_to_ego=base_to_ego)
        )
        return {
            "video_tensor": video,
            "state": state,
            "lang": lang,
        }

    def predict_action(
        self,
        examples: list[dict],
        unnorm_key: str | None = None,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        if unnorm_key not in (None, "robodojo_arx_x5"):
            raise ValueError("Ego4WAM RoboDojo only supports unnorm_key='robodojo_arx_x5'")
        if not isinstance(examples, list) or not examples:
            raise ValueError("examples must be a non-empty list")
        for key in ("num_inference_steps", "num_ddim_steps"):
            requested = kwargs.pop(key, None)
            if requested is not None and int(requested) != self.sampler_steps:
                raise ValueError(f"{key}={requested} conflicts with fixed sampler steps={self.sampler_steps}")

        samples = [self.build_sample(example) for example in examples]
        output = self.model.predict_action(samples, **kwargs)
        normalized = np.asarray(output["normalized_actions"], dtype=np.float32)
        expected = (len(samples), 32, 32)
        if normalized.shape != expected or not np.isfinite(normalized).all():
            raise ValueError(f"Ego4WAM normalized actions must be finite {expected}, got {normalized.shape}")
        actions = self.encoder.unnormalize(normalized)[..., :14]
        return {"actions": actions.astype(np.float32, copy=False)}


__all__ = ["Ego4WAMRoboDojoPolicy"]
