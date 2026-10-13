# Copyright (c) 2026 Ego4WAM Authors.
# SPDX-License-Identifier: Apache-2.0
# Derived from the historical RoboDojo XPolicyLab starVLA adapter.
# Packaged independently for Ego4WAM without changing the transport contract.

"""RoboDojo raw observation transport helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np

RAW_OBSERVATION_SCHEMA = "camera_observations_v1"
CAMERA_ROLES = (
    ("head", ("cam_head", "head_camera")),
    ("left_wrist", ("cam_left_wrist", "left_camera")),
    ("right_wrist", ("cam_right_wrist", "right_camera")),
)
CAMERA_LABELS = tuple(role for role, _camera_names in CAMERA_ROLES)
GRIPPER_INDICES = (6, 13)


def decode_rgb_image(image: Any) -> np.ndarray:
    """Decode transport encodings without changing image geometry or color order."""

    if isinstance(image, (bytes, bytearray, memoryview)):
        image = np.frombuffer(bytes(image), dtype=np.uint8)

    values = np.asarray(image)
    if values.ndim == 1 and values.dtype == np.uint8:
        decoded = cv2.imdecode(values, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("failed to decode compressed image bytes")
        values = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB image, got shape {values.shape}")
    if values.dtype != np.uint8:
        raise TypeError(f"expected uint8 RGB image, got dtype {values.dtype}")
    return np.ascontiguousarray(values)


def validate_raw_state(state: Any, *, action_dim: int = 14) -> np.ndarray:
    """Validate a physical joint state without normalizing it."""

    values = np.asarray(state, dtype=np.float32)
    if values.shape != (action_dim,):
        raise ValueError(f"expected raw state shape ({action_dim},), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("raw state contains NaN or infinity")
    for index in GRIPPER_INDICES:
        if index >= action_dim:
            continue
        if not -1.0e-4 <= float(values[index]) <= 1.0001:
            raise ValueError(f"raw gripper state at index {index} must be in [0, 1], got {values[index]}")
    return np.ascontiguousarray(values)


def _camera_mapping(observation: Mapping[str, Any], camera_names: Sequence[str]) -> tuple[str, Mapping[str, Any]]:
    vision = observation.get("vision", {})
    if not isinstance(vision, Mapping):
        raise TypeError("observation['vision'] must be a mapping")
    for camera_name in camera_names:
        camera = vision.get(camera_name)
        if camera is not None:
            if not isinstance(camera, Mapping):
                raise TypeError(f"camera {camera_name!r} must be a mapping")
            return camera_name, camera
    raise KeyError(f"missing camera from candidates: {list(camera_names)}")


def _required_camera_value(camera: Mapping[str, Any], camera_name: str, keys: Sequence[str]) -> Any:
    for key in keys:
        if key in camera:
            return camera[key]
    raise KeyError(f"camera {camera_name!r} is missing required field from {list(keys)}")


def _validate_camera_matrix(value: Any, *, name: str, shapes: tuple[tuple[int, int], ...]) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape not in shapes:
        expected = " or ".join(f"{rows}x{columns}" for rows, columns in shapes)
        raise ValueError(f"{name} must be {expected}, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return np.ascontiguousarray(matrix)


def _world_to_camera(value: Any, *, name: str) -> np.ndarray:
    """Convert RoboDojo's USD camera-to-world pose to OpenCV world-to-camera."""

    camera_to_world_usd = _validate_camera_matrix(value, name=name, shapes=((4, 4),))
    if abs(float(np.linalg.det(camera_to_world_usd))) < 1.0e-12:
        raise ValueError(f"{name} must be invertible")
    usd_from_opencv = np.diag((1.0, -1.0, -1.0, 1.0))
    camera_to_world_opencv = camera_to_world_usd @ usd_from_opencv
    return np.ascontiguousarray(np.linalg.inv(camera_to_world_opencv))


def _raw_depth(value: Any, *, image_shape: tuple[int, int], name: str) -> np.ndarray:
    depth = np.asarray(value)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.shape != image_shape:
        raise ValueError(f"{name} must match RGB shape {image_shape}, got {depth.shape}")
    if not np.issubdtype(depth.dtype, np.floating):
        raise TypeError(f"{name} must contain metric floating-point depth, got dtype {depth.dtype}")
    return np.ascontiguousarray(depth.astype(np.float32, copy=False))


def build_raw_observation(
    observation: Mapping[str, Any],
    *,
    state: Any,
    fallback_instruction: str = "",
) -> dict[str, Any]:
    """Build the server-owned preprocessing schema from untouched simulator data."""

    instruction = observation.get("instruction") or observation.get("instructions")
    if isinstance(instruction, (list, tuple)):
        instruction = instruction[0] if instruction else ""
    if instruction in (None, ""):
        instruction = fallback_instruction

    camera_observations = {}
    for role, camera_names in CAMERA_ROLES:
        camera_name, camera = _camera_mapping(observation, camera_names)
        image = decode_rgb_image(_required_camera_value(camera, camera_name, ("color", "rgb", "colors")))
        depth = _raw_depth(
            _required_camera_value(camera, camera_name, ("depth",)),
            image_shape=image.shape[:2],
            name=f"{camera_name}.depth",
        )
        intrinsic = _validate_camera_matrix(
            _required_camera_value(camera, camera_name, ("intrinsic_matrix",)),
            name=f"{camera_name}.intrinsic_matrix",
            shapes=((3, 3), (4, 4)),
        )
        world_to_camera = _world_to_camera(
            _required_camera_value(camera, camera_name, ("extrinsic_matrix", "extrinsics_matrix")),
            name=f"{camera_name}.extrinsic_matrix",
        )
        camera_observations[role] = {
            "image": image,
            "color_space": "rgb",
            "depth": depth,
            "depth_unit": "meters",
            "intrinsic": intrinsic,
            "T_world2cam": world_to_camera,
        }

    return {
        "lang": str(instruction),
        "camera_observations": camera_observations,
        "state": validate_raw_state(state),
    }


def build_world_to_camera_by_role(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    world_to_camera_by_role = {}
    for role, camera_names in CAMERA_ROLES:
        camera_name, camera = _camera_mapping(observation, camera_names)
        world_to_camera_by_role[role] = _world_to_camera(
            _required_camera_value(camera, camera_name, ("extrinsic_matrix", "extrinsics_matrix")),
            name=f"{camera_name}.extrinsic_matrix",
        )
    return world_to_camera_by_role


__all__ = [
    "CAMERA_LABELS",
    "RAW_OBSERVATION_SCHEMA",
    "build_raw_observation",
    "build_world_to_camera_by_role",
    "decode_rgb_image",
    "validate_raw_state",
]
