# Copyright (c) 2026 Ego4WAM Authors.
# SPDX-License-Identifier: Apache-2.0

"""Fixed three-camera, nine-frame video composition used by Ego4WAM."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _aspect_pad_5_by_4(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"camera frames must be [T,H,W,3], got {frames.shape}")
    height, width = frames.shape[1:3]
    unit = max((width + 4) // 5, (height + 3) // 4)
    canvas_width, canvas_height = unit * 5, unit * 4
    pad_w, pad_h = canvas_width - width, canvas_height - height
    return np.ascontiguousarray(
        np.pad(
            frames,
            ((0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)),
            mode="constant",
        )
    )


def _resize(frames: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(frames.transpose(0, 3, 1, 2))).float()
    resized = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False, antialias=True)
    return resized.permute(0, 2, 3, 1).numpy()


def compose_robodojo_video(camera_frames: list[np.ndarray] | tuple[np.ndarray, ...]) -> np.ndarray:
    """Compose left wrist, right wrist, and head frames into ``[3,T,384,320]``."""

    if len(camera_frames) != 3:
        raise ValueError(f"Ego4WAM RoboDojo requires three cameras, got {len(camera_frames)}")
    # Historical compose_video normalizes each view before bilinear resizing.
    # Keeping that order avoids the fp32 drift from resizing 0..255 first.
    left, right, head = (
        2.0 * (_aspect_pad_5_by_4(np.asarray(frames)).astype(np.float32) / 255.0) - 1.0
        for frames in camera_frames
    )
    head = _resize(head, (256, 320))
    left = _resize(left, (128, 160))
    right = _resize(right, (128, 160))
    mosaic = np.concatenate((head, np.concatenate((left, right), axis=2)), axis=1)
    return np.ascontiguousarray(mosaic.transpose(3, 0, 1, 2), dtype=np.float32)


def compose_robodojo_live_video(
    camera_images: list[np.ndarray] | tuple[np.ndarray, ...],
) -> np.ndarray:
    """Replay historical online preprocessing for one RGB frame per camera.

    XPolicyLab provides raw views in ``[left wrist, right wrist, head]`` order.
    The historical server first aspect-padded every view to 320x256 and then
    built the 320x384 mosaic (including the second wrist resize). Keeping that
    two-stage resize is important for old/new action parity.
    """

    if len(camera_images) != 3:
        raise ValueError(f"Ego4WAM RoboDojo requires three live cameras, got {len(camera_images)}")
    # The historical raw-observation adapter performs this first resize in
    # OpenCV while pixels are still uint8. Preserve that rounding exactly;
    # the later mosaic resize remains the torch/float path used by training.
    import cv2

    prepared = []
    for image in camera_images:
        values = np.asarray(image)
        if values.ndim != 3 or values.shape[-1] != 3 or values.dtype != np.uint8:
            raise ValueError(
                "live camera images must be uint8 HWC RGB, "
                f"got shape={values.shape}, dtype={values.dtype}"
            )
        padded = _aspect_pad_5_by_4(values[None])[0]
        prepared.append(cv2.resize(padded, (320, 256))[None])
    return compose_robodojo_video(prepared)
