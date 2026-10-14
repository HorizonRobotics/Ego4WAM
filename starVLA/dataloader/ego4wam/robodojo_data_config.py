# Copyright (c) 2026 Beijing Horizon Information Technology Company Limited.
# Author: Ego4WAM Team.
# SPDX-License-Identifier: Apache-2.0

"""Packaged registration for the unchanged Ego4WAM RoboDojo data contract."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform


class Ego4WAMRoboDojoDataConfig:
    """The fixed RoboDojo data contract used by both Ego4WAM attention modes."""

    embodiment_tag = EmbodimentTag.ARX_X5
    video_keys = [  # noqa: RUF012 - keep the historical DataConfig class attributes
        "video.cam_left_wrist",
        "video.cam_right_wrist",
        "video.cam_high",
    ]
    state_keys = ["state.joints_eef"]  # noqa: RUF012
    action_keys = ["action.joints_eef"]  # noqa: RUF012
    language_keys = ["annotation.human.action.task_description"]  # noqa: RUF012

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=list(range(0, 33, 4)),
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(delta_indices=[0], modality_keys=self.state_keys),
            "action": ModalityConfig(
                delta_indices=list(range(32)),
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=[0],
                modality_keys=self.language_keys,
            ),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[])

    def make_dataset(self, **dataset_kwargs):
        from starVLA.dataloader.ego4wam import Ego4WAMRoboDojoDataset

        return Ego4WAMRoboDojoDataset(**dataset_kwargs)
