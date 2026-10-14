# Source provenance: StarVLA project (https://github.com/starVLA/starVLA).
# Modifications Copyright (c) 2026 Beijing Horizon Information Technology Company Limited.
# Author: Ego4WAM Team.
# SPDX-License-Identifier: Apache-2.0 AND MIT
# Ego4WAM additions/modifications: Apache-2.0; upstream StarVLA portions: MIT.
# See LICENSES/StarVLA-MIT.txt for the retained upstream license.

"""The RoboDojo-only data registry used by the Ego4WAM release.

Registration lives in the installed package, not example-directory discovery.
The LeRobot reader and mixture sampler retain their upstream behavior.
"""

from starVLA.dataloader.ego4wam.robodojo_data_config import Ego4WAMRoboDojoDataConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag as EmbodimentTag

ROBOT_TYPE_CONFIG_MAP = {"ego4wam_robodojo_arx_x5": Ego4WAMRoboDojoDataConfig()}
DATASET_NAMED_MIXTURES = {
    "ego4wam_robodojo_v21": [("RoboDojo_lerobot_v21_video", 1.0, "ego4wam_robodojo_arx_x5")],
}
ROBOT_TYPE_TO_EMBODIMENT_TAG = {key: config.embodiment_tag for key, config in ROBOT_TYPE_CONFIG_MAP.items()}
