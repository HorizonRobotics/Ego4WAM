# Copyright (c) StarVLA Team.
# Modifications Copyright (c) 2026 Beijing Horizon Information Technology Company Limited.
# Author: Ego4WAM Team.
# SPDX-License-Identifier: Apache-2.0 AND MIT
# Ego4WAM additions/modifications: Apache-2.0; upstream StarVLA portions: MIT.
# See LICENSES/StarVLA-MIT.txt for the retained upstream license.

"""RoboDojo data components for Ego4WAM.

Training imports lerobot_datasets lazily from train_ego4wam.create_dataloader.
Importing the online policy must not initialize unrelated training readers.
"""
