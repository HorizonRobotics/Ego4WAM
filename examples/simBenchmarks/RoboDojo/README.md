<!--
Modifications Copyright (c) 2026 Beijing Horizon Information Technology Company Limited.
Author: Ego4WAM Team.
SPDX-License-Identifier: MIT
-->

# Ego4WAM on RoboDojo

This release contains only Ego4WAM RoboDojo post-training and the paper's local evaluation path.

- Four post-training recipes: Wan initialization or video-pretrained initialization, each with Base/Joint attention. See [the repository guide](../../../README.md#-robodojo-post-training) and `train_files_ego4wam/`.
- Model/simulator/proxy setup, historical source pins, evaluation and result aggregation: [RoboDojo reproduction guide](../../../docs/RoboDojo.md) and [evaluation files](eval_files_ego4wam/README.md).
- Data registration is packaged in `starVLA/dataloader/ego4wam/robodojo_data_config.py`; the underlying StarVLA/NVIDIA LeRobot reader is retained.

Other StarVLA benchmarks, model backends and deployment servers are not included. The Python package remains named `starVLA` so existing Ego4WAM checkpoint keys and commands remain compatible.
