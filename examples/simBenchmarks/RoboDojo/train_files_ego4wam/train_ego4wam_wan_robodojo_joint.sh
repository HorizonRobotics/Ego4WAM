#!/usr/bin/env bash
# Copyright (c) 2026 Ego4WAM Authors.
# SPDX-License-Identifier: MIT
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export config_yaml="${SCRIPT_DIR}/ego4wam_wan_robodojo_joint.yaml"
export training_entrypoint=starVLA/training/train_ego4wam.py
export accelerate_config=examples/simBenchmarks/RoboDojo/train_files_ego4wam/ego4wam_deepspeed_zero2.yaml
exec bash "${SCRIPT_DIR}/run_robodojo_train.sh" "$@"
