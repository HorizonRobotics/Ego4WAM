#!/usr/bin/env bash
# Copyright (c) 2026 Ego4WAM Authors.
# SPDX-License-Identifier: MIT
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" != "--help" && "${1:-}" != "-h" ]]; then
  : "${EGO4WAM_PRETRAINED_DIR:?Set EGO4WAM_PRETRAINED_DIR to Ego4WAM_VideoPretrained}"
  for item in model.safetensors inference_config.yaml model_assets; do
    [[ -e "${EGO4WAM_PRETRAINED_DIR}/${item}" ]] || { echo "Missing pretrained item: ${item}" >&2; exit 1; }
  done
fi
export config_yaml="${SCRIPT_DIR}/ego4wam_video_pretrained_robodojo_base.yaml"
export training_entrypoint=starVLA/training/train_ego4wam.py
export accelerate_config=examples/simBenchmarks/RoboDojo/train_files_ego4wam/ego4wam_deepspeed_zero2.yaml
exec bash "${SCRIPT_DIR}/run_robodojo_train.sh" "$@"
