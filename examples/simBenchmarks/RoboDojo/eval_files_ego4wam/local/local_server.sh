#!/usr/bin/env bash
# Copyright (c) 2026 Beijing Horizon Information Technology Company Limited.
# Author: Ego4WAM Team.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
: "${CKPT:?Set CKPT to the HF Joint release directory}"
: "${EGO4WAM_PYTHON:?Set EGO4WAM_PYTHON to the model Python executable}"
for item in model.safetensors inference_config.yaml model_assets; do
  [[ -e "${CKPT}/${item}" ]] || { echo "Missing release item: ${CKPT}/${item}" >&2; exit 1; }
done
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${SERVER_CUDA_VISIBLE_DEVICES:-0,1}"
unset DEBUG
cd "${REPO_ROOT}"
exec "${EGO4WAM_PYTHON}" -u deployment/model_server/server_policy.py \
  --ckpt_path "${CKPT}" --robodojo-ego4wam --device cuda:0,cuda:1 \
  --port "${PORT:-6789}" --idle_timeout -1 \
  --max-batch-size "${MAX_BATCH_SIZE:-10}" --batch-timeout-ms "${BATCH_TIMEOUT_MS:-20}"
