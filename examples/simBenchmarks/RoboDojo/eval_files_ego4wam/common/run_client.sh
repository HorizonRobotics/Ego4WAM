#!/usr/bin/env bash
# Copyright (c) 2026 Ego4WAM Authors.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
: "${CKPT:?Set CKPT to the HF release directory}"
: "${ROBODOJO_ROOT:?Set ROBODOJO_ROOT to the patched historical source checkout}"
: "${ROBODOJO_ASSET_ROOT:?Set ROBODOJO_ASSET_ROOT to the benchmark assets}"
: "${ROBODOJO_PYTHON:?Set ROBODOJO_PYTHON to the simulator Python executable}"
: "${ROBODOJO_PROXY_PYTHON:?Set ROBODOJO_PROXY_PYTHON to the separate proxy Python executable}"
export HOST="${HOST:-127.0.0.1}"
"${ROBODOJO_PYTHON}" "${SCRIPT_DIR}/../prepare_runtime.py" --robodojo-root "${ROBODOJO_ROOT}"
export CLIENT_SHARED_RUN_ID="${CLIENT_SHARED_RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
source "${SCRIPT_DIR}/task_sets.sh"
resolve_robodojo_task_names
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
source "${SCRIPT_DIR}/source_paths.sh"
export PATH="$(dirname "${ROBODOJO_PYTHON}"):${PATH}"
cd "${REPO_ROOT}"
exec "${ROBODOJO_PYTHON}" -u "${SCRIPT_DIR}/run_client.py"
