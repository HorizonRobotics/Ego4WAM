#!/usr/bin/env bash
# Copyright (c) 2026 Ego4WAM Authors.
# SPDX-License-Identifier: MIT
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../common/run_client.sh" "$@"
