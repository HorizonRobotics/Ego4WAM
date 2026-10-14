#!/usr/bin/env bash
# Copyright (c) 2026 Beijing Horizon Information Technology Company Limited.
# Author: Ego4WAM Team.
# SPDX-License-Identifier: Apache-2.0
set -uo pipefail

max_retries="${ROBODOJO_MAX_BASH_RETRIES:-3}"
attempt=0

while true; do
  "$@"
  status=$?
  case "${status}" in
    0)
      exit 0
      ;;
    98)
      echo "[eval_entry] restarting after completed batch (status=${status})" >&2
      sleep 5
      ;;
    99|134|139)
      attempt=$((attempt + 1))
      if (( attempt >= max_retries )); then
        echo "[eval_entry] giving up after ${attempt} restart(s) (status=${status})" >&2
        exit "${status}"
      fi
      echo "[eval_entry] restarting after status=${status} (${attempt}/${max_retries})" >&2
      sleep 5
      ;;
    *)
      exit "${status}"
      ;;
  esac
done
