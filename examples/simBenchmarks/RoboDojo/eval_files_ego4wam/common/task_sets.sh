#!/usr/bin/env bash
# Copyright (c) 2026 Beijing Horizon Information Technology Company Limited.
# Author: Ego4WAM Team.
# SPDX-License-Identifier: MIT

ROBODOJO_TASK_SETS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/task_sets" && pwd)"

robodojo_available_task_sets() {
  local task_set_path
  for task_set_path in "${ROBODOJO_TASK_SETS_DIR}"/*.txt; do
    [[ -f "${task_set_path}" ]] || continue
    basename "${task_set_path}" .txt
  done
}

resolve_robodojo_task_names() {
  if [[ -n "${TASK_NAMES:-}" ]]; then
    if [[ -n "${TASK_SET_NAME:-}" ]]; then
      echo "[ROBODOJO CLIENT] TASK_SET_NAME cannot be combined with TASK_NAMES" >&2
      return 1
    fi
    export TASK_NAMES
    return
  fi

  local task_set_name="${TASK_SET_NAME:-FULL_TASKS}"
  if [[ ! "${task_set_name}" =~ ^[A-Za-z0-9_]+$ ]]; then
    echo "[ROBODOJO CLIENT] invalid TASK_SET_NAME=${task_set_name}" >&2
    return 1
  fi

  local task_set_path="${ROBODOJO_TASK_SETS_DIR}/${task_set_name}.txt"
  if [[ ! -f "${task_set_path}" ]]; then
    echo "[ROBODOJO CLIENT] unknown TASK_SET_NAME=${task_set_name}" >&2
    echo "[ROBODOJO CLIENT] available task sets: $(robodojo_available_task_sets | tr '\n' ' ')" >&2
    return 1
  fi

  local task_name
  local -a task_names=()
  while IFS= read -r task_name || [[ -n "${task_name}" ]]; do
    task_name="${task_name%$'\r'}"
    [[ -z "${task_name}" || "${task_name}" == \#* ]] && continue
    if [[ ! "${task_name}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
      echo "[ROBODOJO CLIENT] invalid task name in ${task_set_path}: ${task_name}" >&2
      return 1
    fi
    task_names+=("${task_name}")
  done < "${task_set_path}"

  if (( ${#task_names[@]} == 0 )); then
    echo "[ROBODOJO CLIENT] task set is empty: ${task_set_path}" >&2
    return 1
  fi

  TASK_SET_NAME="${task_set_name}"
  TASK_NAMES="${task_names[*]}"
  export TASK_SET_NAME TASK_NAMES
}
