#!/usr/bin/env bash
# Copyright (c) 2026 Ego4WAM Authors.
# SPDX-License-Identifier: Apache-2.0
# Source this script. The simulator must use the checked historical forks,
# not a different globally installed IsaacLab or cuRobo version.
: "${ROBODOJO_ROOT:?Set ROBODOJO_ROOT to the pinned source checkout}"
ego4wam_source_paths="${ROBODOJO_ROOT}/third_party/curobo"
for ego4wam_extension in isaaclab isaaclab_assets isaaclab_tasks isaaclab_rl isaaclab_mimic; do
  ego4wam_extension_path="${ROBODOJO_ROOT}/third_party/IsaacLab/source/${ego4wam_extension}"
  if [[ ! -d "${ego4wam_extension_path}" ]]; then
    echo "Missing historical source package: ${ego4wam_extension_path}" >&2
    return 1
  fi
  ego4wam_source_paths="${ego4wam_extension_path}:${ego4wam_source_paths}"
done
export PYTHONPATH="${ego4wam_source_paths}${PYTHONPATH:+:${PYTHONPATH}}"
unset ego4wam_source_paths ego4wam_extension ego4wam_extension_path
