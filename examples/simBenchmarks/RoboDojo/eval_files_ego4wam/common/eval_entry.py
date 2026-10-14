#!/usr/bin/env python3
# Copyright (c) 2026 Beijing Horizon Information Technology Company Limited.
# Author: Ego4WAM Team.
# SPDX-License-Identifier: Apache-2.0

"""Run RoboDojo evaluation with an environment-controlled batch mode."""

from __future__ import annotations

import os
from copy import deepcopy

import numpy as np


class _EnvOriginObservationAdapter:
    """Attach each replicated environment's world origin to its observation.

    RoboDojo camera extrinsics are expressed in the global Isaac scene.  The
    Ego4WAM policy computes TCP positions in the per-environment frame,
    so its proxy must first remove the translation used to lay replicated
    environments out on the simulation grid.
    """

    def __init__(self, task_env) -> None:
        self._task_env = task_env

    def __getattr__(self, name):
        return getattr(self._task_env, name)

    def _attach(self, observation):
        if not isinstance(observation, dict):
            raise TypeError(
                "RoboDojo observation must be a dict before adding env_origin, " f"got {type(observation).__name__}"
            )
        if "env_idx" not in observation:
            raise KeyError("RoboDojo observation is missing env_idx")
        env_idx = int(observation["env_idx"])
        env_origins = getattr(self._task_env, "env_origins", None)
        if env_origins is None:
            raise AttributeError("RoboDojo TASK_ENV is missing env_origins")
        origin = env_origins[env_idx]
        if hasattr(origin, "detach"):
            origin = origin.detach()
        if hasattr(origin, "cpu"):
            origin = origin.cpu()
        if hasattr(origin, "numpy"):
            origin = origin.numpy()
        origin = np.asarray(origin, dtype=np.float32).reshape(-1)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError(f"RoboDojo env_origin[{env_idx}] must be one finite 3-vector, " f"got shape={origin.shape}")
        enriched = dict(observation)
        enriched["env_origin"] = np.ascontiguousarray(origin)
        return enriched

    def get_obs(self, *args, **kwargs):
        return self._attach(self._task_env.get_obs(*args, **kwargs))

    def get_obs_batch(self, *args, **kwargs):
        return [self._attach(observation) for observation in self._task_env.get_obs_batch(*args, **kwargs)]


def _patch_ego4wam_eval_with_env_origins() -> None:
    """Patch only this process's stock deploy loop for the isolated Ego4WAM proxy."""

    from XPolicyLab.policy.Ego4WAM import deploy as ego4wam_deploy

    if getattr(ego4wam_deploy, "_ego4wam_env_origin_patch", False):
        return
    for function_name in ("eval_one_episode", "eval_one_episode_batch"):
        original = getattr(ego4wam_deploy, function_name)

        def wrapped(TASK_ENV, model_client, _original=original):
            return _original(
                TASK_ENV=_EnvOriginObservationAdapter(TASK_ENV),
                model_client=model_client,
            )

        setattr(ego4wam_deploy, function_name, wrapped)
    ego4wam_deploy._ego4wam_env_origin_patch = True


def parse_eval_batch(value: str | None) -> bool | None:
    if value in (None, ""):
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"ROBODOJO_EVAL_BATCH must be a boolean, got {value!r}")


def main() -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "policy_proxy_raw"))
    from proxy_raw_bootstrap import register_policy

    register_policy()
    from src.eval_client import main as robodojo_main

    if os.getenv("ROBODOJO_REQUIRE_HEAD_EXTRINSIC", "0") == "1":
        # Do not edit the shared RoboDojo image's env_cfg/arx_x5.yml.  Patch
        # only the in-memory eval config for TCP-conditioned state proxies, before
        # RoboDojo constructs ObsManager.
        original_load_yaml = robodojo_main.load_yaml

        def load_yaml_with_head_extrinsic(path, *args, **kwargs):
            config = original_load_yaml(path, *args, **kwargs)
            if isinstance(config, dict) and "observation" in config:
                config = deepcopy(config)
                vision = config.setdefault("observation", {}).setdefault("vision", {})
                vision["extrinsic_matrix"] = True
            return config

        robodojo_main.load_yaml = load_yaml_with_head_extrinsic
        _patch_ego4wam_eval_with_env_origins()

    eval_batch = parse_eval_batch(os.getenv("ROBODOJO_EVAL_BATCH"))
    if eval_batch is not None:
        robodojo_main._eval_batch_from_deploy = lambda _policy_name: eval_batch
    robodojo_main.main()


if __name__ == "__main__":
    main()
