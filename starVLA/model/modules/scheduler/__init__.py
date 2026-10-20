# Copyright (c) 2026 Beijing Horizon Information Technology Company Limited.
# Author: Ego4WAM Team.
# SPDX-License-Identifier: MIT
"""Scheduler modules (diffusion / flow-matching schedulers).

Built-in names:

- ``WanContinuousFlowMatchScheduler`` — continuous-time flow-matching scheduler
  for the FastWAM video/action experts (ported from the Wan2.2 reference).
"""

from __future__ import annotations

from .wan_scheduler import WanContinuousFlowMatchScheduler

__all__ = ["WanContinuousFlowMatchScheduler"]
