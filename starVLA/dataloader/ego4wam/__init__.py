# Copyright (c) 2026 Beijing Horizon Information Technology Company Limited.
# Author: Ego4WAM Team.
# SPDX-License-Identifier: MIT

"""Fixed data contracts used by Ego4WAM."""

__all__ = ["Ego4WAMRoboDojoDataset"]


def __getattr__(name):
    # Online FK/video preprocessing does not require the LeRobot training stack.
    if name == "Ego4WAMRoboDojoDataset":
        from .robodojo_dataset import Ego4WAMRoboDojoDataset

        globals()[name] = Ego4WAMRoboDojoDataset
        return Ego4WAMRoboDojoDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
