# Copyright (c) 2026 Ego4WAM Authors.
# SPDX-License-Identifier: MIT
"""Validate evaluation coverage without changing the benchmark's scoring."""

from __future__ import annotations

import math


def validate_result(result: dict, expected_episodes: int) -> dict:
    """Reject partial, inconsistent or non-finite results before aggregation."""
    count = result.get("eval_time")
    if isinstance(count, bool) or not isinstance(count, int) or count != expected_episodes or count < 1:
        raise ValueError(f"Incomplete episode coverage: expected {expected_episodes}, got {count!r}")
    details = result.get("details")
    if not isinstance(details, dict) or set(details) != {str(index) for index in range(count)}:
        raise ValueError("Episode details must contain each completed episode exactly once")
    successes = 0
    total_score = 0.0
    layout_ids = set()
    for episode_id, detail in details.items():
        if not isinstance(detail, dict) or not isinstance(detail.get("success"), bool):
            raise ValueError(f"Invalid success flag in episode {episode_id}")
        layout_id = detail.get("layout_id")
        if isinstance(layout_id, bool) or not isinstance(layout_id, int) or layout_id in layout_ids:
            raise ValueError(f"Invalid or duplicate layout_id in episode {episode_id}: {layout_id!r}")
        layout_ids.add(layout_id)
        score = float(detail["score"])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"Invalid score in episode {episode_id}: {score}")
        successes += detail["success"]
        total_score += score
    success_rate = float(result["success_rate"])
    score = float(result["score"])
    if not math.isclose(success_rate, successes / count, abs_tol=1e-7):
        raise ValueError("Reported success rate disagrees with episode details")
    if not math.isclose(score, 100.0 * total_score / count, abs_tol=1e-5):
        raise ValueError("Reported score disagrees with episode details")
    return {"episodes": count, "successes": successes, "success_rate": success_rate, "score": score}
