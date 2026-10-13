#!/usr/bin/env python3
# Copyright (c) 2026 Ego4WAM Authors.
# SPDX-License-Identifier: MIT

"""Merge RoboDojo configs and compute historical dimension-balanced metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.simBenchmarks.RoboDojo.eval_files_ego4wam.common.result_validation import validate_result

OFFICIAL_DIMENSIONS = (
    "generalization",
    "precision",
    "long-horizon",
    "memory",
    "open",
)


def canonical_task_name(config_name: str) -> str:
    return config_name.removesuffix("_random")


def summarize_dimensions(task_results: dict[str, dict]) -> tuple[dict[str, dict], float, float]:
    dimension_values = {dimension: [] for dimension in OFFICIAL_DIMENSIONS}
    unknown_dimensions = set()
    for result in task_results.values():
        dimension = result["dimension"]
        if dimension not in dimension_values:
            unknown_dimensions.add(dimension)
            continue
        dimension_values[dimension].append(result)
    if unknown_dimensions:
        raise ValueError(f"unknown RoboDojo dimensions: {sorted(unknown_dimensions)}")

    by_dimension = {}
    for dimension in OFFICIAL_DIMENSIONS:
        results = dimension_values[dimension]
        if not results:
            continue
        by_dimension[dimension] = {
            "task_mean_success_rate": sum(result["success_rate"] for result in results) / len(results),
            "task_mean_score": sum(result["score"] for result in results) / len(results),
            "tasks": len(results),
        }

    overall_success_rate = (
        sum(result["task_mean_success_rate"] for result in by_dimension.values()) / len(by_dimension)
        if by_dimension
        else 0.0
    )
    overall_score = (
        sum(result["task_mean_score"] for result in by_dimension.values()) / len(by_dimension) if by_dimension else 0.0
    )
    return by_dimension, overall_success_rate, overall_score


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()

    manifest_path = args.root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_configs = manifest.get("configs", manifest.get("tasks", []))
    if not expected_configs:
        raise ValueError(f"manifest contains no RoboDojo configs: {manifest_path}")
    config_to_task = manifest.get(
        "config_to_task",
        {config_name: canonical_task_name(config_name) for config_name in expected_configs},
    )
    canonical_tasks = manifest.get("canonical_tasks", list(dict.fromkeys(config_to_task.values())))
    raw_task_dimensions = manifest.get("task_dimensions", {})
    config_dimensions = manifest.get("config_dimensions", {})
    task_configs = {task_name: [] for task_name in canonical_tasks}
    for config_name in expected_configs:
        task_name = config_to_task.get(config_name, canonical_task_name(config_name))
        task_configs.setdefault(task_name, []).append(config_name)
    task_dimensions = {}
    for task_name, config_names in task_configs.items():
        first_config = config_names[0]
        task_dimensions[task_name] = raw_task_dimensions.get(
            task_name,
            config_dimensions.get(first_config, raw_task_dimensions.get(first_config, "unknown")),
        )

    config_results = {}
    missing_configs = []
    invalid_configs = {}
    for config_name in expected_configs:
        result_path = args.root / "results" / f"{config_name}.json"
        if not result_path.is_file():
            missing_configs.append(config_name)
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        expected_episodes = manifest.get("task_weights", {}).get(config_name)
        if expected_episodes is None:
            raise ValueError(f"Missing expected episode count for {config_name} in the run manifest")
        try:
            coverage = validate_result(result, int(expected_episodes))
        except (KeyError, TypeError, ValueError) as exc:
            missing_configs.append(config_name)
            invalid_configs[config_name] = str(exc)
            continue
        success_rate = float(result["success_rate"])
        eval_time = int(result["eval_time"])
        config_results[config_name] = {
            "canonical_task_name": config_to_task.get(config_name, canonical_task_name(config_name)),
            "success_rate": success_rate,
            "eval_time": eval_time,
            "successes": coverage["successes"],
            "score": float(result.get("score", 0.0)),
        }

    task_results = {}
    missing_tasks = []
    total_episodes = 0

    for task_name, config_names in task_configs.items():
        if any(config_name not in config_results for config_name in config_names):
            missing_tasks.append(task_name)
            continue
        members = [config_results[config_name] for config_name in config_names]
        eval_time = sum(member["eval_time"] for member in members)
        successes = sum(member["successes"] for member in members)
        success_rate = successes / eval_time if eval_time else 0.0
        score = sum(member["score"] * member["eval_time"] for member in members) / eval_time if eval_time else 0.0
        dimension = task_dimensions.get(task_name, "unknown")
        task_results[task_name] = {
            "success_rate": success_rate,
            "eval_time": eval_time,
            "successes": successes,
            "score": score,
            "dimension": dimension,
            "configs": config_names,
        }
        total_episodes += eval_time

    by_dimension, overall_success_rate, overall_score = summarize_dimensions(task_results)
    summary = {
        "run_id": manifest["run_id"],
        "expected_tasks": len(task_configs),
        "completed_tasks": len(task_results),
        "missing_tasks": missing_tasks,
        "expected_configs": len(expected_configs),
        "completed_configs": len(config_results),
        "missing_configs": missing_configs,
        "invalid_configs": invalid_configs,
        "overall_success_rate": overall_success_rate,
        "overall_score": overall_score,
        "overall_aggregation": "dimension_balanced",
        "total_episodes": total_episodes,
        "by_dimension": by_dimension,
        "tasks": task_results,
        "configs": config_results,
    }
    (args.root / "aggregate.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        f"run_id={manifest['run_id']}",
        f"completed_tasks={len(task_results)}/{len(task_configs)}",
        f"completed_configs={len(config_results)}/{len(expected_configs)}",
        f"overall_success_rate={overall_success_rate:.4f}",
        f"overall_score={overall_score:.4f}",
        "overall_aggregation=dimension_balanced",
        f"total_episodes={total_episodes}",
    ]
    if missing_tasks:
        lines.append(f"missing_tasks={' '.join(missing_tasks)}")
    if missing_configs:
        lines.append(f"missing_configs={' '.join(missing_configs)}")
    lines.append("")
    for dimension, result in by_dimension.items():
        lines.append(
            f"{dimension}: success_rate={result['task_mean_success_rate']:.4f} "
            f"score={result['task_mean_score']:.4f} tasks={result['tasks']}"
        )
    lines.append("")
    for task_name, result in sorted(task_results.items()):
        lines.append(
            f"{task_name}: success_rate={result['success_rate']:.4f} "
            f"({result['successes']}/{result['eval_time']}) "
            f"score={result['score']:.4f} dimension={result['dimension']} "
            f"configs={','.join(result['configs'])}"
        )
    (args.root / "aggregate.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return 1 if missing_configs else 0


if __name__ == "__main__":
    raise SystemExit(main())
