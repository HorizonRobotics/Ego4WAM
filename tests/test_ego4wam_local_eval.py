# Copyright (c) 2026 Beijing Horizon Information Technology Company Limited.
# Author: Ego4WAM Team.
# SPDX-License-Identifier: MIT
"""CPU checks for the historical local evaluation plumbing, not its success rate."""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from deployment.model_server.tools.ego4wam_websocket_policy_server import Ego4WAMWebsocketPolicyServer

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "examples/simBenchmarks/RoboDojo/eval_files_ego4wam"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FakePolicy:
    def __init__(self):
        self.calls = []

    def predict_action(self, examples, unnorm_key):
        self.calls.append((list(examples), unnorm_key))
        return {"actions": np.asarray(examples)[:, None, None]}


def test_microbatch_routes_multiple_requests_without_mixing_results():
    async def run():
        policy = FakePolicy()
        server = Ego4WAMWebsocketPolicyServer(policy, max_batch_size=4, batch_timeout_ms=5)
        server._pending_condition = asyncio.Condition()
        dispatcher = asyncio.create_task(server._batch_dispatcher())
        requests = [
            {"request_id": "first", "examples": [1, 2], "unnorm_key": "a"},
            {"request_id": "second", "examples": [3], "unnorm_key": "b"},
            {"request_id": "third", "examples": [4], "unnorm_key": "a"},
        ]
        try:
            results = await asyncio.gather(*(server._route_message(message) for message in requests))
        finally:
            dispatcher.cancel()
            with pytest.raises(asyncio.CancelledError):
                await dispatcher
        assert [item["request_id"] for item in results] == ["first", "second", "third"]
        assert [item["data"]["actions"].flatten().tolist() for item in results] == [[1, 2], [3], [4]]
        assert policy.calls == [([1, 2, 4], "a"), ([3], "b")]

    asyncio.run(run())


def test_microbatch_rejects_oversized_request_without_running_policy():
    policy = FakePolicy()
    server = Ego4WAMWebsocketPolicyServer(policy, max_batch_size=2)
    response = asyncio.run(server._route_message({"examples": [1, 2, 3], "unnorm_key": "a"}))
    assert not response["ok"]
    assert policy.calls == []


@pytest.mark.parametrize("examples", [None, [], "invalid"])
def test_microbatch_invalid_examples_cannot_bypass_the_dispatcher(examples):
    policy = FakePolicy()
    server = Ego4WAMWebsocketPolicyServer(policy)
    response = asyncio.run(server._route_message({"examples": examples}))
    assert not response["ok"]
    assert not policy.calls


def test_episode_coverage_and_metrics_must_agree():
    module = load_module("ego4wam_result_validation", EVAL / "common/result_validation.py")
    result = {
        "eval_time": 2,
        "success_rate": 0.5,
        "score": 65.0,
        "details": {
            "0": {"layout_id": 0, "success": True, "score": 1.0},
            "1": {"layout_id": 1, "success": False, "score": 0.3},
        },
    }
    assert module.validate_result(result, 2)["successes"] == 1
    with pytest.raises(ValueError, match="coverage"):
        module.validate_result(result, 50)
    for key, value in [("success_rate", 0.0), ("score", float("nan"))]:
        changed = copy.deepcopy(result)
        changed[key] = value
        with pytest.raises(ValueError):
            module.validate_result(changed, 2)
    changed = copy.deepcopy(result)
    changed["details"]["1"]["layout_id"] = 0
    with pytest.raises(ValueError, match="duplicate"):
        module.validate_result(changed, 2)


def test_video_coverage_requires_three_nonempty_files_per_episode(tmp_path):
    client = load_module("ego4wam_video_validation", EVAL / "common/run_client.py")
    episode = tmp_path / "build_tower/episode_0000000"
    episode.mkdir(parents=True)
    for filename in client.EXPECTED_VIDEO_FILENAMES:
        (episode / filename).write_bytes(b"video")
    assert len(client.collect_task_videos(tmp_path, "build_tower", 1)) == 3
    with pytest.raises(RuntimeError, match="Expected videos"):
        client.collect_task_videos(tmp_path, "build_tower", 2)
    (episode / next(iter(client.EXPECTED_VIDEO_FILENAMES))).write_bytes(b"")
    with pytest.raises(RuntimeError, match="empty"):
        client.collect_task_videos(tmp_path, "build_tower", 1)


def test_microbatch_inference_errors_stay_with_their_request():
    class BrokenPolicy:
        def predict_action(self, **kwargs):
            raise ValueError("bad observation")

    server = Ego4WAMWebsocketPolicyServer(BrokenPolicy())
    result = asyncio.run(server._route_message({"request_id": "bad", "examples": [1]}))
    assert result["request_id"] == "bad"
    assert not result["ok"]
    assert result["error"]["message"] == "bad observation"


def test_summary_keeps_dimension_balancing():
    module = load_module("ego4wam_summary", EVAL / "common/summarize_results.py")
    dimensions, success_rate, score = module.summarize_dimensions(
        {
            "a": {"dimension": "precision", "success_rate": 1.0, "score": 100.0},
            "b": {"dimension": "precision", "success_rate": 0.0, "score": 0.0},
            "c": {"dimension": "open", "success_rate": 0.0, "score": 0.0},
        }
    )
    assert dimensions["precision"]["tasks"] == 2
    assert success_rate == 0.25
    assert score == 25.0


def test_env_origin_adapter_keeps_each_environment_separate():
    module = load_module("ego4wam_entry", EVAL / "common/eval_entry.py")

    class Environment:
        env_origins = np.asarray([[0, 0, 0], [10, 20, 0]], dtype=np.float32)

        def get_obs_batch(self):
            return [{"env_idx": 1}, {"env_idx": 0}]

    result = module._EnvOriginObservationAdapter(Environment()).get_obs_batch()
    assert result[0]["env_origin"].tolist() == [10, 20, 0]
    assert result[1]["env_origin"].tolist() == [0, 0, 0]


def test_public_local_defaults_and_no_newer_compensation():
    client = load_module("ego4wam_client", EVAL / "common/run_client.py")
    assert client.SUPPORTED_ACTION_ENCODINGS == {"joints_eef_32_abs"}
    server_script = (EVAL / "local/local_server.sh").read_text()
    assert '${MAX_BATCH_SIZE:-10}' in server_script
    assert '${BATCH_TIMEOUT_MS:-20}' in server_script
    model = (EVAL / "policy_proxy_raw/policy_modules/Ego4WAM/model.py").read_text()
    assert "previous_world_to_camera_by_env" in model  # Original geometry convention.
    assert "self._infer_chunks(envs_needing_chunk)" in model  # Original task batch.
    for source in [model, (EVAL / "policy_proxy_raw/policy_modules/Ego4WAM/deploy.py").read_text()]:
        assert "sampler_seed" not in source
        assert "oneframe" not in source
        assert "frame_cache" not in source
    task_names = (EVAL / "common/task_sets/FULL_TASKS.txt").read_text().splitlines()
    task_names = [name for name in task_names if name and not name.startswith("#")]
    assert len(task_names) == 54
    assert len({client.canonical_task_name(name) for name in task_names}) == 42


def test_source_paths_prepend_the_pinned_forks(tmp_path):
    root = tmp_path / "RoboDojo"
    extensions = ["isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl", "isaaclab_mimic"]
    for extension in extensions:
        (root / f"third_party/IsaacLab/source/{extension}").mkdir(parents=True)
    (root / "third_party/curobo").mkdir(parents=True)
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; printf "%s" "$PYTHONPATH"', "source-test", str(EVAL / "common/source_paths.sh")],
        env={**os.environ, "ROBODOJO_ROOT": str(root), "PYTHONPATH": "unrelated-site-packages"},
        check=True,
        capture_output=True,
        text=True,
    )
    entries = result.stdout.split(":")
    assert entries[-1] == "unrelated-site-packages"
    assert str(root / "third_party/curobo") in entries[:-1]
    for extension in extensions:
        assert str(root / f"third_party/IsaacLab/source/{extension}") in entries[:-1]
