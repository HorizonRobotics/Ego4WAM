# Copyright (c) 2026 Beijing Horizon Information Technology Company Limited.
# Author: Ego4WAM Team.
# SPDX-License-Identifier: Apache-2.0
"""The trimmed release must remain runnable without unrelated StarVLA backends."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_release_has_only_ego4wam_framework_and_robodojo_examples():
    frameworks = ROOT / "starVLA/model/framework"
    assert {p.name for p in frameworks.glob("*/") if list(p.rglob("*.py"))} == {"WM4A"}
    assert {p.name for p in (frameworks / "WM4A").glob("*.py")} == {"__init__.py", "Ego4WAM.py"}
    recipes = ROOT / "examples/simBenchmarks/RoboDojo"
    assert {p.name for p in recipes.iterdir() if p.is_dir()} == {"eval_files_ego4wam", "train_files_ego4wam"}
    assert not list((ROOT / "examples/realRobots").rglob("*.py"))
    assert not list((ROOT / "examples/modelExtensions").rglob("*.py"))


def test_retained_internal_imports_resolve():
    def exists(module):
        path = ROOT.joinpath(*module.split("."))
        return path.with_suffix(".py").is_file() or path.is_dir()

    for directory in ("starVLA", "deployment", "examples"):
        for path in (ROOT / directory).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
                if isinstance(node, ast.ImportFrom) and not node.level and node.module:
                    names.append(node.module)
                for name in names:
                    if name.split(".")[0] in {"starVLA", "deployment"}:
                        assert exists(name), (path, name)


def test_registry_only_exposes_the_unchanged_ego4wam_contract():
    from starVLA.dataloader.gr00t_lerobot.registry import (
        DATASET_NAMED_MIXTURES,
        ROBOT_TYPE_CONFIG_MAP,
        ROBOT_TYPE_TO_EMBODIMENT_TAG,
    )

    assert DATASET_NAMED_MIXTURES == {
        "ego4wam_robodojo_v21": [("RoboDojo_lerobot_v21_video", 1.0, "ego4wam_robodojo_arx_x5")]
    }
    assert set(ROBOT_TYPE_CONFIG_MAP) == set(ROBOT_TYPE_TO_EMBODIMENT_TAG) == {"ego4wam_robodojo_arx_x5"}
    config = ROBOT_TYPE_CONFIG_MAP["ego4wam_robodojo_arx_x5"]
    modalities = config.modality_config()
    assert modalities["video"].delta_indices == list(range(0, 33, 4))
    assert modalities["action"].delta_indices == list(range(32))
    assert modalities["state"].delta_indices == [0]
    assert config.transform().transforms == []


def test_ego4wam_server_keeps_historical_batching_and_rng_defaults():
    from deployment.model_server import server_policy

    args = server_policy.build_argparser().parse_args(["--ckpt_path", "/models/Ego4WAM", "--robodojo-ego4wam"])
    assert args.seed is None
    assert args.max_batch_size == 10
    assert args.batch_timeout_ms == 20.0
    wrapper = SimpleNamespace(metadata={"action_chunk_size": 32, "server_preprocessing": True})
    with (
        mock.patch.object(server_policy, "_build_wrapper", return_value=wrapper),
        mock.patch.object(server_policy, "Ego4WAMWebsocketPolicyServer") as server,
        mock.patch.object(server_policy, "set_seed_everywhere") as seed,
    ):
        server_policy.main(args)
    seed.assert_not_called()
    server.return_value.serve_forever.assert_called_once_with()
    options = server.call_args.kwargs
    assert options["policy"] is wrapper
    assert options["max_batch_size"] == 10
    assert options["batch_timeout_ms"] == 20.0
    assert options["metadata"]["sampling_mode"] == "historical_global_rng"


def test_non_ego4wam_server_flags_are_not_silently_accepted():
    from deployment.model_server.server_policy import build_argparser

    with pytest.raises(SystemExit):
        build_argparser().parse_args(["--ckpt_path", "/models/Ego4WAM", "--use_bf16"])
