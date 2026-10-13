# Copyright (c) 2026 Ego4WAM Authors.
# SPDX-License-Identifier: Apache-2.0
"""Source-level guards for Ego4WAM's deliberately small public config."""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_SOURCE = REPO_ROOT / "starVLA/model/framework/WM4A/Ego4WAM.py"


def _framework_tree() -> ast.Module:
    return ast.parse(FRAMEWORK_SOURCE.read_text(encoding="utf-8"))


def test_ego4wam_public_config_only_exposes_interaction_mode() -> None:
    tree = _framework_tree()
    config_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Ego4WAMDefaultConfig"
    )
    fields = {
        node.target.id
        for node in config_class.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert fields == {"name", "interaction_mode"}


def test_ego4wam_rejects_removed_framework_config_names() -> None:
    source = FRAMEWORK_SOURCE.read_text(encoding="utf-8")
    removed = {
        "variant",
        "encoder_type",
        "training_mode",
        "world_model",
        "action_model",
        "codec",
        "text",
        "mot",
        "scheduler",
        "loss",
        "profiling",
        "proprio",
        "action",
        "video",
        "camera_layout",
        "action_encoding",
    }
    for name in removed:
        assert f"fw.{name}" not in source
        assert f"framework.{name}" not in source


def test_ego4wam_has_one_registered_framework_name() -> None:
    tree = _framework_tree()
    registrations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        function = node.func
        if isinstance(function, ast.Attribute) and function.attr == "register":
            argument = node.args[0]
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                registrations.append(argument.value)
    assert registrations == ["Ego4WAM"]


def test_base_and_joint_only_change_action_to_video_interaction() -> None:
    tree = _framework_tree()

    def rules(name: str) -> list[tuple[str, str, str]]:
        assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        )
        assert isinstance(assignment.value, (ast.Tuple, ast.List))
        return [
            tuple(ast.literal_eval(argument) for argument in element.args)
            for element in assignment.value.elts
            if isinstance(element, ast.Call)
        ]

    base = rules("_BASE_INTERACTION")
    joint = rules("_JOINT_INTERACTION")
    assert base == [
        ("video.video", "video.video", "first_frame_causal"),
        ("action.action", "video.video", "key_first_frame"),
        ("action.action", "action.action", "full"),
    ]
    assert joint == [
        ("video.video", "video.video", "first_frame_causal"),
        ("action.action", "video.video", "full"),
        ("action.action", "action.action", "full"),
    ]
