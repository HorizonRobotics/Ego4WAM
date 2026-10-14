# Copyright (c) StarVLA Team.
# Modifications Copyright (c) 2026 Beijing Horizon Information Technology Company Limited.
# Author: Ego4WAM Team.
# SPDX-License-Identifier: Apache-2.0 AND MIT
# Ego4WAM additions/modifications: Apache-2.0; upstream StarVLA portions: MIT.
# See LICENSES/StarVLA-MIT.txt for the retained upstream license.

import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import torch
from omegaconf import OmegaConf

from starVLA.model.framework import base_framework


def _checkpoint_config(use_canonical_forward=True):
    return {
        "framework": {
            "name": "DummyFramework",
            "action_model": {
                "action_horizon": 8,
                "diffusion_model_cfg": {
                    "use_canonical_forward": use_canonical_forward,
                    "num_inference_timesteps": 4,
                },
            },
        },
        "trainer": {"pretrained_checkpoint": "original.pt"},
        "datasets": {"vla_data": {"data_mix": "dummy_mix"}},
    }


class _FakeFramework:
    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict, strict=True):
        self.loaded_state_dict = state_dict
        self.loaded_strict = strict



class ConfigOverrideLoadTest(unittest.TestCase):
    def test_flat_inference_artifact_needs_no_dataset_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            checkpoint_path = artifact_dir / "model.safetensors"
            checkpoint_path.write_bytes(b"placeholder")
            (artifact_dir / "inference_config.yaml").write_text(
                textwrap.dedent("""
                    framework:
                      name: DummyFramework
                    trainer:
                      pretrained_checkpoint: null
                    """).lstrip()
            )

            config, norm_stats = base_framework.read_mode_config(checkpoint_path)

            self.assertEqual(config["framework"]["name"], "DummyFramework")
            self.assertEqual(norm_stats, {})

    def _load_with_config(self, config, overrides=None):
        captured = {}

        def fake_build_framework(cfg):
            captured["cfg"] = cfg
            return _FakeFramework()

        with (
            mock.patch.object(base_framework, "read_mode_config", return_value=(config, {"stats": {}})),
            mock.patch.object(base_framework, "build_framework", side_effect=fake_build_framework),
            mock.patch.object(torch, "load", return_value={}),
        ):
            model = base_framework.baseframework.from_pretrained(
                "/tmp/run/checkpoints/steps_1_pytorch_model.pt",
                config_overrides=overrides,
            )
        return model, captured["cfg"]

    def test_from_pretrained_without_overrides_preserves_checkpoint_config(self):
        _, cfg = self._load_with_config(_checkpoint_config(use_canonical_forward=True))

        self.assertIs(
            OmegaConf.select(cfg, "framework.action_model.diffusion_model_cfg.use_canonical_forward"),
            True,
        )

    def test_false_override_resolves_to_bool_false_before_build_framework(self):
        _, cfg = self._load_with_config(
            _checkpoint_config(use_canonical_forward=True),
            ["framework.action_model.diffusion_model_cfg.use_canonical_forward=false"],
        )

        resolved = OmegaConf.select(cfg, "framework.action_model.diffusion_model_cfg.use_canonical_forward")
        self.assertIs(resolved, False)

    def test_override_has_precedence_over_checkpoint_yaml(self):
        _, cfg = self._load_with_config(
            _checkpoint_config(use_canonical_forward=True),
            ["framework.action_model.diffusion_model_cfg.num_inference_timesteps=6"],
        )

        self.assertEqual(
            OmegaConf.select(cfg, "framework.action_model.diffusion_model_cfg.num_inference_timesteps"),
            6,
        )

    def test_repeatable_overrides_are_accepted(self):
        _, cfg = self._load_with_config(
            _checkpoint_config(use_canonical_forward=True),
            (
                "framework.action_model.diffusion_model_cfg.use_canonical_forward=false",
                "framework.action_model.diffusion_model_cfg.num_inference_timesteps=6",
            ),
        )

        self.assertIs(
            OmegaConf.select(cfg, "framework.action_model.diffusion_model_cfg.use_canonical_forward"),
            False,
        )
        self.assertEqual(
            OmegaConf.select(cfg, "framework.action_model.diffusion_model_cfg.num_inference_timesteps"),
            6,
        )

    def test_duplicate_override_later_value_wins(self):
        _, cfg = self._load_with_config(
            _checkpoint_config(use_canonical_forward=True),
            [
                "framework.action_model.diffusion_model_cfg.use_canonical_forward=true",
                "framework.action_model.diffusion_model_cfg.use_canonical_forward=false",
            ],
        )

        self.assertIs(
            OmegaConf.select(cfg, "framework.action_model.diffusion_model_cfg.use_canonical_forward"),
            False,
        )

    def test_invalid_override_syntax_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "Expected KEY=VALUE"):
            self._load_with_config(_checkpoint_config(), ["framework.action_model.bad_override"])

    def test_bare_string_config_overrides_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "sequence of KEY=VALUE strings"):
            self._load_with_config(
                _checkpoint_config(),
                "framework.action_model.diffusion_model_cfg.use_canonical_forward=false",
            )

    def test_empty_string_config_overrides_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "sequence of KEY=VALUE strings"):
            self._load_with_config(_checkpoint_config(), "")

    def test_empty_list_config_overrides_is_valid_noop(self):
        _, cfg = self._load_with_config(_checkpoint_config(use_canonical_forward=True), [])

        self.assertIs(
            OmegaConf.select(cfg, "framework.action_model.diffusion_model_cfg.use_canonical_forward"),
            True,
        )

    def test_empty_tuple_config_overrides_is_valid_noop(self):
        _, cfg = self._load_with_config(_checkpoint_config(use_canonical_forward=True), ())

        self.assertIs(
            OmegaConf.select(cfg, "framework.action_model.diffusion_model_cfg.use_canonical_forward"),
            True,
        )

    def test_unrelated_interpolation_is_not_resolved_when_override_applies(self):
        config = _checkpoint_config()
        config["run_root_dir"] = "/tmp/root"
        config["run_id"] = "run_a"
        config["output_dir"] = "${run_root_dir}/${run_id}"

        merged = base_framework.merge_config_overrides(
            config,
            ["framework.action_model.diffusion_model_cfg.use_canonical_forward=false"],
        )

        self.assertEqual(merged["output_dir"], "${run_root_dir}/${run_id}")

    def test_checkpoint_config_file_is_not_modified(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            ckpt_dir = run_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True)
            ckpt_path = ckpt_dir / "steps_1_pytorch_model.pt"
            ckpt_path.write_bytes(b"placeholder")
            config_path = run_dir / "config.yaml"
            config_text = textwrap.dedent("""
                framework:
                  name: DummyFramework
                  action_model:
                    action_horizon: 8
                    diffusion_model_cfg:
                      use_canonical_forward: true
                trainer:
                  pretrained_checkpoint: original.pt
                datasets:
                  vla_data:
                    data_mix: dummy_mix
                """).lstrip()
            config_path.write_text(config_text)
            (run_dir / "dataset_statistics.json").write_text("{}")

            with (
                mock.patch.object(base_framework, "build_framework", return_value=_FakeFramework()),
                mock.patch.object(torch, "load", return_value={}),
            ):
                base_framework.baseframework.from_pretrained(
                    str(ckpt_path),
                    config_overrides=["framework.action_model.diffusion_model_cfg.use_canonical_forward=false"],
                )

            self.assertEqual(config_path.read_text(), config_text)
