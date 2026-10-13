# Copyright (c) 2026 Ego4WAM Authors.
# SPDX-License-Identifier: Apache-2.0
"""Recipe, initialization, optimizer and release-format regression checks."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch import nn

from starVLA.training.trainer_utils import ego4wam_training as training

ROOT = Path(__file__).resolve().parents[1]
RECIPES = ROOT / "examples/simBenchmarks/RoboDojo/train_files_ego4wam"


def test_training_directories_keep_starvla_and_ego4wam_separate():
    original = ROOT / "examples/simBenchmarks/RoboDojo/train_files"
    assert (original / "data_registry/data_config.py").is_file()
    assert (original / "modality.json").is_file()
    assert len(list(original.glob("starvla_*.yaml"))) == 3
    assert not list(original.glob("*ego4wam*"))
    assert not list(RECIPES.glob("starvla_*.yaml"))
    assert not (RECIPES / "data_registry").exists()
    assert not (RECIPES / "modality.json").exists()
    for directory, entrypoint in ((original, "train_starvla.py"), (RECIPES, "train_ego4wam.py")):
        assert entrypoint in (directory / "run_robodojo_train.sh").read_text()


def test_copied_launcher_and_original_have_independent_defaults(tmp_path):
    meta = tmp_path / "RoboDojo_lerobot_v21_video/meta"
    meta.mkdir(parents=True)
    for name in ("info.json", "modality.json"):
        (meta / name).write_text("{}")
    env = {**os.environ, "robodojo_data_root": str(tmp_path), "DRY_RUN": "1", "NUM_PROCESSES": "8"}
    for name in ("config_yaml", "training_entrypoint", "accelerate_config"):
        env.pop(name, None)
    for directory, entrypoint, config in (
        ("train_files", "train_starvla.py", "starvla_robodojo_v21_qwenoft_h50_q99.yaml"),
        ("train_files_ego4wam", "train_ego4wam.py", "ego4wam_wan_robodojo_joint.yaml"),
    ):
        script = ROOT / "examples/simBenchmarks/RoboDojo" / directory / "run_robodojo_train.sh"
        result = subprocess.run(["bash", str(script)], cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
        assert f"starVLA/training/{entrypoint}" in result.stdout
        assert f"{directory}/{config}" in result.stdout


class ConfigWriter:
    def save_pretrained(self, directory):
        self.save_config(directory)

    def save_config(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(json.dumps({"_name_or_path": "/internal/model/source"}))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.embed_tokens = nn.Embedding(4, 2)
        self.shared = self.encoder.embed_tokens
        self.config = ConfigWriter()


class TinyModel(nn.Module):
    def __init__(self, interaction_mode="base"):
        super().__init__()
        self.interaction_mode = interaction_mode
        self.backbone = nn.Module()
        self.backbone.video_dit = nn.Linear(2, 2)
        self.backbone.vae = nn.Linear(2, 2).requires_grad_(False)
        self.backbone.vae._vae = ConfigWriter()
        self.backbone.text_encoder = nn.Module()
        self.backbone.text_encoder._encoder = Encoder().requires_grad_(False)
        self.backbone.tokenizer = SimpleNamespace(_tokenizer=ConfigWriter())
        self.action_model = nn.Linear(2, 2)
        self.proprio_encoder = nn.Linear(2, 2)
        self.mot = SimpleNamespace(gradient_checkpoint=False)

    def forward(self, examples):
        device = self.action_model.weight.device
        values = torch.ones(len(examples), 2, device=device, dtype=self.action_model.weight.dtype)
        result = self.action_model(self.backbone.video_dit(values) + self.proprio_encoder(values))
        loss = result.float().square().mean()
        return {"action_loss": loss, "loss_metrics": {"video": float(loss.detach())}}


@pytest.mark.parametrize("source", ["wan", "video_pretrained"])
@pytest.mark.parametrize("mode", ["base", "joint"])
def test_four_recipes_only_change_initialization_and_interaction(source, mode, monkeypatch):
    monkeypatch.setenv("EGO4WAM_PRETRAINED_DIR", "/models/Ego4WAM_VideoPretrained")
    cfg = OmegaConf.load(RECIPES / f"ego4wam_{source}_robodojo_{mode}.yaml")
    assert OmegaConf.to_container(cfg.framework) == {"name": "Ego4WAM", "interaction_mode": mode}
    assert cfg.trainer.pretrained_checkpoint == (None if source == "wan" else "/models/Ego4WAM_VideoPretrained")
    assert cfg.trainer.max_train_steps == 40000
    assert cfg.trainer.num_warmup_steps == 4000
    assert cfg.datasets.vla_data.per_device_batch_size == 16
    assert cfg.datasets.vla_data.balance_trajectory_weights is False
    configs = []
    for other_source in ("wan", "video_pretrained"):
        for other_mode in ("base", "joint"):
            other = OmegaConf.to_container(
                OmegaConf.load(RECIPES / f"ego4wam_{other_source}_robodojo_{other_mode}.yaml"), resolve=True
            )
            other.pop("run_id")
            other["framework"].pop("interaction_mode")
            other["trainer"].pop("pretrained_checkpoint")
            configs.append(other)
    assert all(item == configs[0] for item in configs)


@pytest.mark.parametrize("source", ["wan", "video_pretrained"])
@pytest.mark.parametrize("mode", ["base", "joint"])
def test_initialization_uses_existing_strict_loader(source, mode, tmp_path, monkeypatch):
    monkeypatch.setenv("EGO4WAM_PRETRAINED_DIR", str(tmp_path))
    cfg = OmegaConf.load(RECIPES / f"ego4wam_{source}_robodojo_{mode}.yaml")
    calls = []

    def strict_loader(weights, config_overrides):
        calls.append((weights, config_overrides))
        return TinyModel(mode)

    monkeypatch.setattr(training.baseframework, "from_pretrained", strict_loader)
    monkeypatch.setattr(training, "build_framework", lambda cfg: TinyModel(cfg.framework.interaction_mode))
    model = training.initialize_model(cfg)
    assert bool(calls) == (source == "video_pretrained")
    if calls:
        assert calls == [(str(tmp_path / "model.safetensors"), [f"framework.interaction_mode={mode}"])]
    assert model.interaction_mode == mode
    assert model.mot.gradient_checkpoint is True
    assert model.action_model.weight.dtype == torch.float32
    assert model.backbone.vae.weight.dtype == torch.bfloat16
    assert not model.backbone.vae.weight.requires_grad


def test_optimizer_covers_only_trainable_parameters_and_excludes_bias_decay():
    model = TinyModel()
    cfg = OmegaConf.load(RECIPES / "ego4wam_wan_robodojo_base.yaml")
    groups = training.optimizer_groups(model, cfg)
    selected = [id(parameter) for group in groups for parameter in group["params"]]
    expected = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
    assert len(selected) == len(set(selected))
    assert set(selected) == set(expected)
    for group in groups:
        assert group["lr"] == 1e-4
        assert all(parameter.ndim == 1 for parameter in group["params"]) == (group["weight_decay"] == 0)


@pytest.mark.parametrize("mode", ["base", "joint"])
def test_bundle_export_is_flat_and_restores_tied_alias(mode, tmp_path):
    model = TinyModel(mode)
    output = tmp_path / "bundle"
    training.save_model_bundle(model, model.state_dict(), output)
    assert {path.name for path in output.iterdir()} == {
        "README.md",
        "LICENSE",
        "inference_config.yaml",
        "model.safetensors",
        "model_assets",
    }
    cfg = OmegaConf.load(output / "inference_config.yaml")
    assert set(cfg.framework) == {"name", "interaction_mode"}
    assert cfg.framework.interaction_mode == mode
    for path in (output / "model_assets").rglob("*.json"):
        assert "/internal/model/source" not in path.read_text()
    state = load_file(output / "model.safetensors")
    assert "backbone.text_encoder._encoder.shared.weight" not in state
    from starVLA.model.framework.WM4A.Ego4WAM import Ego4WAMFramework

    state = Ego4WAMFramework.prepare_checkpoint_state_dict(state)
    model.to(dtype=torch.bfloat16).load_state_dict(state, strict=True)
    assert all(parameter.dtype == torch.bfloat16 for parameter in state.values())
    with pytest.raises(FileExistsError):
        training.save_model_bundle(model, model.state_dict(), output)


@pytest.mark.parametrize("source", ["wan", "video_pretrained"])
@pytest.mark.parametrize("mode", ["base", "joint"])
def test_shell_recipes_dry_run_from_outside_repo(source, mode, tmp_path, monkeypatch):
    data = tmp_path / "datasets/RoboDojo_lerobot_v21_video/meta"
    data.mkdir(parents=True)
    (data / "info.json").write_text(json.dumps({}))
    pretrained = tmp_path / "pretrained"
    pretrained.mkdir()
    for item in ("model.safetensors", "inference_config.yaml"):
        (pretrained / item).write_text("test")
    (pretrained / "model_assets").mkdir()
    monkeypatch.setenv("EGO4WAM_PRETRAINED_DIR", str(pretrained))
    monkeypatch.setenv("robodojo_data_root", str(tmp_path / "datasets"))
    monkeypatch.setenv("DRY_RUN", "1")
    monkeypatch.setenv("NUM_PROCESSES", "8")
    result = subprocess.run(
        ["bash", str(RECIPES / f"train_ego4wam_{source}_robodojo_{mode}.sh"), "--trainer.max_train_steps", "2"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "starVLA/training/train_ego4wam.py" in result.stdout
    assert f"ego4wam_{source}_robodojo_{mode}.yaml" in result.stdout
    assert "--trainer.max_train_steps 2" in result.stdout
    assert "ego4wam_deepspeed_zero2.yaml" in result.stdout


def test_local_wan_assets_do_not_call_huggingface(tmp_path):
    from starVLA.model.modules.world_model.WanMoT import _maybe_download

    assert _maybe_download(str(tmp_path), ["transformer/*.safetensors"]) == str(tmp_path)


def test_deepspeed_config_has_single_mixed_precision_source():
    cfg = OmegaConf.load(RECIPES / "ego4wam_deepspeed_zero2.yaml")
    assert "mixed_precision" not in cfg
    ds = json.loads((ROOT / cfg.deepspeed_config.deepspeed_config_file).read_text())
    assert ds["bf16"]["enabled"] is True
    assert ds["fp16"]["enabled"] is False
    assert ds["gradient_accumulation_steps"] == "auto"
    assert ds["gradient_clipping"] == "auto"
    assert ds["zero_optimization"]["stage"] == 2


@pytest.mark.parametrize("source", ["wan", "video_pretrained"])
@pytest.mark.parametrize("mode", ["base", "joint"])
def test_tiny_training_loop_accumulation_checkpoint_and_resume(source, mode, tmp_path):
    # Separate CPU subprocesses avoid sharing Accelerate's singleton state or
    # occupying evaluation GPUs. This exercises the real trainer loop, not the
    # full-size architecture; the latter has separate CUDA/math/load tests.
    code = (
        """
import importlib.util
import json
from pathlib import Path
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, Dataset
from starVLA.training import train_ego4wam as trainer
spec = importlib.util.spec_from_file_location("training_fixtures", "TEST_FILE")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
class Data(Dataset):
    def __len__(self): return 8
    def __getitem__(self, i): return {"index": i}
    def set_epoch(self, epoch): self.epoch = epoch
cfg = OmegaConf.load("RECIPE_FILE")
cfg.run_root_dir = "OUTPUT_DIR"
cfg.trainer.max_train_steps = 2
cfg.trainer.num_warmup_steps = 0
cfg.trainer.save_interval = 1
cfg.trainer.logging_frequency = 1
cfg.trainer.gradient_accumulation_steps = 2
cfg.datasets.vla_data.per_device_batch_size = 1
cfg.datasets.vla_data.data_root_dir = "OUTPUT_DIR"
trainer.initialize_model = lambda cfg: fixtures.TinyModel(cfg.framework.interaction_mode)
trainer.create_dataloader = lambda cfg: DataLoader(Data(), batch_size=1, collate_fn=list)
trainer.run(cfg)
root = Path(cfg.run_root_dir) / cfg.run_id
metadata = json.loads((root / "checkpoints/steps_2/training_state.json").read_text())
assert metadata["completed_steps"] == 2
assert metadata["batches_consumed"] == 4
assert metadata["scheduler"]["last_epoch"] == 2
from safetensors.torch import load_file
assert (root / "final_model/model.safetensors").is_file()
cfg.trainer.resume_from_checkpoint = str(root / "checkpoints/steps_2")
cfg.run_id += "_resume"
cfg.trainer.max_train_steps = 3
trainer.run(cfg)
resumed = Path(cfg.run_root_dir) / cfg.run_id
metadata = json.loads((resumed / "checkpoints/steps_3/training_state.json").read_text())
assert metadata["completed_steps"] == 3
assert metadata["batches_consumed"] == 6
assert metadata["scheduler"]["last_epoch"] == 3
assert (resumed / "final_model/model.safetensors").is_file()
""".replace(
            "TEST_FILE", str(Path(__file__).resolve())
        )
        .replace("RECIPE_FILE", str(RECIPES / f"ego4wam_{source}_robodojo_{mode}.yaml"))
        .replace("OUTPUT_DIR", str(tmp_path))
    )
    env = {
        **os.environ,
        "STARVLA_DISABLE_DEEPSPEED": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "EGO4WAM_PRETRAINED_DIR": str(tmp_path),
    }
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
