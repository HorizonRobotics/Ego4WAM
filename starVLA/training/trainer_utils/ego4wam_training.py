# Copyright (c) 2026 Ego4WAM Authors.
# SPDX-License-Identifier: Apache-2.0
"""Training plumbing for the fixed Ego4WAM model and released artifact format."""

import json
import shutil
from pathlib import Path

import torch
from omegaconf import OmegaConf

from starVLA.model.framework.base_framework import baseframework, build_framework


def initialize_model(cfg):
    """Wan initialization or strict full-model loading through the existing loader.

    The same video-pretrained Base bundle initializes either interaction mode.
    No action/state weights are dropped or reinitialized after loading.
    """
    if cfg.framework.name != "Ego4WAM" or cfg.framework.interaction_mode not in {"base", "joint"}:
        raise ValueError("Expected Ego4WAM with base/joint interaction_mode")
    if set(cfg.framework) != {"name", "interaction_mode"}:
        raise ValueError("Only framework.name and framework.interaction_mode are supported")
    source = cfg.trainer.get("pretrained_checkpoint")
    if source:
        path = Path(source).expanduser().resolve(strict=True)
        weights = path / "model.safetensors" if path.is_dir() else path
        model = baseframework.from_pretrained(
            str(weights),
            config_overrides=[f"framework.interaction_mode={cfg.framework.interaction_mode}"],
        )
        if getattr(model, "interaction_mode", None) != cfg.framework.interaction_mode:
            raise ValueError("Loaded checkpoint is not the requested Ego4WAM interaction mode")
    else:
        model = build_framework(cfg)
    # Frozen encoders keep BF16 storage. Standard AdamW keeps trainable master
    # parameters in FP32, with BF16 autocast/DeepSpeed for forward/backward.
    model.backbone.vae.to(dtype=torch.bfloat16)
    model.backbone.text_encoder.to(dtype=torch.bfloat16)
    for module in (model.backbone.video_dit, model.action_model, model.proprio_encoder):
        module.to(dtype=torch.float32)
    model.mot.gradient_checkpoint = bool(cfg.trainer.get("gradient_checkpointing", True))
    model.train()
    return model


def optimizer_groups(model, cfg):
    """Historical trainable prefixes, learning rates and bias/norm decay split."""
    rates = cfg.trainer.learning_rate
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        prefix = next(
            (key for key in ("backbone.video_dit", "action_model", "proprio_encoder") if name.startswith(key + ".")),
            None,
        )
        if prefix is None:
            raise ValueError(f"Unexpected trainable parameter: {name}")
        decay = float(cfg.trainer.optimizer.weight_decay)
        if parameter.ndim < 2 or name.endswith(".bias"):
            decay = 0.0
        key = (prefix, decay)
        group = groups.setdefault(
            key,
            {
                "params": [],
                "lr": float(rates.get(prefix, rates.base)),
                "weight_decay": decay,
                "name": f"{prefix}/{'decay' if decay else 'no_decay'}",
            },
        )
        group["params"].append(parameter)
    if not groups:
        raise ValueError("No trainable Ego4WAM parameters")
    return list(groups.values())


def save_model_bundle(model, state_dict, output):
    """Export exactly the same flat model package as the released HF weights.

    Caller gathers state on all ranks and invokes this only on the main rank.
    Write to a sibling staging directory and publish only after all files exist.
    Optimizer/training state belongs outside this bundle.
    """
    from safetensors.torch import save_file

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite model bundle: {output}")
    staging = output.with_name(output.name + ".exporting")
    staging.mkdir(parents=True, exist_ok=False)
    # The UMT5 alias is restored by prepare_checkpoint_state_dict in the one
    # canonical loader. It must not be serialized twice as shared storage.
    alias = "backbone.text_encoder._encoder.shared.weight"
    canonical = "backbone.text_encoder._encoder.encoder.embed_tokens.weight"
    if alias in state_dict:
        if canonical not in state_dict or not torch.equal(state_dict[alias], state_dict[canonical]):
            raise ValueError("UMT5 tied embedding aliases disagree")
    tensors = {
        name: value.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        for name, value in state_dict.items()
        if name != alias
    }
    save_file(tensors, str(staging / "model.safetensors"))
    assets = staging / "model_assets"
    model.backbone.vae._vae.save_config(str(assets / "vae"))
    model.backbone.text_encoder._encoder.config.save_pretrained(str(assets / "text_encoder"))
    model.backbone.tokenizer._tokenizer.save_pretrained(str(assets / "tokenizer"))
    # Library config writers can persist the absolute source checkpoint path.
    # These bookkeeping fields are not needed to load bundled local assets.
    for config_path in assets.rglob("*.json"):
        if config_path.name not in {"config.json", "tokenizer_config.json"}:
            continue
        config = json.loads(config_path.read_text())
        for key in ("_name_or_path", "name_or_path"):
            config.pop(key, None)
        if config.get("tokenizer_file") and Path(config["tokenizer_file"]).is_absolute():
            config.pop("tokenizer_file")
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    OmegaConf.save(
        OmegaConf.create(
            {
                "framework": {"name": "Ego4WAM", "interaction_mode": model.interaction_mode},
                "trainer": {"pretrained_checkpoint": None},
            }
        ),
        staging / "inference_config.yaml",
    )
    shutil.copyfile(Path(__file__).parent / "assets/Apache-2.0.txt", staging / "LICENSE")
    (staging / "README.md").write_text(
        "---\nlicense: apache-2.0\n---\n\n# Ego4WAM\n\n"
        f"RoboDojo posttrained weights; interaction mode: {model.interaction_mode}.\n",
        encoding="utf-8",
    )
    staging.rename(output)
