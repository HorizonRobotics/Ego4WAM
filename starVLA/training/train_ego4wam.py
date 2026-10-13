# Copyright (c) 2026 Ego4WAM Authors.
# SPDX-License-Identifier: Apache-2.0
"""RoboDojo posttraining with StarVLA's LeRobot adapter and Accelerate/ZeRO-2.

All four recipes share this loop and the existing strict checkpoint loader.
The framework's action_loss is the sum of video and action denoising losses.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from accelerate import Accelerator, DataLoaderConfiguration, DeepSpeedPlugin
from accelerate.utils import GradientAccumulationPlugin, set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import get_scheduler

from starVLA.training.trainer_utils.ego4wam_training import initialize_model, optimizer_groups, save_model_bundle
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args


def create_dataloader(cfg):
    # Lazy import: --help and model construction do not need training-only deps.
    from starVLA.dataloader.lerobot_datasets import collate_fn, get_vla_dataset

    data = cfg.datasets.vla_data
    if data.data_mix != "ego4wam_robodojo_v21":
        raise ValueError("These recipes require the historical Ego4WAM RoboDojo adapter")
    dataset = get_vla_dataset(
        data,
        balance_dataset_weights=False,
        balance_trajectory_weights=False,
        seed=int(cfg.seed),
    )
    workers = int(data.get("num_workers", 4))
    kwargs = {"prefetch_factor": 2} if workers else {}
    # New worker copies per epoch receive the mixture's updated epoch value.
    # Persistent workers would silently repeat the first epoch's seeded draws.
    return DataLoader(
        dataset,
        batch_size=int(data.per_device_batch_size),
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=collate_fn,
        **kwargs,
    )


def run(cfg):
    accumulation = int(cfg.trainer.gradient_accumulation_steps)
    if accumulation < 1 or int(cfg.trainer.max_train_steps) < 1:
        raise ValueError("Training and accumulation step counts must be positive")
    if min(int(cfg.trainer.logging_frequency), int(cfg.trainer.save_interval)) < 1:
        raise ValueError("Logging and save intervals must be positive")
    plugin = (
        None
        if os.environ.get("STARVLA_DISABLE_DEEPSPEED") == "1"
        else DeepSpeedPlugin(
            gradient_accumulation_steps=accumulation,
            gradient_clipping=float(cfg.trainer.gradient_clipping),
        )
    )
    accelerator = Accelerator(
        mixed_precision="bf16",
        # ZeRO-2 owns gradient partitioning and accumulation; its no_sync is
        # incompatible with partitioned gradients. Keep communication enabled.
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=accumulation,
            sync_each_batch=plugin is not None,
        ),
        deepspeed_plugin=plugin,
        step_scheduler_with_optimizer=False,
        dataloader_config=DataLoaderConfiguration(
            dispatch_batches=False,
            use_seedable_sampler=True,
            data_seed=int(cfg.seed),
        ),
    )
    resume = cfg.trainer.get("resume_from_checkpoint")
    output = Path(cfg.run_root_dir) / cfg.run_id
    output_exists = output.exists()
    accelerator.wait_for_everyone()
    if output_exists and not resume:
        raise FileExistsError(f"Use a new run_id or trainer.resume_from_checkpoint: {output}")
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output / "config.full.yaml", resolve=True)
    accelerator.wait_for_everyone()

    set_seed(int(cfg.seed))
    if resume:
        # Build using the saved bundle, then restore optimizer/RNG state below.
        saved_mode = OmegaConf.load(Path(resume) / "model/inference_config.yaml").framework.interaction_mode
        if saved_mode != cfg.framework.interaction_mode:
            raise ValueError("A training resume cannot change interaction_mode")
        source_cfg = OmegaConf.merge(cfg, {"trainer": {"pretrained_checkpoint": str(Path(resume) / "model")}})
    else:
        source_cfg = cfg
    model = initialize_model(source_cfg)
    loader = create_dataloader(cfg)
    if len(loader) == 0:
        raise ValueError("Dataset is too small for the requested per-device batch size")
    optimizer = torch.optim.AdamW(
        optimizer_groups(model, cfg),
        lr=float(cfg.trainer.learning_rate.base),
        betas=tuple(cfg.trainer.optimizer.betas),
        eps=float(cfg.trainer.optimizer.eps),
        fused=torch.cuda.is_available(),
    )
    scheduler = get_scheduler(
        cfg.trainer.lr_scheduler_type,
        optimizer,
        num_warmup_steps=int(cfg.trainer.num_warmup_steps),
        num_training_steps=int(cfg.trainer.max_train_steps),
        scheduler_specific_kwargs=OmegaConf.to_container(cfg.trainer.scheduler_specific_kwargs),
    )
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    completed, epoch, consumed = 0, 0, 0
    resume_contract = {
        "interaction_mode": cfg.framework.interaction_mode,
        "world_size": accelerator.num_processes,
        "per_device_batch_size": int(cfg.datasets.vla_data.per_device_batch_size),
        "gradient_accumulation_steps": accumulation,
        "seed": int(cfg.seed),
        "data_root_dir": str(Path(cfg.datasets.vla_data.data_root_dir).resolve()),
    }
    if resume:
        metadata = json.loads((Path(resume) / "training_state.json").read_text())
        if metadata["contract"] != resume_contract:
            raise ValueError("Resume requires the same data, seed, batch size and distributed topology")
        completed, epoch, consumed = (int(metadata[key]) for key in ("completed_steps", "epoch", "batches_consumed"))
        accelerator.load_state(str(Path(resume) / "state"))
        scheduler.load_state_dict(metadata["scheduler"])
    else:
        set_seed(int(cfg.seed) + accelerator.process_index)

    def checkpoint():
        directory = output / "checkpoints" / f"steps_{completed}"
        accelerator.save_state(str(directory / "state"))
        state = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            save_model_bundle(accelerator.unwrap_model(model), state, directory / "model")
            (directory / "training_state.json").write_text(
                json.dumps(
                    {
                        "completed_steps": completed,
                        "epoch": epoch,
                        "batches_consumed": consumed,
                        "scheduler": scheduler.state_dict(),
                        "contract": resume_contract,
                    },
                    indent=2,
                )
                + "\n"
            )
        accelerator.wait_for_everyone()

    optimizer.zero_grad(set_to_none=True)
    while completed < int(cfg.trainer.max_train_steps):
        loader.dataset.set_epoch(epoch)
        loader.set_epoch(epoch)
        iterator = accelerator.skip_first_batches(loader, consumed) if consumed else loader
        for examples in iterator:
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    result = model(examples)
                    loss = result["action_loss"]
                finite_ranks = accelerator.reduce(torch.isfinite(loss.detach()).all().float(), reduction="sum")
                if finite_ranks.item() != accelerator.num_processes:
                    raise FloatingPointError("Non-finite Ego4WAM training loss")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), float(cfg.trainer.gradient_clipping))
                optimizer.step()
                if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                    scheduler.step()
                    completed += 1
                optimizer.zero_grad(set_to_none=True)
            consumed += 1
            if accelerator.sync_gradients:
                if completed % int(cfg.trainer.logging_frequency) == 0:
                    metrics = {"step": completed, "loss": float(loss.detach()), **result.get("loss_metrics", {})}
                    accelerator.print(json.dumps(metrics))
                    if accelerator.is_main_process:
                        with (output / "metrics.jsonl").open("a") as stream:
                            stream.write(json.dumps(metrics) + "\n")
                if completed and completed % int(cfg.trainer.save_interval) == 0:
                    checkpoint()
                if completed >= int(cfg.trainer.max_train_steps):
                    break
        else:
            epoch += 1
            consumed = 0

    state = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        save_model_bundle(accelerator.unwrap_model(model), state, output / "final_model")
    accelerator.wait_for_everyone()
    accelerator.print(f"Training complete: {output / 'final_model'}")
    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_yaml", required=True)
    args, overrides = parser.parse_known_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config_yaml), OmegaConf.from_dotlist(normalize_dotlist_args(overrides)))
    run(cfg)


if __name__ == "__main__":
    main()
