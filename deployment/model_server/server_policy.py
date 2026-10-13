# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].
# Modifications Copyright (c) 2026 Ego4WAM Authors.
# SPDX-License-Identifier: Apache-2.0 AND MIT
# Ego4WAM additions/modifications: Apache-2.0; upstream StarVLA portions: MIT.
# See LICENSES/StarVLA-MIT.txt for the retained upstream license.

import argparse
import logging
import os
import socket
from pathlib import Path

from deployment.model_server.seed_utils import set_seed_everywhere
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


def PolicyServerWrapper(*args, **kwargs):
    """Lazily import the standard wrapper while preserving its injection point."""
    from deployment.model_server.policy_wrapper import PolicyServerWrapper as wrapper_class

    return wrapper_class(*args, **kwargs)


def _build_wrapper(args):
    device = getattr(args, "device", "cuda")
    if not isinstance(device, str):
        device = "cuda"
    if getattr(args, "robodojo_ego4wam", False) is True:
        from deployment.model_server.ego4wam_robodojo_policy import (
            Ego4WAMRoboDojoPolicy,
        )

        artifact_dir = Path(args.ckpt_path).expanduser()
        if not (artifact_dir / "model.safetensors").is_file():
            raise FileNotFoundError("--robodojo-ego4wam requires an Ego4WAM release directory, " f"got {artifact_dir}")
        return Ego4WAMRoboDojoPolicy(artifact_dir, device=device)
    return PolicyServerWrapper(
        ckpt_path=args.ckpt_path,
        device=device,
        use_bf16=args.use_bf16,
        config_overrides=args.config_override,
    )


def main(args) -> None:
    """Build the policy wrapper and start the websocket server.

    The wrapper now owns un-normalization + chunk_size discovery so that all
    eval clients (LIBERO / SimplerEnv / etc.) just need to forward `examples`
    and consume already-unnormalized actions from the response.
    """
    seed = getattr(args, "seed", None)
    if not isinstance(seed, int):
        seed = None
    if seed is not None:
        set_seed_everywhere(seed)

    config_overrides = getattr(args, "config_override", [])
    if config_overrides:
        override_keys = [item.split("=", 1)[0] for item in config_overrides]
        logging.info("Applying config override keys: %s", override_keys)
    wrapper = _build_wrapper(args)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # =========================================================================
    # !!! TRAIN / TEST CONSISTENCY — READ BEFORE SERVING !!!
    # -------------------------------------------------------------------------
    # This server replays the *training-time* observation contract. The eval
    # client MUST feed observations exactly as the model saw them at TRAIN time,
    # otherwise the success rate silently drops (no error is raised):
    #   - state   : is proprioceptive state used? (use_state) and its dim/order
    #   - img size: resize / crop resolution (e.g. 224x224)
    #   - img num : how many camera views are fed
    #   - img order: the ordering of those camera views
    #   - action normalization: unnorm_key / dataset stats must match training
    # `wrapper.metadata` (logged below and sent at handshake) exposes
    # action_chunk_size / state_keys / action_keys — cross-check these against
    # the training config used to produce `args.ckpt_path`.
    # =========================================================================
    logging.warning(
        "[TRAIN/TEST CONSISTENCY CHECK] serving ckpt=%s — verify eval observations "
        "(state / image size / image count / image order / action normalization) match "
        "the training config. metadata=%s",
        args.ckpt_path,
        wrapper.metadata,
    )

    # start websocket server; wrapper.metadata is sent at handshake.
    metadata = dict(wrapper.metadata)
    metadata["seed"] = seed
    server_type = WebsocketPolicyServer
    server_options = {}
    if getattr(args, "robodojo_ego4wam", False) is True:
        from deployment.model_server.tools.ego4wam_websocket_policy_server import (
            Ego4WAMWebsocketPolicyServer,
        )

        server_type = Ego4WAMWebsocketPolicyServer
        server_options = {
            "max_batch_size": args.max_batch_size,
            "batch_timeout_ms": args.batch_timeout_ms,
            "shutdown_token": os.getenv("POLICY_SERVER_SHUTDOWN_TOKEN") or None,
        }
        metadata.update(
            max_batch_size=args.max_batch_size,
            batch_timeout_ms=args.batch_timeout_ms,
            sampling_mode="historical_global_rng",
        )
    server = server_type(
        policy=wrapper,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=metadata,
        **server_options,
    )
    logging.info("server running ... metadata=%s", metadata)
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument(
        "--device",
        default="cuda",
        help="One device, or an Ego4WAM main/text pair such as cuda:0,cuda:1.",
    )
    parser.add_argument(
        "--robodojo-ego4wam",
        action="store_true",
        help="Serve a converted Ego4WAM artifact through the RoboDojo raw-observation contract.",
    )
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed Python, NumPy, and PyTorch before loading the policy"
    )
    parser.add_argument("--idle_timeout", type=int, default=1800, help="Idle timeout in seconds, -1 means never close")
    parser.add_argument("--max-batch-size", type=int, default=10, help="Ego4WAM microbatch capacity")
    parser.add_argument("--batch-timeout-ms", type=float, default=20.0, help="Ego4WAM microbatch collection window")
    parser.add_argument(
        "--config_override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable OmegaConf dotlist override applied before model construction.",
    )
    return parser


def start_debugpy_once():
    """start debugpy once"""
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10095))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10095 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    if os.getenv("DEBUG", False):
        print("🔍 DEBUGPY is enabled")
        start_debugpy_once()
    main(args)
