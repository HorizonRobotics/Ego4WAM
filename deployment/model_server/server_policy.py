# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""Serve released Ego4WAM artifacts for the RoboDojo raw-observation client."""

import argparse
import logging
import os
import socket
from pathlib import Path

from deployment.model_server.ego4wam_robodojo_policy import Ego4WAMRoboDojoPolicy
from deployment.model_server.seed_utils import set_seed_everywhere
from deployment.model_server.tools.ego4wam_websocket_policy_server import Ego4WAMWebsocketPolicyServer


def _build_wrapper(args):
    artifact_dir = Path(args.ckpt_path).expanduser()
    if not (artifact_dir / "model.safetensors").is_file():
        raise FileNotFoundError(f"Ego4WAM requires a released model directory, got {artifact_dir}")
    return Ego4WAMRoboDojoPolicy(artifact_dir, device=args.device)


def main(args) -> None:
    seed = args.seed
    if seed is not None:
        set_seed_everywhere(seed)
    wrapper = _build_wrapper(args)
    hostname = socket.gethostname()
    logging.info("Creating server (host: %s, ip: %s)", hostname, socket.gethostbyname(hostname))
    logging.warning(
        "[TRAIN/TEST CONSISTENCY CHECK] serving ckpt=%s; verify the RoboDojo "
        "camera/state/action/normalization contract. metadata=%s",
        args.ckpt_path,
        wrapper.metadata,
    )
    metadata = dict(wrapper.metadata)
    metadata.update(
        seed=seed,
        max_batch_size=args.max_batch_size,
        batch_timeout_ms=args.batch_timeout_ms,
        sampling_mode="historical_global_rng",
    )
    server = Ego4WAMWebsocketPolicyServer(
        policy=wrapper,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=metadata,
        max_batch_size=args.max_batch_size,
        batch_timeout_ms=args.batch_timeout_ms,
        shutdown_token=os.getenv("POLICY_SERVER_SHUTDOWN_TOKEN") or None,
    )
    logging.info("server running ... metadata=%s", metadata)
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_path", required=True, help="Released Ego4WAM model directory")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--device", default="cuda", help="One device or a pair such as cuda:0,cuda:1")
    # Retained for compatibility with the public local_server.sh command.
    parser.add_argument("--robodojo-ego4wam", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=None, help="Optional model RNG seed; historical default is unset")
    parser.add_argument("--idle_timeout", type=int, default=1800, help="Idle timeout in seconds; -1 disables it")
    parser.add_argument("--max-batch-size", type=int, default=10)
    parser.add_argument("--batch-timeout-ms", type=float, default=20.0)
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(build_argparser().parse_args())
