#!/usr/bin/env python3
# Copyright (c) 2026 Ego4WAM Authors.
# SPDX-License-Identifier: MIT

"""Ask one or more Ego4WAM policy servers to stop."""

from __future__ import annotations

import argparse
import os
import sys

import websockets.sync.client

from deployment.model_server.tools import msgpack_numpy


def send_shutdown(host: str, port: int, timeout: float, shutdown_token: str) -> None:
    for key in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        os.environ.pop(key, None)
    packer = msgpack_numpy.Packer()
    with websockets.sync.client.connect(
        f"ws://{host}:{port}", compression=None, max_size=None, open_timeout=timeout
    ) as conn:
        conn.recv()  # metadata handshake
        conn.send(
            packer.pack(
                {
                    "type": "shutdown",
                    "request_id": f"shutdown-{port}",
                    "shutdown_token": shutdown_token,
                }
            )
        )
        response = conn.recv()
        if isinstance(response, str):
            raise RuntimeError(response)
        decoded = msgpack_numpy.unpackb(response)
        if not decoded.get("ok", False):
            raise RuntimeError(f"shutdown rejected: {decoded}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--base-port", type=int, required=True)
    parser.add_argument("--num-servers", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--shutdown-token", required=True)
    args = parser.parse_args()

    failed = False
    for index in range(args.num_servers):
        port = args.base_port + index
        try:
            send_shutdown(args.host, port, args.timeout, args.shutdown_token)
            print(f"[ROBODOJO CLIENT] shutdown sent to {args.host}:{port}")
        except Exception as exc:
            failed = True
            print(f"[ROBODOJO CLIENT] failed to shutdown {args.host}:{port}: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
