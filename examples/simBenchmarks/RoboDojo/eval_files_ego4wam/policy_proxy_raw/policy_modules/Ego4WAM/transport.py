# Copyright (c) 2026 Beijing Horizon Information Technology Company Limited.
# Author: Ego4WAM Team.
# SPDX-License-Identifier: Apache-2.0
"""Raw model RPC compatible with both historical Python environments."""

from __future__ import annotations

import inspect

from websockets.sync.client import connect

from deployment.model_server.tools import msgpack_numpy


class RawPolicyClient:
    def __init__(self, host: str, port: int) -> None:
        options = {"compression": None, "max_size": None, "open_timeout": 150}
        if "proxy" in inspect.signature(connect).parameters:
            options["proxy"] = None
        self.connection = connect(f"ws://{host}:{port}", **options)
        self.metadata = self._receive()

    def _receive(self) -> dict:
        response = self.connection.recv()
        if isinstance(response, str):
            raise RuntimeError(response)
        response = msgpack_numpy.unpackb(response)
        if not isinstance(response, dict):
            raise TypeError("Model RPC response must be a mapping")
        return response

    def get_server_metadata(self) -> dict:
        return self.metadata

    def predict_action(self, query: dict) -> dict:
        self.connection.send(msgpack_numpy.Packer().pack(query))
        return self._receive()

    def close(self) -> None:
        self.connection.close()
