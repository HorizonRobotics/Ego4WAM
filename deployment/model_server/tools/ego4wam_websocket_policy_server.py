# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

import asyncio
import collections
import dataclasses
import hmac
import logging
import time
import traceback
from typing import Any

import websockets.asyncio.server
import websockets.frames

from . import msgpack_numpy


@dataclasses.dataclass
class _PendingRequest:
    msg: dict
    future: asyncio.Future
    examples_len: int


class Ego4WAMWebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int = 10093,
        idle_timeout: int = -1,  # Negative disables automatic shutdown
        metadata: dict | None = None,
        max_batch_size: int = 8,
        batch_timeout_ms: float = 10.0,
        shutdown_token: str | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._idle_timeout = idle_timeout
        self._max_batch_size = max(1, int(max_batch_size))
        self._batch_timeout_s = max(0.0, float(batch_timeout_ms) / 1000.0)
        self._shutdown_token = shutdown_token
        self._pending_requests: collections.deque[_PendingRequest] = collections.deque()
        self._pending_condition: asyncio.Condition | None = None
        self._dispatcher_task: asyncio.Task | None = None
        self._shutdown_task: asyncio.Task | None = None
        self._server: Any | None = None
        self._shutdown_requested = False
        self._last_active = time.time()
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        self._pending_condition = asyncio.Condition()
        self._dispatcher_task = asyncio.create_task(self._batch_dispatcher())
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            self._server = server
            try:
                if self._idle_timeout > 0:
                    await self._idle_watchdog(server)
                else:
                    await server.serve_forever()
            finally:
                self._server = None
                if self._dispatcher_task is not None:
                    self._dispatcher_task.cancel()
                    try:
                        await self._dispatcher_task
                    except asyncio.CancelledError:
                        pass

    async def _idle_watchdog(self, server):
        """Close the server after the configured idle timeout."""
        while True:
            await asyncio.sleep(5)
            if self._shutdown_requested:
                server.close()
                await server.wait_closed()
                break
            if time.time() - self._last_active > self._idle_timeout:
                logging.info(f"Idle timeout ({self._idle_timeout}s) reached, shutting down server.")
                server.close()
                await server.wait_closed()
                break

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                msg = msgpack_numpy.unpackb(await websocket.recv())
                self._last_active = time.time()  # Refresh active time on every request
                ret = await self._route_message(msg)  # route message
                await websocket.send(packer.pack(ret))
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    # route logic: recognize request from client
    async def _route_message(self, msg: dict) -> dict:
        """
        Route rules (fault-tolerant):
        - Supports messages of form:
            {"type": "ping|init|infer|reset|shutdown", "request_id": "...", "payload": {...}}
          or a flat dict (will be treated as payload).
        - Does NOT raise inside this function: all exceptions are caught and encoded in response.
        """
        if not isinstance(msg, dict):
            return {
                "status": "error",
                "ok": False,
                "type": "inference_result",
                "request_id": "default",
                "error": {"message": "Payload must be a dict", "payload_type": str(type(msg))},
            }

        req_id = msg.get("request_id", "default")
        mtype = msg.get("type", "infer")  # default = infer

        # ping
        if mtype == "ping":
            return {"status": "ok", "ok": True, "type": "ping", "request_id": req_id}

        elif mtype == "shutdown":
            if not self._is_authorized_shutdown(msg):
                return {
                    "status": "error",
                    "ok": False,
                    "type": "shutdown",
                    "request_id": req_id,
                    "error": {"message": "Shutdown requires a valid shutdown token."},
                }
            self._request_shutdown()
            return {"status": "ok", "ok": True, "type": "shutdown", "request_id": req_id}

        # infer --> framework.predict_action
        elif mtype == "infer" or mtype == "predict_action":
            return await self._enqueue_infer(msg)

        # unknow request type
        else:
            return {
                "status": "error",
                "ok": False,
                "type": "unknown",
                "request_id": req_id,
                "error": {"message": f"Unsupported message type '{mtype}'"},
            }

    def _request_shutdown(self) -> None:
        if self._shutdown_requested:
            return
        self._shutdown_requested = True
        logging.info("Shutdown requested by client; closing server.")
        self._shutdown_task = asyncio.create_task(self._shutdown_soon())

    def _is_authorized_shutdown(self, msg: dict) -> bool:
        """Require an explicit shared secret for the remote control operation."""
        supplied_token = msg.get("shutdown_token")
        return bool(
            self._shutdown_token
            and isinstance(supplied_token, str)
            and hmac.compare_digest(supplied_token, self._shutdown_token)
        )

    async def _shutdown_soon(self) -> None:
        await asyncio.sleep(0.1)
        if self._server is not None:
            self._server.close()

    async def _enqueue_infer(self, msg: dict) -> dict:
        examples = msg.get("examples")
        if not isinstance(examples, list) or len(examples) == 0:
            return self._error_response(msg, "examples must be a non-empty list")
        if len(examples) > self._max_batch_size:
            return self._error_response(
                msg,
                f"Request contains {len(examples)} examples, exceeding " f"max_batch_size={self._max_batch_size}",
            )

        if self._pending_condition is None:
            return await asyncio.to_thread(self._infer_one, msg)

        loop = asyncio.get_running_loop()
        pending = _PendingRequest(
            msg=msg,
            future=loop.create_future(),
            examples_len=len(examples),
        )
        async with self._pending_condition:
            self._pending_requests.append(pending)
            self._pending_condition.notify()
        return await pending.future

    async def _batch_dispatcher(self) -> None:
        while True:
            first = await self._pop_next_request()
            if self._batch_timeout_s > 0:
                await asyncio.sleep(self._batch_timeout_s)

            batch = [first]
            batch_key = self._batch_key(first.msg)
            examples_count = first.examples_len
            async with self._pending_condition:
                skipped: collections.deque[_PendingRequest] = collections.deque()
                while self._pending_requests and examples_count < self._max_batch_size:
                    candidate = self._pending_requests.popleft()
                    if (
                        self._batch_key(candidate.msg) == batch_key
                        and examples_count + candidate.examples_len <= self._max_batch_size
                    ):
                        batch.append(candidate)
                        examples_count += candidate.examples_len
                    else:
                        skipped.append(candidate)
                while skipped:
                    self._pending_requests.appendleft(skipped.pop())

            await self._serve_batch(batch)

    async def _pop_next_request(self) -> _PendingRequest:
        async with self._pending_condition:
            while not self._pending_requests:
                await self._pending_condition.wait()
            return self._pending_requests.popleft()

    async def _serve_batch(self, batch: list[_PendingRequest]) -> None:
        try:
            response = await asyncio.to_thread(self._infer_batch, batch)
        except Exception as exc:
            logging.exception("Policy batch inference error (batch_size=%d)", len(batch))
            for pending in batch:
                if not pending.future.done():
                    pending.future.set_result(self._error_response(pending.msg, str(exc)))
            return

        for pending, item_response in zip(batch, response, strict=True):
            if not pending.future.done():
                pending.future.set_result(item_response)

    def _infer_batch(self, batch: list[_PendingRequest]) -> list[dict]:
        request_sizes = [pending.examples_len for pending in batch]
        logging.info(
            "Serving batched inference: requests=%d websocket_requests=%d request_sizes=%s",
            sum(request_sizes),
            len(batch),
            request_sizes,
        )
        if len(batch) == 1:
            return [self._infer_one(batch[0].msg)]

        merged_msg = dict(batch[0].msg)
        merged_examples = []
        for pending in batch:
            examples = pending.msg["examples"]
            merged_examples.extend(examples)
        merged_msg["examples"] = merged_examples

        output_dict = self._policy.predict_action(**self._policy_kwargs(merged_msg))
        split_data = self._split_batched_data(output_dict, request_sizes)
        return [self._ok_response(pending.msg, data) for pending, data in zip(batch, split_data, strict=True)]

    def _infer_one(self, msg: dict) -> dict:
        req_id = msg.get("request_id", "default")
        try:
            output_dict = self._policy.predict_action(**self._policy_kwargs(msg))
        except Exception as e:
            logging.exception("Policy inference error (request_id=%s)", req_id)
            return self._error_response(msg, str(e))
        return self._ok_response(msg, output_dict)

    def _ok_response(self, msg: dict, data: dict) -> dict:
        return {
            "status": "ok",
            "ok": True,
            "type": "inference_result",
            "request_id": msg.get("request_id", "default"),
            "data": data,
        }

    def _error_response(self, msg: dict, message: str) -> dict:
        return {
            "status": "error",
            "ok": False,
            "type": "inference_result",
            "request_id": msg.get("request_id", "default"),
            "error": {
                "message": message,
            },
        }

    def _batch_key(self, msg: dict) -> tuple:
        return tuple(
            (key, self._stable_repr(value))
            for key, value in sorted(msg.items())
            if key not in {"examples", "request_id", "shutdown_token", "type"}
        )

    def _stable_repr(self, value: Any) -> str:
        if isinstance(value, dict):
            return "{" + ",".join(f"{k}:{self._stable_repr(v)}" for k, v in sorted(value.items())) + "}"
        if isinstance(value, (list, tuple)):
            return "[" + ",".join(self._stable_repr(v) for v in value) + "]"
        return repr(value)

    def _policy_kwargs(self, msg: dict) -> dict:
        return {key: value for key, value in msg.items() if key not in {"request_id", "shutdown_token", "type"}}

    def _split_batched_data(self, data: dict, request_sizes: list[int]) -> list[dict]:
        total = sum(request_sizes)
        split_data = [dict() for _ in request_sizes]
        for key, value in data.items():
            if hasattr(value, "shape") and len(value.shape) > 0 and value.shape[0] == total:
                start = 0
                for idx, size in enumerate(request_sizes):
                    split_data[idx][key] = value[start : start + size]
                    start += size
            elif isinstance(value, list) and len(value) == total:
                start = 0
                for idx, size in enumerate(request_sizes):
                    split_data[idx][key] = value[start : start + size]
                    start += size
            else:
                for item in split_data:
                    item[key] = value
        return split_data
