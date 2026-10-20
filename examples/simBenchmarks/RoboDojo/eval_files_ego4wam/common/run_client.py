#!/usr/bin/env python3
# Copyright (c) 2026 Beijing Horizon Information Technology Company Limited.
# Author: Ego4WAM Team.
# SPDX-License-Identifier: MIT

"""Schedule RoboDojo runnable configs while tracking canonical benchmark tasks."""

from __future__ import annotations

import json
import os
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from websockets.exceptions import WebSocketException
from websockets.sync.client import connect as websocket_connect

from deployment.model_server.tools import msgpack_numpy
from examples.simBenchmarks.RoboDojo.eval_files_ego4wam.common.result_validation import validate_result

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
SUMMARY_SCRIPT = SCRIPT_DIR / "summarize_results.py"
SHUTDOWN_SCRIPT = SCRIPT_DIR / "shutdown_servers.py"
RAW_PROXY_BOOTSTRAP = REPO_ROOT / "examples/simBenchmarks/RoboDojo/eval_files_ego4wam/policy_proxy_raw/proxy_raw_bootstrap.py"
LOG_LOCK = threading.Lock()
KNOWN_NONFATAL_LOG_MARKERS = (
    "extensions config 'extension.toml' doesn't exist",
    "glfw initialization failed",
    "failed to startup plugin carb.windowing-glfw.plugin",
    "failed to open the default display",
    "cuda_visible_devices environment variable is set",
    "cuda device enumeration and omniverse device enumeration are different",
    "setting cuda_visible_devices can lead to undesired behavior or crashes",
    "skipping nvidia gpu due cuda being in bad state",
    "please restart your system if cuda is known to work",
    "iommu is enabled",
    "running cuda peer-to-peer bandwidth and latency validation",
    "source: omni.hydra was already registered",
    "acquiring non optional plugin interface",
    "omni.isaac.dynamic_control is deprecated",
    "requestsdependencywarning",
    "warnings.warn(",
    "pxr.semantics is deprecated",
    "found duplicate of category 'replicator'",
    "category 'replicator' not accepted",
    "no material configuration file",
    "disabling key-value database because another kit process is locking it",
    "enable_external_forces_every_iteration",
    "audio device is misconfigured or broken",
    "failed to retrieve the capabilities for device",
    "using a null streamer instead",
    "attribute viewporthandle not found",
    "failed to obtain username for current process",
    "invalid volume error percentage",
    "must be > 3 to avoid blank textures",
    "are inconsistent with the pixel resolution aspect ratio",
    "refers to a path outside the scope of the reference",
    "usd->fabric: unhandled array type string[]",
    "unhandled attribute type vtarray<std::string>",
    "could not find module '::zddhttps_3a_2f",
    "loading mdlmodule to db",
    "mdlmodule is not in the db",
    "usd_mdl: in loadmodule",
    "unable to find sdrshadernode",
    "conversion from 'float' to 'float2' must be explicit",
    "imemorybudgetmanagerfactory",
)
EXPECTED_VIDEO_FILENAMES = frozenset({"head.mp4", "left_wrist.mp4", "right_wrist.mp4"})
RAW_OBSERVATION_SCHEMA = "camera_observations_v1"
SUPPORTED_ACTION_ENCODINGS = frozenset({"joints_eef_32_abs"})
NON_PROGRESS_LOG_MARKERS = (
    "zenity",
    "hang detected",
    "kit appears to be hanging",
    "showing message box with the command",
    "command returned:",
    "user declined, ignoring",
)


def log(
    message: str,
    *,
    task_id: int | None = None,
    total_tasks: int | None = None,
    config_id: int | None = None,
    total_configs: int | None = None,
) -> None:
    progress = []
    if task_id is not None and total_tasks is not None:
        progress.append(f"task {task_id}/{total_tasks}")
    if config_id is not None and total_configs is not None:
        progress.append(f"config {config_id}/{total_configs}")
    prefix = f"[{' '.join(progress)}]" if progress else "[ROBODOJO CLIENT]"
    with LOG_LOCK:
        print(f"{prefix} {message}", flush=True)


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value in (None, "") else int(value)


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def canonical_task_name(config_name: str) -> str:
    return config_name.removesuffix("_random")


def parse_worker_info() -> tuple[int, int]:
    index = env_int("CLIENT_WORKER_INDEX", 0)
    count = env_int("CLIENT_NUM_WORKERS", 1)
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"invalid worker topology index={index}, count={count}")
    return index, count


def receive_server_metadata(host: str, port: int, timeout: float) -> dict[str, Any]:
    with websocket_connect(
        f"ws://{host}:{port}",
        open_timeout=timeout,
        close_timeout=timeout,
        compression=None,
        max_size=None,
    ) as connection:
        message = connection.recv(timeout=timeout)
    if isinstance(message, str):
        raise TypeError(f"server {host}:{port} returned text metadata: {message}")
    metadata = msgpack_numpy.unpackb(message)
    if not isinstance(metadata, dict):
        raise TypeError(f"server {host}:{port} metadata must be a dict, got {type(metadata).__name__}")
    return metadata


def wait_for_server_metadata(
    host: str,
    base_port: int,
    num_servers: int,
    max_wait: int,
) -> list[dict[str, Any]]:
    deadline = None if max_wait < 0 else time.monotonic() + max_wait
    while True:
        metadata_by_port: dict[int, dict[str, Any]] = {}
        errors: dict[int, str] = {}
        for port in range(base_port, base_port + num_servers):
            try:
                metadata_by_port[port] = receive_server_metadata(host, port, timeout=2.0)
            except (OSError, TimeoutError, WebSocketException, TypeError, ValueError) as exc:
                errors[port] = str(exc)
        if not errors:
            return [metadata_by_port[port] for port in range(base_port, base_port + num_servers)]
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"servers unavailable or missing metadata at {host}: {errors}")
        log(f"waiting for policy server metadata; errors={errors}")
        time.sleep(3)


def wait_for_port(host: str, port: int, process: subprocess.Popen[Any], timeout: int) -> None:
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError(f"proxy readiness check only supports a local host, got {host}")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"proxy exited before listening on {host}:{port}")
        if local_port_is_listening(port):
            return
        time.sleep(1)
    raise TimeoutError(f"proxy did not listen on {host}:{port} within {timeout}s")


def local_port_is_listening(port: int) -> bool:
    for table_path in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        try:
            lines = table_path.read_text(encoding="ascii").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 4 or fields[3] != "0A":
                continue
            if int(fields[1].rsplit(":", 1)[1], 16) == port:
                return True
    return False


def resolve_server_contract(
    metadata_list: list[dict[str, Any]],
    *,
    ckpt: Path,
    unnorm_key: str,
) -> tuple[dict[str, Any], int]:
    if not metadata_list:
        raise ValueError("server metadata list must not be empty")
    metadata = metadata_list[0]
    for server_index, candidate in enumerate(metadata_list[1:], start=1):
        if candidate != metadata:
            raise ValueError(
                "policy server metadata mismatch: "
                f"server[0]={json.dumps(metadata, sort_keys=True, default=str)}, "
                f"server[{server_index}]={json.dumps(candidate, sort_keys=True, default=str)}"
            )

    metadata_ckpt = metadata.get("ckpt_path")
    if metadata_ckpt and Path(metadata_ckpt).resolve() != ckpt:
        raise ValueError(f"server checkpoint mismatch: expected {ckpt}, got {metadata_ckpt}")

    available_unnorm_keys = metadata.get("available_unnorm_keys")
    if not isinstance(available_unnorm_keys, list) or unnorm_key not in available_unnorm_keys:
        raise ValueError(f"UNNORM_KEY={unnorm_key!r} is unavailable; server keys={available_unnorm_keys}")

    action_chunk_size = int(metadata.get("action_chunk_size", 0))
    if action_chunk_size < 1:
        raise ValueError(f"invalid server action_chunk_size={action_chunk_size}")

    if metadata.get("server_preprocessing") is not True:
        raise ValueError("RoboDojo raw observations require server_preprocessing=true")
    if metadata.get("raw_observation_schema") != RAW_OBSERVATION_SCHEMA:
        raise ValueError(
            "RoboDojo raw observation schema mismatch: "
            f"expected {RAW_OBSERVATION_SCHEMA!r}, got {metadata.get('raw_observation_schema')!r}"
        )
    encoding = metadata.get("joint_action_encoding")
    if encoding not in SUPPORTED_ACTION_ENCODINGS:
        raise ValueError("RoboDojo Ego4WAM supports exactly " f"{sorted(SUPPORTED_ACTION_ENCODINGS)}, got {encoding!r}")
    if metadata.get("camera_roles") != ["head", "left_wrist", "right_wrist"]:
        raise ValueError("RoboDojo raw server must advertise camera_roles=" "['head', 'left_wrist', 'right_wrist']")

    return metadata, action_chunk_size


def ensure_port_available(host: str, port: int) -> None:
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError(f"proxy port check only supports a local host, got {host}")
    if local_port_is_listening(port):
        raise OSError(f"proxy port is already listening: {host}:{port}")


def load_inventory(robodojo_root: Path) -> tuple[list[str], dict[str, str]]:
    command = [
        sys.executable,
        str(robodojo_root / "scripts/internal/task_inventory.py"),
        "--format",
        "json",
        "--check",
    ]
    result = subprocess.run(
        command,
        cwd=robodojo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    entries = [entry for entry in payload["tasks"] if entry.get("runnable")]
    return [entry["name"] for entry in entries], {entry["name"]: entry.get("dimension", "unknown") for entry in entries}


def parse_task_names(all_tasks: list[str]) -> list[str]:
    requested = os.getenv("TASK_NAMES", "").replace(",", " ").split()
    tasks = requested or all_tasks
    unknown = sorted(set(tasks) - set(all_tasks))
    if unknown:
        raise ValueError(f"unknown RoboDojo tasks: {unknown}")
    max_tasks = env_int("MAX_TASKS", -1)
    return tasks if max_tasks < 0 else tasks[:max_tasks]


def task_weights(robodojo_root: Path, tasks: list[str], num_episodes: str) -> dict[str, int]:
    if num_episodes != "native":
        return {task: int(num_episodes) for task in tasks}
    config = yaml.safe_load((robodojo_root / "task/RoboDojo/config/_task.yml").read_text(encoding="utf-8"))
    common_eval_num = int(config.get("common", {}).get("eval_nums", 50))
    overrides = config.get("tasks", {})
    return {task: int(overrides.get(task, {}).get("eval_nums", common_eval_num)) for task in tasks}


def partition_tasks(tasks: list[str], weights: dict[str, int], worker_count: int) -> list[list[str]]:
    partitions = [[] for _ in range(worker_count)]
    totals = [0 for _ in range(worker_count)]
    for task in tasks:
        worker_index = min(
            range(worker_count),
            key=lambda index: (totals[index], len(partitions[index]), index),
        )
        partitions[worker_index].append(task)
        totals[worker_index] += weights[task]
    return partitions


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def collect_task_videos(video_root: Path, task_name: str, expected_episodes: int) -> list[Path]:
    task_video_root = video_root / task_name
    video_paths = sorted(task_video_root.glob("episode_*/*.mp4"))
    if not video_paths:
        raise RuntimeError("SAVE_VIDEO=1 but RoboDojo produced no mp4 files")

    invalid_paths = [
        path
        for path in video_paths
        if not re.fullmatch(r"episode_\d{7}", path.parent.name) or path.name not in EXPECTED_VIDEO_FILENAMES
    ]
    if invalid_paths:
        raise RuntimeError(f"RoboDojo produced non-standard video paths: {invalid_paths}")

    incomplete_episodes = []
    for episode_dir in sorted({path.parent for path in video_paths}):
        filenames = {path.name for path in episode_dir.glob("*.mp4")}
        if filenames != EXPECTED_VIDEO_FILENAMES:
            incomplete_episodes.append((episode_dir, sorted(filenames)))
    if incomplete_episodes:
        raise RuntimeError(f"RoboDojo produced incomplete episode videos: {incomplete_episodes}")
    expected_ids = {f"episode_{index:07d}" for index in range(expected_episodes)}
    if {path.parent.name for path in video_paths} != expected_ids:
        raise RuntimeError(f"Expected videos for {expected_episodes} episodes, got {len(video_paths) // 3}")
    if any(path.stat().st_size == 0 for path in video_paths):
        raise RuntimeError("RoboDojo produced empty video files")
    return video_paths


def analyze_log(path: Path) -> dict[str, Any]:
    warning_count = 0
    error_count = 0
    known_nonfatal_count = 0
    known_nonfatal_warning_count = 0
    known_nonfatal_error_count = 0
    warning_lines = []
    error_lines = []
    known_nonfatal_lines = []
    actionable_warning_lines = []
    actionable_error_lines = []
    if not path.is_file():
        return {
            "warnings": 0,
            "errors": 0,
            "known_nonfatal": 0,
            "actionable_warnings": 0,
            "actionable_errors": 0,
            "warning_examples": [],
            "error_examples": [],
            "known_nonfatal_examples": [],
            "actionable_warning_examples": [],
            "actionable_error_examples": [],
        }
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        lowered = line.lower()
        is_warning = "warning" in lowered or "[warn" in lowered
        is_error = any(
            marker in lowered
            for marker in (
                "traceback (most recent call last)",
                "[error",
                "error:",
                "runtimeerror",
                "cuda error",
                "segmentation fault",
                "exception:",
            )
        )
        is_known_nonfatal = any(marker in lowered for marker in KNOWN_NONFATAL_LOG_MARKERS)
        if is_warning:
            warning_count += 1
            if len(warning_lines) < 20:
                warning_lines.append(line[-1000:])
            if not is_known_nonfatal and len(actionable_warning_lines) < 20:
                actionable_warning_lines.append(line[-1000:])
        if is_error:
            error_count += 1
            if len(error_lines) < 20:
                error_lines.append(line[-1000:])
            if not is_known_nonfatal and len(actionable_error_lines) < 20:
                actionable_error_lines.append(line[-1000:])
        if (is_warning or is_error) and is_known_nonfatal:
            known_nonfatal_count += 1
            known_nonfatal_warning_count += int(is_warning)
            known_nonfatal_error_count += int(is_error)
            if len(known_nonfatal_lines) < 20:
                known_nonfatal_lines.append(line[-1000:])
    return {
        "warnings": warning_count,
        "errors": error_count,
        "known_nonfatal": known_nonfatal_count,
        "actionable_warnings": warning_count - known_nonfatal_warning_count,
        "actionable_errors": error_count - known_nonfatal_error_count,
        "warning_examples": warning_lines,
        "error_examples": error_lines,
        "known_nonfatal_examples": known_nonfatal_lines,
        "actionable_warning_examples": actionable_warning_lines,
        "actionable_error_examples": actionable_error_lines,
    }


@dataclass(frozen=True)
class RuntimeConfig:
    ckpt: Path
    host: str
    base_port: int
    num_servers: int
    worker_index: int
    worker_count: int
    shared_run_id: str
    robodojo_root: Path
    asset_root: Path
    staging_root: Path
    output_root: Path
    video_root: Path
    num_episodes: str
    num_envs: int
    save_video: bool
    task_max_retries: int
    task_inactivity_timeout: int
    proxy_start_timeout: int
    seed: int
    unnorm_key: str
    action_chunk_size: int
    execute_horizon: int
    server_metadata: dict[str, Any]
    proxy_python: Path


def terminate_process_group(process: subprocess.Popen[Any]) -> None:
    """Stop a process group even if its leader exits before its children."""
    process_group_id = process.pid
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def contains_meaningful_eval_output(chunk: bytes) -> bool:
    text = chunk.decode("utf-8", errors="replace")
    for line in re.split(r"[\r\n]+", text):
        normalized = line.strip().lower()
        if normalized and not any(marker in normalized for marker in NON_PROGRESS_LOG_MARKERS):
            return True
    return False


def run_logged_process(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    output_stream: Any,
    output_path: Path,
    inactivity_timeout: int,
) -> int:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=output_stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        last_size = output_path.stat().st_size
    except FileNotFoundError:
        last_size = 0
    last_activity = time.monotonic()
    try:
        while True:
            try:
                return process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    stat = output_path.stat()
                except FileNotFoundError:
                    stat = None
                if stat is not None and stat.st_size != last_size:
                    if stat.st_size < last_size:
                        last_size = 0
                    with output_path.open("rb") as stream:
                        stream.seek(last_size)
                        chunk = stream.read()
                    last_size = stat.st_size
                    if contains_meaningful_eval_output(chunk):
                        last_activity = time.monotonic()
                        continue
                inactive_for = time.monotonic() - last_activity
                if inactive_for < inactivity_timeout:
                    continue
                output_stream.write(
                    f"[run_client] no evaluation output for {int(inactive_for)}s; "
                    "terminating the task so it can resume from the last completed batch.\n"
                )
                output_stream.flush()
                terminate_process_group(process)
                raise TimeoutError(
                    f"evaluation produced no output for {int(inactive_for)}s " f"(limit={inactivity_timeout}s)"
                ) from None
    except BaseException:
        terminate_process_group(process)
        raise


def find_result(config: RuntimeConfig, task_name: str, task_run_id: str) -> Path:
    candidates = list(config.staging_root.glob(f"RoboDojo/{task_name}/Ego4WAM/arx_x5/**/{task_run_id}/_result.json"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one result for task={task_name}, run_id={task_run_id}, got {candidates}")
    return candidates[0]


def run_task(
    config: RuntimeConfig,
    task_name: str,
    canonical_name: str,
    task_id: int,
    total_tasks: int,
    config_id: int,
    total_configs: int,
    gpu_id: str,
    server_index: int,
    proxy_port: int,
    slot_name: str,
) -> dict[str, Any]:
    server_port = config.base_port + server_index
    task_run_id = sanitize(f"{config.shared_run_id}_w{config.worker_index}_{slot_name}_{task_name}")
    log_dir = config.output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    eval_log = log_dir / f"{task_name}.eval.log"
    proxy_log = log_dir / f"{task_name}.proxy.log"
    status_path = config.output_root / "task_status" / f"{task_name}.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)

    task_status: dict[str, Any] = {
        "task_name": task_name,
        "canonical_task_name": canonical_name,
        "task_id": task_id,
        "total_tasks": total_tasks,
        "config_id": config_id,
        "total_configs": total_configs,
        "worker_index": config.worker_index,
        "gpu_id": gpu_id,
        "server_index": server_index,
        "server_port": server_port,
        "proxy_port": proxy_port,
        "run_id": task_run_id,
        "num_envs": config.num_envs,
        "attempts": 0,
        "success": False,
    }
    base_env = os.environ.copy()
    base_env.update(
        {
            "PATH": f"{Path(sys.executable).parent}:{base_env.get('PATH', '')}",
            "PYTHONUNBUFFERED": "1",
            "ROBODOJO_ASSETS_PATH": str(config.asset_root),
            "ROBODOJO_EVAL_RESULT_ROOT": str(config.staging_root),
            "ROBODOJO_SAVE_VIDEO": "1" if config.save_video else "0",
            "ROBODOJO_VIDEO_ROOT": str(config.video_root),
            "ROBODOJO_TASK_ID": task_name,
            "ROBODOJO_RUN_ID": task_run_id,
            "ROBODOJO_MAX_BASH_RETRIES": os.getenv("ROBODOJO_MAX_BASH_RETRIES", "3"),
            "ROBODOJO_ROOT": str(config.robodojo_root),
            "PYTHONPATH": f"{REPO_ROOT}:{base_env.get('PYTHONPATH', '')}",
        }
    )
    if config.num_episodes == "native":
        base_env.pop("EVAL_NUM", None)
    else:
        base_env["EVAL_NUM"] = config.num_episodes
    eval_env = base_env.copy()
    eval_env.update(
        {
            "PYTHONPATH": (
                f"{config.robodojo_root}:{config.robodojo_root / 'XPolicyLab'}:" f"{base_env.get('PYTHONPATH', '')}"
            ),
            "CUDA_VISIBLE_DEVICES": gpu_id,
            "ROBODOJO_EVAL_BATCH": "1" if config.num_envs > 1 else "0",
            "ROBODOJO_REQUIRE_HEAD_EXTRINSIC": "1",
        }
    )
    proxy_command = [
        str(config.proxy_python),
        str(RAW_PROXY_BOOTSTRAP),
        "--config_path",
        str(SCRIPT_DIR.parents[0] / "policy_proxy_raw/policy_modules/Ego4WAM/deploy.yml"),
        "--host",
        "127.0.0.1",
        "--port",
        str(proxy_port),
        "--overrides",
        "bench_name=RoboDojo",
        f"task_name={task_name}",
        f"checkpoint_path={config.ckpt}",
        "env_cfg_type=arx_x5",
        "policy_name=Ego4WAM",
        "action_type=joint",
        "action_dim=14",
        f"eval_batch={config.num_envs > 1}",
        f"unnorm_key={config.unnorm_key}",
        f"execute_horizon={config.execute_horizon}",
        f"ego4wam_server_host={config.host}",
        f"ego4wam_server_port={server_port}",
        "ws_ping_interval_s=None",
        "ws_ping_timeout_s=None",
    ]
    eval_command = [
        "bash",
        str(SCRIPT_DIR / "eval_entry.sh"),
        sys.executable,
        str(SCRIPT_DIR / "eval_entry.py"),
        "--task_name",
        task_name,
        "--env_cfg_type",
        "arx_x5",
        "--num_envs",
        str(config.num_envs),
        "--device_id",
        "0",
        "--policy_name",
        "Ego4WAM",
        "--port",
        str(proxy_port),
        "--host",
        "127.0.0.1",
        "--additional_info",
        f"ckpt_name={config.ckpt.stem},action_type=joint",
        "--seed",
        str(config.seed),
        "--protocol",
        "ws",
        "--policy_server_url",
        f"ws://127.0.0.1:{proxy_port}",
        "--enable_cameras",
        "--kit_args",
        " --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera",
        "--headless",
    ]

    last_error = ""
    for attempt in range(config.task_max_retries + 1):
        task_status["attempts"] = attempt + 1
        proxy_process = None
        log(
            f"START task={task_name} gpu={gpu_id} server={server_port} " f"slot={slot_name} attempt={attempt + 1}",
            task_id=task_id,
            total_tasks=total_tasks,
            config_id=config_id,
            total_configs=total_configs,
        )
        try:
            ensure_port_available("127.0.0.1", proxy_port)
            with proxy_log.open("a", encoding="utf-8") as proxy_stream:
                proxy_stream.write(f"\n===== attempt {attempt + 1} =====\n")
                proxy_stream.flush()
                proxy_process = subprocess.Popen(
                    proxy_command,
                    cwd=config.robodojo_root,
                    env=base_env,
                    stdout=proxy_stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                wait_for_port(
                    "127.0.0.1",
                    proxy_port,
                    proxy_process,
                    config.proxy_start_timeout,
                )
                with eval_log.open("a", encoding="utf-8") as eval_stream:
                    eval_stream.write(f"\n===== attempt {attempt + 1} =====\n")
                    eval_stream.flush()
                    return_code = run_logged_process(
                        eval_command,
                        cwd=config.robodojo_root,
                        env=eval_env,
                        output_stream=eval_stream,
                        output_path=eval_log,
                        inactivity_timeout=config.task_inactivity_timeout,
                    )
            if return_code != 0:
                raise RuntimeError(f"evaluation exited with status {return_code}")

            result_path = find_result(config, task_name, task_run_id)
            result_payload = json.loads(result_path.read_text(encoding="utf-8"))
            expected_episodes = task_weights(config.robodojo_root, [task_name], config.num_episodes)[task_name]
            coverage = validate_result(result_payload, expected_episodes)
            if config.save_video:
                video_paths = collect_task_videos(config.video_root, task_name, expected_episodes)
                task_status["videos"] = [str(path.relative_to(config.output_root)) for path in video_paths]
            final_result_path = config.output_root / "results" / f"{task_name}.json"
            final_result_path.parent.mkdir(parents=True, exist_ok=True)
            final_result_path.write_text(
                json.dumps(result_payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            task_status["success"] = True
            task_status["coverage"] = coverage
            task_status["result"] = {
                "success_rate": result_payload.get("success_rate"),
                "eval_time": result_payload.get("eval_time"),
                "score": result_payload.get("score"),
            }
            log(
                f"DONE task={task_name} gpu={gpu_id} server={server_port} "
                f"success_rate={result_payload.get('success_rate')}",
                task_id=task_id,
                total_tasks=total_tasks,
                config_id=config_id,
                total_configs=total_configs,
            )
            break
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            log(
                f"FAIL task={task_name} attempt={attempt + 1}: {last_error}",
                task_id=task_id,
                total_tasks=total_tasks,
                config_id=config_id,
                total_configs=total_configs,
            )
            if attempt >= config.task_max_retries:
                task_status["error"] = last_error
        finally:
            if proxy_process is not None:
                terminate_process_group(proxy_process)

    task_status["eval_log"] = analyze_log(eval_log)
    task_status["proxy_log"] = analyze_log(proxy_log)
    status_path.write_text(
        json.dumps(task_status, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return task_status


def wait_for_workers(output_root: Path, worker_count: int, max_wait: int) -> list[dict[str, Any]]:
    status_dir = output_root / "worker_status"
    deadline = time.monotonic() + max_wait
    while True:
        statuses = []
        for worker_index in range(worker_count):
            path = status_dir / f"worker_{worker_index}.json"
            if not path.is_file():
                break
            try:
                statuses.append(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                break
        if len(statuses) == worker_count:
            return statuses
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for workers: {len(statuses)}/{worker_count}")
        log(f"waiting for workers: {len(statuses)}/{worker_count}")
        time.sleep(5)


def allocate_proxy_ports(host: str, base_port: int, count: int) -> list[int]:
    sockets = []
    ports = []
    try:
        for port in range(base_port, base_port + 10000):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind((host, port))
            except OSError:
                sock.close()
                continue
            sockets.append(sock)
            ports.append(port)
            if len(ports) == count:
                return ports
    finally:
        for sock in sockets:
            sock.close()
    raise OSError(f"cannot allocate {count} proxy ports from {host}:{base_port}")


def shutdown_servers(config: RuntimeConfig) -> int:
    if not env_bool("SHUTDOWN_SERVERS", False):
        return 0
    shutdown_token = os.getenv("POLICY_SERVER_SHUTDOWN_TOKEN", "")
    if not shutdown_token:
        log("POLICY_SERVER_SHUTDOWN_TOKEN is empty; leaving servers running")
        return 0
    command = [
        sys.executable,
        str(SHUTDOWN_SCRIPT),
        "--host",
        config.host,
        "--base-port",
        str(config.base_port),
        "--num-servers",
        str(config.num_servers),
        "--shutdown-token",
        shutdown_token,
    ]
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


def main() -> int:
    if env_bool("ROBODOJO_EEF_CONTROL", False) or env_bool("ROBODOJO_SEND_SIM_EE_POSE", False):
        raise NotImplementedError(
            "eef_ee_poses evaluation is reserved but not supported by the raw "
            "RoboDojo client; unset ROBODOJO_EEF_CONTROL and "
            "ROBODOJO_SEND_SIM_EE_POSE"
        )
    ckpt = Path(os.environ["CKPT"]).resolve()
    host = os.environ["HOST"]
    base_port = env_int("BASE_PORT", 6789)
    num_servers = env_int("NUM_SERVERS", 1)
    worker_index, worker_count = parse_worker_info()
    shared_run_id = os.environ["CLIENT_SHARED_RUN_ID"].strip()
    if not shared_run_id:
        raise ValueError("CLIENT_SHARED_RUN_ID must not be empty")
    robodojo_root = Path(os.environ["ROBODOJO_ROOT"]).expanduser().resolve()
    asset_root = Path(os.environ["ROBODOJO_ASSET_ROOT"]).expanduser().resolve()
    if not ckpt.is_dir() or not (ckpt / "model.safetensors").is_file():
        raise FileNotFoundError("CKPT must be an exported model directory containing " f"model.safetensors: {ckpt}")
    if not robodojo_root.is_dir():
        raise FileNotFoundError(f"RoboDojo root does not exist: {robodojo_root}")
    if not asset_root.is_dir():
        raise FileNotFoundError(f"RoboDojo asset root does not exist: {asset_root}")
    if not (asset_root / "Robots/x5/curobo.yml").is_file():
        raise FileNotFoundError(
            "Missing generated Robots/x5/curobo.yml; link the downloaded Assets "
            "into RoboDojo and run utils/update_embodiment_config_path.py first"
        )
    image_asset_path = robodojo_root / "Assets"
    if not image_asset_path.exists():
        try:
            image_asset_path.symlink_to(asset_root, target_is_directory=True)
        except PermissionError as exc:
            raise PermissionError(f"cannot create {image_asset_path}; mount ROBODOJO_ASSET_ROOT there") from exc

    default_output_root = REPO_ROOT / "results" / "ego4wam_robodojo" / shared_run_id
    output_root = Path(os.getenv("OUTPUT_ROOT") or default_output_root).expanduser()
    video_root = output_root / "video"
    staging_root = Path(
        os.getenv("ROBODOJO_STAGING_ROOT")
        or REPO_ROOT / "tmp" / "robodojo_horizon" / shared_run_id / f"worker_{worker_index}" / "eval_result"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    video_root.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)

    num_episodes = os.getenv("NUM_EPISODES", os.getenv("EVAL_NUM", "native")).strip().lower()
    if num_episodes != "native" and int(num_episodes) < 1:
        raise ValueError(f"NUM_EPISODES must be native or positive, got {num_episodes}")
    num_envs = env_int("NUM_ENVS", 10)
    if num_envs < 1:
        raise ValueError(f"NUM_ENVS must be positive, got {num_envs}")
    unnorm_key = os.getenv("UNNORM_KEY", "robodojo_arx_x5").strip()
    if not unnorm_key:
        raise ValueError("UNNORM_KEY must not be empty")
    metadata_list = wait_for_server_metadata(
        host,
        base_port,
        num_servers,
        env_int("SERVER_MAX_WAIT", 1800),
    )
    server_metadata, action_chunk_size = resolve_server_contract(
        metadata_list,
        ckpt=ckpt,
        unnorm_key=unnorm_key,
    )
    all_configs, config_dimensions = load_inventory(robodojo_root)
    selected_configs = parse_task_names(all_configs)
    execute_horizon = env_int("EXECUTE_HORIZON", 32)
    if execute_horizon < 1:
        raise ValueError(f"EXECUTE_HORIZON must be positive, got {execute_horizon}")
    if execute_horizon > action_chunk_size:
        raise ValueError(f"EXECUTE_HORIZON={execute_horizon} exceeds action_chunk_size={action_chunk_size}")
    config_ids = {config_name: config_id for config_id, config_name in enumerate(selected_configs, start=1)}
    config_to_task = {config_name: canonical_task_name(config_name) for config_name in selected_configs}
    canonical_tasks = list(dict.fromkeys(config_to_task.values()))
    task_ids = {task_name: task_id for task_id, task_name in enumerate(canonical_tasks, start=1)}
    canonical_task_dimensions = {}
    for config_name in selected_configs:
        canonical_task_dimensions.setdefault(config_to_task[config_name], config_dimensions.get(config_name, "unknown"))
    weights = task_weights(robodojo_root, selected_configs, num_episodes)
    partitions = partition_tasks(selected_configs, weights, worker_count)
    worker_configs = partitions[worker_index]

    gpu_ids = [item.strip() for item in os.getenv("CLIENT_CUDA_VISIBLE_DEVICES", "2").split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("CLIENT_CUDA_VISIBLE_DEVICES must contain at least one GPU")
    tasks_per_gpu = env_int("TASKS_PER_GPU", 1)
    if tasks_per_gpu < 1:
        raise ValueError("TASKS_PER_GPU must be positive")
    task_inactivity_timeout = env_int("TASK_INACTIVITY_TIMEOUT", 1200)
    if task_inactivity_timeout < 1:
        raise ValueError("TASK_INACTIVITY_TIMEOUT must be positive")

    config = RuntimeConfig(
        ckpt=ckpt,
        host=host,
        base_port=base_port,
        num_servers=num_servers,
        worker_index=worker_index,
        worker_count=worker_count,
        shared_run_id=shared_run_id,
        robodojo_root=robodojo_root,
        asset_root=asset_root,
        staging_root=staging_root,
        output_root=output_root,
        video_root=video_root,
        num_episodes=num_episodes,
        num_envs=num_envs,
        save_video=env_bool("SAVE_VIDEO", True),
        task_max_retries=env_int("TASK_MAX_RETRIES", 1),
        task_inactivity_timeout=task_inactivity_timeout,
        proxy_start_timeout=env_int("PROXY_START_TIMEOUT", 120),
        seed=env_int("SEED", 0),
        unnorm_key=unnorm_key,
        action_chunk_size=action_chunk_size,
        execute_horizon=execute_horizon,
        server_metadata=server_metadata,
        proxy_python=Path(os.getenv("ROBODOJO_PROXY_PYTHON", sys.executable)),
    )
    if not config.proxy_python.is_file():
        raise FileNotFoundError(f"proxy Python does not exist: {config.proxy_python}")

    if worker_index == 0:
        manifest = {
            "run_id": shared_run_id,
            "seed": config.seed,
            "baseline": "historical_joint",
            "ckpt": str(ckpt),
            "host": host,
            "base_port": base_port,
            "num_servers": num_servers,
            "worker_count": worker_count,
            "gpus_per_worker": len(gpu_ids),
            "tasks_per_gpu": tasks_per_gpu,
            "num_episodes": num_episodes,
            "num_envs": num_envs,
            "save_video": config.save_video,
            "task_inactivity_timeout": config.task_inactivity_timeout,
            "unnorm_key": config.unnorm_key,
            "action_chunk_size": config.action_chunk_size,
            "execute_horizon": config.execute_horizon,
            "server_metadata": config.server_metadata,
            "output_root": str(config.output_root),
            "video_root": str(config.video_root),
            "configs": selected_configs,
            "config_ids": config_ids,
            "config_to_task": config_to_task,
            "config_dimensions": {
                config_name: config_dimensions.get(config_name, "unknown") for config_name in selected_configs
            },
            "canonical_tasks": canonical_tasks,
            "task_ids": task_ids,
            "task_dimensions": canonical_task_dimensions,
            "tasks": selected_configs,
            "task_weights": weights,
            "partitions": partitions,
        }
        (output_root / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    log(
        f"worker={worker_index}/{worker_count} configs={len(worker_configs)}/{len(selected_configs)} "
        f"benchmark_tasks={len(canonical_tasks)} gpus={gpu_ids} tasks_per_gpu={tasks_per_gpu} "
        f"num_envs={num_envs} unnorm_key={config.unnorm_key} "
        f"action_chunk_size={config.action_chunk_size} execute_horizon={config.execute_horizon} "
        f"output={output_root}"
    )

    total_tasks = len(canonical_tasks)
    total_configs = len(selected_configs)
    task_queue: queue.Queue[tuple[int, int, str, str]] = queue.Queue()
    for config_name in worker_configs:
        canonical_name = config_to_task[config_name]
        task_queue.put((task_ids[canonical_name], config_ids[config_name], canonical_name, config_name))
    task_statuses: list[dict[str, Any]] = []
    status_lock = threading.Lock()
    threads = []
    slot_count = len(gpu_ids) * tasks_per_gpu
    proxy_ports = allocate_proxy_ports(
        "127.0.0.1",
        env_int("PROXY_BASE_PORT", 20000),
        slot_count,
    )

    def slot_worker(gpu_id: str, gpu_ordinal: int, local_slot: int, slot_index: int) -> None:
        server_index = (
            worker_index * len(gpu_ids) * tasks_per_gpu + gpu_ordinal * tasks_per_gpu + local_slot
        ) % num_servers
        proxy_port = proxy_ports[slot_index]
        slot_name = f"g{gpu_id}_slot{local_slot}"
        while True:
            try:
                task_id, config_id, canonical_name, task_name = task_queue.get_nowait()
            except queue.Empty:
                return
            try:
                status = run_task(
                    config,
                    task_name,
                    canonical_name,
                    task_id,
                    total_tasks,
                    config_id,
                    total_configs,
                    gpu_id,
                    server_index,
                    proxy_port,
                    slot_name,
                )
                with status_lock:
                    task_statuses.append(status)
            finally:
                task_queue.task_done()

    slot_index = 0
    for gpu_ordinal, gpu_id in enumerate(gpu_ids):
        for local_slot in range(tasks_per_gpu):
            thread = threading.Thread(
                target=slot_worker,
                args=(gpu_id, gpu_ordinal, local_slot, slot_index),
                name=f"robodojo-gpu{gpu_id}-slot{local_slot}",
            )
            thread.start()
            threads.append(thread)
            slot_index += 1
    for thread in threads:
        thread.join()

    worker_success = all(status.get("success") for status in task_statuses) and len(task_statuses) == len(worker_configs)
    failed_configs = sorted(status["task_name"] for status in task_statuses if not status.get("success"))
    failed_tasks = sorted({status["canonical_task_name"] for status in task_statuses if not status.get("success")})
    worker_status = {
        "worker_index": worker_index,
        "configs": worker_configs,
        "canonical_tasks": list(dict.fromkeys(config_to_task[config_name] for config_name in worker_configs)),
        "tasks": worker_configs,
        "completed": len(task_statuses),
        "success": worker_success,
        "failed_configs": failed_configs,
        "failed_tasks": failed_tasks,
    }
    worker_status_dir = output_root / "worker_status"
    worker_status_dir.mkdir(parents=True, exist_ok=True)
    (worker_status_dir / f"worker_{worker_index}.json").write_text(
        json.dumps(worker_status, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log(f"worker {worker_index} complete success={worker_success}")

    if worker_index != 0:
        return 0 if worker_success else 1

    statuses = wait_for_workers(
        output_root,
        worker_count,
        env_int("CLIENT_WORKER_SYNC_MAX_WAIT", 14400),
    )
    summary_status = subprocess.run(
        [sys.executable, str(SUMMARY_SCRIPT), "--root", str(output_root)],
        cwd=REPO_ROOT,
        check=False,
    ).returncode
    shutdown_status = shutdown_servers(config)
    all_workers_success = all(status.get("success") for status in statuses)
    return 0 if all_workers_success and summary_status == 0 and shutdown_status == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"FATAL {type(exc).__name__}: {exc}")
        raise
