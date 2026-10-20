# Copyright (c) 2026 Beijing Horizon Information Technology Company Limited.
# Author: Ego4WAM Team.
# SPDX-License-Identifier: MIT AND Apache-2.0
# Derived from the historical RoboDojo XPolicyLab starVLA adapter.
# Ego4WAM modifications: MIT; inherited XPolicyLab portions: Apache-2.0.
# See LICENSES/XPolicyLab-Apache-2.0.txt for the retained upstream license.
# Packaged independently for Ego4WAM without changing the rollout loop.

import json
import os
import time

_episode_index = 0


def _next_episode_index():
    global _episode_index
    current_episode_index = _episode_index
    _episode_index += 1
    return current_episode_index


def _emit_rollout_timing(mode, episode_index, action_step_ns, pose_capture_ns, pose_update_ns):
    total_rollout_step_ns = [
        action_ns + capture_ns + update_ns
        for action_ns, capture_ns, update_ns in zip(action_step_ns, pose_capture_ns, pose_update_ns)
    ]
    payload = {
        "variant": os.getenv("ROBODOJO_BENCHMARK_VARIANT", "unknown"),
        "mode": mode,
        "pid": os.getpid(),
        "episode_index": episode_index,
        "count": len(total_rollout_step_ns),
        "action_step_ns": action_step_ns,
        "pose_capture_ns": pose_capture_ns,
        "pose_update_ns": pose_update_ns,
        "total_rollout_step_ns": total_rollout_step_ns,
    }
    print("ROBODOJO_ROLLOUT_TIMING " + json.dumps(payload, separators=(",", ":")), flush=True)


def eval_one_episode(TASK_ENV, model_client):
    episode_index = _next_episode_index()
    action_step_ns = []
    pose_capture_ns = []
    pose_update_ns = []
    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end():
        obs = TASK_ENV.get_obs()
        model_client.call(func_name="update_obs", obs=obs)
        actions = model_client.call(func_name="get_action")

        for action_idx, action in enumerate(actions):
            action_start_ns = time.perf_counter_ns()
            TASK_ENV.take_action(action)
            action_end_ns = time.perf_counter_ns()
            action_duration_ns = action_end_ns - action_start_ns
            capture_duration_ns = 0
            update_duration_ns = 0
            episode_ended = TASK_ENV.is_episode_end()
            chunk_ended = action_idx + 1 == len(actions)

            if episode_ended or chunk_ended:
                action_step_ns.append(action_duration_ns)
                pose_capture_ns.append(capture_duration_ns)
                pose_update_ns.append(update_duration_ns)
                break

            capture_camera_pose = action_idx + 2 == len(actions)
            capture_start_ns = time.perf_counter_ns()
            pose_observations = TASK_ENV.record_video_frame_batch([0], capture_camera_pose=capture_camera_pose)
            capture_end_ns = time.perf_counter_ns()
            capture_duration_ns = capture_end_ns - capture_start_ns
            if capture_camera_pose:
                update_start_ns = time.perf_counter_ns()
                model_client.call(func_name="update_camera_pose_batch", obs=pose_observations)
                update_end_ns = time.perf_counter_ns()
                update_duration_ns = update_end_ns - update_start_ns
            action_step_ns.append(action_duration_ns)
            pose_capture_ns.append(capture_duration_ns)
            pose_update_ns.append(update_duration_ns)

    _emit_rollout_timing("single", episode_index, action_step_ns, pose_capture_ns, pose_update_ns)


def eval_one_episode_batch(TASK_ENV, model_client):
    episode_index = _next_episode_index()
    action_step_ns = []
    pose_capture_ns = []
    pose_update_ns = []
    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end():
        env_idx_list = TASK_ENV.get_running_env_idx_list()
        obs_list = TASK_ENV.get_obs_batch(env_idx_list)
        model_client.call(func_name="update_obs_batch", obs=obs_list)
        actions = model_client.call(func_name="get_action_batch", obs=env_idx_list)

        chunk_size = len(actions[0])
        for action_idx in range(chunk_size):
            current_action_list = [env_actions[action_idx] for env_actions in actions]
            action_start_ns = time.perf_counter_ns()
            TASK_ENV.take_action_batch(current_action_list, env_idx_list)
            action_end_ns = time.perf_counter_ns()
            action_duration_ns = action_end_ns - action_start_ns
            capture_duration_ns = 0
            update_duration_ns = 0
            episode_ended = TASK_ENV.is_episode_end()
            chunk_ended = action_idx + 1 == chunk_size

            if episode_ended or chunk_ended:
                action_step_ns.append(action_duration_ns)
                pose_capture_ns.append(capture_duration_ns)
                pose_update_ns.append(update_duration_ns)
                break

            running = set(TASK_ENV.get_running_env_idx_list())
            active_batch_idx = [i for i, env_idx in enumerate(env_idx_list) if env_idx in running]

            actions = [actions[i] for i in active_batch_idx]
            env_idx_list = [env_idx_list[i] for i in active_batch_idx]
            capture_camera_pose = action_idx + 2 == chunk_size
            capture_start_ns = time.perf_counter_ns()
            pose_observations = TASK_ENV.record_video_frame_batch(env_idx_list, capture_camera_pose=capture_camera_pose)
            capture_end_ns = time.perf_counter_ns()
            capture_duration_ns = capture_end_ns - capture_start_ns
            if capture_camera_pose:
                update_start_ns = time.perf_counter_ns()
                model_client.call(func_name="update_camera_pose_batch", obs=pose_observations)
                update_end_ns = time.perf_counter_ns()
                update_duration_ns = update_end_ns - update_start_ns
            action_step_ns.append(action_duration_ns)
            pose_capture_ns.append(capture_duration_ns)
            pose_update_ns.append(update_duration_ns)

    _emit_rollout_timing("batch", episode_index, action_step_ns, pose_capture_ns, pose_update_ns)
