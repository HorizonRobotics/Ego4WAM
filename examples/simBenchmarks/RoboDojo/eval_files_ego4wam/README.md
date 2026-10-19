<!--
Copyright (c) 2026 Beijing Horizon Information Technology Company Limited.
Author: Ego4WAM Team.
SPDX-License-Identifier: MIT
-->

# Ego4WAM RoboDojo Evaluation

See the [complete setup and reproduction guide](../../../../docs/RoboDojo.md)
for source revisions, asset downloads and environment installation.

This entrypoint uses the historical RoboDojo version used in the paper, not the
current RoboDojo leaderboard protocol. Use the released HorizonRobotics/Ego4WAM Joint
weights and the source revisions checked by `prepare_runtime.py`. Do not run
against a latest RoboDojo checkout or add later camera/prompt compensations.

The model, simulator and XPolicyLab proxy use separate Python environments.
`requirements-model.txt`, `requirements-simulator.txt` and
`requirements-proxy.txt` pin their direct runtime dependencies. They are not
system/extension locks. `constraints-model.txt` records the public-source model
environment that passed strict loading and two-GPU inference;
`constraints-simulator.txt` pins the clean public SDK/Python installation that
passed dependency/startup checks and a ten-episode end-to-end task smoke. Install Isaac
Sim 5.1.0 and the pinned IsaacLab/cuRobo forks
separately; check the reproduction guide for installation validation status.

## Prepare the sources

Inspect `patches/`, then run this from the Ego4WAM/StarVLA repository root:

```bash
python examples/simBenchmarks/RoboDojo/eval_files_ego4wam/prepare_runtime.py \
  --robodojo-root /path/to/RoboDojo --apply
```

This checks four Git revisions and applies only the historical runtime patches.
It will not reset or check out a user's worktree. The client repeats a read-only
check before starting. The independent `policy_proxy_raw/` adapter is loaded
through the pinned XPolicyLab without modifying its starVLA policy.
The client also prefixes the checked IsaacLab/cuRobo source directories on
`PYTHONPATH`; unrelated globally installed fork versions cannot replace them.

## Start the model service

```bash
export CKPT=/path/to/downloaded/Ego4WAM/Ego4WAM_VideoPretrained_RoboDojoJoint
export EGO4WAM_PYTHON=/path/to/model-env/bin/python
export SERVER_CUDA_VISIBLE_DEVICES=0,1
export PORT=6789
bash examples/simBenchmarks/RoboDojo/eval_files_ego4wam/local/local_server.sh
```

The existing Ego4WAM loader strict-loads the release. The dedicated transport
keeps asynchronous microbatching and unseeded global model RNG. The current
release default uses capacity 10 and a 20-ms collection window; the single-task
validation's actual requests contain at most ten active environments. The historical
full-suite command recorded capacity 32, whereas the later single-task script
defaulted to 10. The release intentionally keeps 10/20ms; do not claim the later
script proves the original server launch arguments. `MAX_BATCH_SIZE=32` is an
explicit override for testing that historical full-suite command. There is no
second checkpoint loader.

## Check one task (in another terminal)

```bash
export CKPT=/path/to/downloaded/Ego4WAM/Ego4WAM_VideoPretrained_RoboDojoJoint
export ROBODOJO_ROOT=/path/to/patched/RoboDojo
export ROBODOJO_ASSET_ROOT=/path/to/Assets
export ROBODOJO_PYTHON=/path/to/simulator-env/bin/python
export ROBODOJO_PROXY_PYTHON=/path/to/proxy-env/bin/python
export HOST=127.0.0.1 BASE_PORT=6789
export TASK_NAMES=build_tower SEED=0 NUM_EPISODES=50 NUM_ENVS=10
export EXECUTE_HORIZON=32 SAVE_VIDEO=1 CLIENT_CUDA_VISIBLE_DEVICES=2
export CLIENT_SHARED_RUN_ID=joint_build_tower_seed0
export OUTPUT_ROOT="$PWD/results/ego4wam_robodojo/$CLIENT_SHARED_RUN_ID"
bash examples/simBenchmarks/RoboDojo/eval_files_ego4wam/local/local_client.sh
```

Use a new run ID/output directory for a new experiment. Reuse the same staging
directory and ID only to intentionally resume an interrupted run. Do not point
different workers at different experiments under one run ID.

For a one-batch functional smoke check, set `NUM_EPISODES=10` with
`NUM_ENVS=10` and a distinct run ID. This checks the end-to-end path without
repeating the 50-episode evaluation; do not compare its success rate directly
with the historical 50-episode reference.

The historical comparison is 34/50 successes (68%), score 73.4. Seed 0 selects
the simulator layouts, not a particular model RNG stream; a single-task run
does not exactly repeat a concurrently scheduled full-benchmark run. Report
actual outcomes, not a promised success rate.

The migrated seed-0 / 50-episode check completed with **36/50 (72%)**, score
**77.0**, versus the historical **34/50 (68%)**, score **73.4**. All episode
details/layouts and 150 three-camera videos passed coverage/metadata checks.
This validates one task using freshly prepared pinned sources and the
historical installed simulator SDK; it does not establish a fresh SDK
installation or a full-suite 14.35% reproduction.

Separately, the newly installed public model, simulator and proxy environments
completed one ten-environment batch with **9/10 successes**, score **91.0**.
All ten episode/layout records and 30 H.264 camera videos passed validation.
This checks the clean Python environments end to end, not a fresh operating
system or a second 50-episode success-rate comparison.

## Full local benchmark

Use the same service and client settings, unset `TASK_NAMES`, set
`TASK_SET_NAME=FULL_TASKS`, `NUM_EPISODES=native`, and set
`CLIENT_CUDA_VISIBLE_DEVICES` to the available simulator GPUs. The scheduler
maps 54 configurations to 42 canonical tasks. Standard/random configurations
are episode-weighted, then tasks and five capability dimensions are averaged.
The reference full aggregate is 14.35% over 2,100 episodes; a one-task aggregate
must not be reported as a full-benchmark result.

The run writes `run_manifest.json`, `task_status/`, `results/`, `logs/`,
`aggregate.json`, `aggregate.txt` and `video/<task>/episode_<7 digits>/` under
`OUTPUT_ROOT`. Each episode has `head.mp4`, `left_wrist.mp4` and
`right_wrist.mp4`. A task only counts as complete after episode coverage,
per-episode metrics and all three nonempty videos are validated. Partial runs
return nonzero; their aggregate is explicitly partial. Reaggregate with:

```bash
PYTHONPATH="$PWD" "$ROBODOJO_PYTHON" \
  examples/simBenchmarks/RoboDojo/eval_files_ego4wam/common/summarize_results.py \
  --root "$OUTPUT_ROOT"
```

Stop the service with Ctrl+C after the client finishes. Clients do not shut down
a shared service by default. Remote shutdown requires an explicitly configured
shared token.
