# Ego4WAM Joint: RoboDojo Evaluation

This guide targets the published Joint weights and the historical local evaluation used for the paper. It does **not** target the newer synchronized-data retraining or the current leaderboard protocol.

Status: local evaluation migration and one-task validation completed. The
historical-SDK 50-episode check passed, and fresh public-source model, proxy and
simulator environments passed dependency checks and a separate ten-episode
end-to-end smoke. This is not a full-suite performance reproduction. No image
will be published.

## Model and reference result

Use [HorizonRobotics/Ego4WAM](https://huggingface.co/HorizonRobotics/Ego4WAM/tree/262306ba29e9a4581ac5f714b90711498917444f), revision `262306ba29e9a4581ac5f714b90711498917444f`.

The released `model.safetensors` is 26,913,478,024 bytes, with SHA256:

```text
ae1b4595e1909e4573ad3214e1c5112f1920a34675340f6adfd4f2e9ec6c12c4
```

The release contains `README.md`, `LICENSE`, `inference_config.yaml`, `model.safetensors`, and `model_assets/`. The bundled text encoder, tokenizer and VAE are loaded through the existing Ego4WAM loader; do not require an internal training manifest or a second model-loading implementation. RoboDojo normalization and kinematic assets are included in the code package.

The historical full evaluation reports 14.35% dimension-balanced success rate and 20.29 average score over 42 canonical tasks / 54 runnable configurations / 2,100 episodes. Standard and random configurations are merged using episode counts; tasks are averaged within each of the five dimensions, followed by an equal-weight dimension average. A one-task check does not reproduce the full benchmark aggregate.

The `build_tower` reference is **34/50 successes (68%)**, score **73.4**, environment seed **0**, with ten simulator environments. Predict 32 actions, execute 32 actions before replanning, and use the fixed 20-step Joint Euler sampler. Save all three camera videos.

## Source revisions

| Component | Source | Revision |
| --- | --- | --- |
| RoboDojo | [RoboDojo](https://github.com/robodojo-benchmark/RoboDojo) | `9b4cc885e8f530ed3ab14a30a312ae70242771c4` |
| XPolicyLab | [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) | `fe71eb54675cef495fea817a637386a4f4529153` |
| IsaacLab fork | [yuechen0614/IsaacLab](https://github.com/yuechen0614/IsaacLab) | `afca7b09d60d8beb9c1cb28b43066499940b969b` |
| cuRobo fork | [yuechen0614/curobo](https://github.com/yuechen0614/curobo) | `d17b54ce32cba095c0b000c4c58777075d11de0e` |

Do not use `git submodule update --remote` or replace these revisions with the latest branches.

These revisions alone are **not** the complete historical runtime: it also included local source changes for raw camera fields, camera lifecycle, video recording, configurable asset/result paths, restart/resume, and policy communication. The audited changes are included in `examples/simBenchmarks/RoboDojo/eval_files_ego4wam/patches/`. The IsaacLab app pins `isaacsim.asset.importer.urdf` to `2.4.30` rather than `2.4.31`. Preparation has been tested on clean source checkouts, including an idempotent second check. That does not establish a fresh installation of the complete NVIDIA SDK.

The public `isaacsim-asset==5.1.0.0` wheel's extension metadata confirms that
it includes URDF importer `2.4.30`, matching the historical runtime. This pin
does not require a privately supplied extension.

### Source preparation

From the Ego4WAM/StarVLA code checkout, define your own installation locations:

```bash
export EGO4WAM_CODE="$PWD"
export ROBODOJO_ROOT=/path/to/RoboDojo
export EVAL_CODE="$EGO4WAM_CODE/examples/simBenchmarks/RoboDojo/eval_files_ego4wam"
git clone https://github.com/robodojo-benchmark/RoboDojo.git "$ROBODOJO_ROOT"
git -C "$ROBODOJO_ROOT" checkout 9b4cc885e8f530ed3ab14a30a312ae70242771c4
git -C "$ROBODOJO_ROOT" submodule update --init XPolicyLab third_party/IsaacLab third_party/curobo
git -C "$ROBODOJO_ROOT/XPolicyLab" checkout fe71eb54675cef495fea817a637386a4f4529153
git -C "$ROBODOJO_ROOT/third_party/IsaacLab" checkout afca7b09d60d8beb9c1cb28b43066499940b969b
git -C "$ROBODOJO_ROOT/third_party/curobo" checkout d17b54ce32cba095c0b000c4c58777075d11de0e
python "$EVAL_CODE/prepare_runtime.py" --robodojo-root "$ROBODOJO_ROOT" --apply
python "$EVAL_CODE/prepare_runtime.py" --robodojo-root "$ROBODOJO_ROOT"
```

Use a new checkout for these commands; do not check out over unrelated changes.
The independent Ego4WAM adapter is bundled in the code checkout; do not copy it
over XPolicyLab's existing starVLA policy.

### Assets

Use the official [RoboDojo dataset assets](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/tree/c037c1b3183a030724e694d69a08cb62369ed285/Assets) at revision `c037c1b3183a030724e694d69a08cb62369ed285` (2026-07-26). The historical `build_tower` seed-0 layouts 0–49 have been compared with the public Git-LFS SHA256 pointers: **50/50 match**. `Robots/x5/ARX.usd`, `X5A.urdf`, and `robot_config.yml` also match. The entire asset tree has not yet been hash-verified; do not assume that newer `main` assets are equivalent for all tasks.

Download only assets, not training data or other policies' checkpoints:

```bash
export ROBODOJO_ASSET_ROOT=/path/to/download/RoboDojo-assets/Assets
hf download RoboDojo-Benchmark/RoboDojo --repo-type dataset \
  --revision c037c1b3183a030724e694d69a08cb62369ed285 \
  --include 'Assets/**' --local-dir "$(dirname "$ROBODOJO_ASSET_ROOT")"
```

Expected folders include `Robots/`, `Object/`, `Material/`, and `Eval_Layout/`.
Run as a user with access to the downloaded assets. A successful CUDA or SDK
startup does not prove that asset files are readable.

The public assets contain `curobo_tmp.yml` templates, not the runtime-generated
`curobo.yml`. Generate them for **your own writable asset download**, before
starting evaluation; do not reuse another machine's generated absolute paths:

```bash
test -e "$ROBODOJO_ROOT/Assets" || ln -s "$ROBODOJO_ASSET_ROOT" "$ROBODOJO_ROOT/Assets"
cd "$ROBODOJO_ROOT"
python utils/update_embodiment_config_path.py
cd "$EGO4WAM_CODE"
```

This upstream utility writes generated robot planning configurations under
`Assets/Robots/`. It must not be pointed at a shared read-only asset tree.

## Separate Python environments

The policy and simulator communicate over WebSockets. Do not collapse their environments or install the simulator's Torch version over the model's Torch version.

The exercised machine uses Ubuntu 22.04 / GLIBC 2.35, NVIDIA driver 570.211.01,
RTX 5090 GPUs and FFmpeg 4.4.2 with `libx264`. These are observed test-machine
versions, not a guarantee that every other driver/GPU combination is equivalent.

### Model service: tested versions

| Package | Version |
| --- | --- |
| Python | 3.10.12 |
| torch | 2.8.0+cu128 |
| torchvision | 0.23.0+cu128 |
| transformers | 4.57.0 |
| diffusers | 0.39.0 |
| accelerate | 1.5.2 |
| numpy | 1.26.4 |
| safetensors | 0.8.0 |
| pytorch-kinematics | 0.7.5 |
| websockets | 16.1.1 |
| msgpack | 1.2.1 |
| omegaconf | 2.3.1 |
| opencv-python-headless | 4.11.0.86 |
| Pillow | 11.3.0 |
| einops | 0.8.1 |
| huggingface-hub | 0.34.4 |
| PyYAML | 6.0.3 |
| rich | 15.0.0 |
| tokenizers | 0.22.1 |
| typing_extensions | 4.15.0 |
| qwen-vl-utils | 0.0.14 |

The direct model versions have now been installed from public PyPI / the
official CUDA 12.8 PyTorch index into a clean Python 3.10 environment. `pip check`,
strict loading through the existing loader and two-GPU inference passed.
`constraints-model.txt` records that tested dependency set, including transitive
packages; it is not a freeze of the old development image. The synthetic raw
observation smoke returned finite `[1,32,14]` actions using the 20-step sampler;
it does not establish a benchmark success rate. The environment had neither
`timm` nor LeRobot installed.

Install model dependencies in an independent Python 3.10 environment:

```bash
python3.10 -m venv /path/to/model-env
export EGO4WAM_PYTHON=/path/to/model-env/bin/python
"$EGO4WAM_PYTHON" -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128 \
  -c "$EVAL_CODE/constraints-model.txt"
"$EGO4WAM_PYTHON" -m pip install -r "$EVAL_CODE/requirements-model.txt" \
  -c "$EVAL_CODE/constraints-model.txt"
"$EGO4WAM_PYTHON" -m pip check
export CKPT=/path/to/models/Ego4WAM_VideoPretrained_RoboDojoJoint
/path/to/model-env/bin/hf download HorizonRobotics/Ego4WAM \
  --revision 262306ba29e9a4581ac5f714b90711498917444f --local-dir "$CKPT"
```

Do not install the repository's broad training requirements into this inference
environment. `transformers==4.57.0` is the observed historical version and is
explicitly pinned even though PyPI marks that release as yanked; do not silently
upgrade it when reproducing this baseline.

### Simulator: tested versions

| Package | Version |
| --- | --- |
| Python | 3.11.15 |
| torch | 2.7.0+cu128 |
| torchvision | 0.22.0+cu128 |
| isaacsim | 5.1.0.0 (installation spec: 5.1.0) |
| isaaclab | 0.54.3 (install from the pinned fork) |
| nvidia-curobo | 0.0.0 (install from the pinned fork) |
| warp-lang | 1.11.0 |
| numpy | 1.26.0 |
| opencv-python-headless | 4.11.0.86 |
| PyYAML | 6.0.2 |
| websockets | 12.0 |
| msgpack | 1.2.1 |
| msgpack-numpy | 0.4.8 |
| omegaconf | 2.3.1 |
| h5py | 3.16.0 |
| trimesh | 4.5.1 |
| scipy | 1.15.3 |

The editable fork versions do not uniquely identify their code; the Git revisions above are required.

The historical task proxy was a **third** environment: Python 3.11.15,
`websockets==15.0.1`, `pydantic==2.11.10`, `msgpack==1.2.1`,
`numpy==1.26.0`, `PyYAML==6.0.2`, `opencv-python-headless==4.11.0.86`,
`h5py==3.16.0`, and `msgpack-numpy==0.4.8`. Set `ROBODOJO_PROXY_PYTHON`
to that environment, not the simulator's WebSockets 12 environment.

```bash
python3.11 -m venv /path/to/proxy-env
export ROBODOJO_PROXY_PYTHON=/path/to/proxy-env/bin/python
"$ROBODOJO_PROXY_PYTHON" -m pip install -r "$EVAL_CODE/requirements-proxy.txt"
"$ROBODOJO_PROXY_PYTHON" -m pip check
```

Isaac Sim 5.1.0 has a public Python installation path using Python 3.11 and,
on Linux, GLIBC 2.35 or newer. The source-only simulator recipe under validation
creates a separate environment and selects the historical CUDA 12.8 Torch stack
before installing the SDK; the SDK installation step uses NVIDIA's public installer.
`constraints-simulator.txt` records the clean, conflict-free Python dependency set:

```bash
python3.11 -m venv /path/to/simulator-env
export ROBODOJO_PYTHON=/path/to/simulator-env/bin/python
"$ROBODOJO_PYTHON" -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128 \
  -c "$EVAL_CODE/constraints-simulator.txt"
"$ROBODOJO_PYTHON" -m pip install 'isaacsim[all,extscache]==5.1.0' \
  --extra-index-url https://pypi.nvidia.com \
  -c "$EVAL_CODE/constraints-simulator.txt"
"$ROBODOJO_PYTHON" -m pip install -r "$EVAL_CODE/requirements-simulator.txt" \
  -c "$EVAL_CODE/constraints-simulator.txt"
"$ROBODOJO_PYTHON" -m pip check
```

This installs Isaac Sim, not the complete Ego4WAM evaluation environment. Follow NVIDIA's [Isaac Sim 5.1.0 Python installation instructions](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html) and license requirements. Install the pinned benchmark/forks and the audited historical patches separately. System NVIDIA driver / Vulkan setup and FFmpeg with `libx264` are also needed. `requirements-simulator.txt` records direct runtime dependencies; `constraints-simulator.txt` also pins the tested transitive Python packages. It is not a lock for system libraries or every SDK extension outside the exercised application.

**Simulator installation caveat:** the historical development runtime has
dependency conflicts involving optional SDK/fork tooling (including WebSockets,
Starlette, ONNX and packaging). Running RoboDojo's stock `scripts/install.sh`
also updates submodules to remote branches, so it must not be used unchanged
for this baseline. The clean Python 3.11 SDK/runtime installation now passes
`pip check`, and the SDK application starts with the pinned forks. Its Torch and
Isaac Sim modules resolve to the new environment, with no inherited SDK site
packages. CUDA runtime/NVRTC headers resolve to the public pip packages. The
source launch exposed missing `flatdict`/`tqdm` requirements; their historical
versions have been added. A single parallel batch of ten `build_tower`
episodes completed with status 0 using this clean public-source model,
simulator and proxy setup: 9/10 successes, score 91.0. All ten episode/layout
records and 30 three-camera H.264 videos passed validation. This is a functional
smoke, not the 50-episode historical success-rate comparison.
The completed 50-episode result used the historical installed SDK, not this new
environment; keep that result's provenance separate.

The evaluation launchers import the checked IsaacLab/cuRobo source directories
directly. This deliberately avoids installing the forks' unrelated RL, GUI,
retargeting and test extras, whose metadata introduces the historical
development-environment conflicts. It does not replace the SDK's own
dependencies. Do not substitute a PyPI CUDA 12.6 Torch wheel for the explicit
CUDA 12.8 stack above when reproducing the exercised RTX 5090 environment.

## Observation and sampling invariants

Keep the historical benchmark renderer and policy behavior. Do not add the later RGB one-frame cache, synchronized-capture backport, task-prompt overrides, task/layout/chunk-keyed noise, or the newer retrained checkpoint.

The proxy must forward the historical raw three-camera observation contract, 14-D joint state, live camera transforms, and each replicated environment's `env_origin`. The server performs mosaic construction, FK-derived 32-D state, normalization and action inverse-normalization. No fixed external-camera offset should replace live environment geometry.

The actual pre-paper local adapter forwards all active observations of a task in one request. It also updates camera poses at the penultimate action and uses those historical poses for the next observation; this existing geometry convention is retained, not replaced with the later RGB-frame compensation. The model service retains asynchronous microbatching across requests and global RNG behavior. Environment seed 0 fixes layout selection, but does not identify the unseeded model RNG stream or concurrent request arrival order. The original full evaluation ran multiple tasks against one service. A `build_tower`-only run can therefore produce different trajectories and success rate even with the same weights and environment seed; record the actual result rather than promising exact 34/50 outcomes.

The release default is capacity **10** with a **20-ms** collection window,
matching the single-task script. The paper run manifest did not record server
capacity; its corresponding historical full-suite command specified **32**.
The default of 10 is an explicit release choice, not proof of the original
full-suite server arguments. The current validation has only one task and at
most ten active environments per request, so it does not test a difference in
cross-task microbatch capacity. Set `MAX_BATCH_SIZE=32` explicitly only when
intentionally testing that historical full-suite command.

## Migration acceptance checks

- Install the released evaluation dependencies from public sources and pinned source revisions, without relying on an unpublished image.
- Strict-load the five-item HF release through the existing Ego4WAM loader; reject missing model assets.
- Test request batching/splitting, per-environment origins, action ordering, task failure handling and restart/resume without changing observation timing or policy sampling.
- Run `build_tower`, seed 0, 50 episodes, ten environments, 20 sampler steps and 32-step execution using the migrated entrypoints.
- Save per-episode outcomes, task JSON, logs, run metadata, aggregate metrics and all three camera videos. Report coverage and actual success count before comparing with the historical 68% result.

The migrated seed-0 `build_tower` check completed on 2026-10-12:

| Run | Successes | Success rate | Score |
| --- | --- | --- | --- |
| Historical paper local run | 34/50 | 68% | 73.4 |
| Migrated local run | 36/50 | 72% | 77.0 |

The difference is +4 percentage points (+2 successes) and +3.6 score. The client
exited normally. All 50 episode details, unique layout IDs 0–49 and 150 nonempty
camera videos were checked; FFprobe verified positive durations/frame counts
and valid H.264 video streams for all 150 files. Representative video decoding
and visual inspection also passed. The run used the freshly prepared pinned
sources and historical installed simulator SDK. It verifies this migrated task
path, not a full-suite 14.35% reproduction or fresh simulator SDK installation.

A separate clean-public-environment check also completed on 2026-10-12. Per the
chosen validation scope, it ran just one ten-environment batch: **9/10
successes**, score **91.0**, with complete episode/layout coverage and 30 valid
three-camera videos. The simulator, proxy and model Python environments were
newly installed from public packages; the pinned source forks and released HF
weights were used unchanged. This verifies end-to-end functionality of that
Python setup, not a fresh operating-system installation or a new 50-episode
success-rate comparison.

## Local launch commands

After source/environment/asset preparation, launch the server from the code
checkout (leave it running):

```bash
cd "$EGO4WAM_CODE"
export SERVER_CUDA_VISIBLE_DEVICES=0,1 PORT=6789
bash "$EVAL_CODE/local/local_server.sh"
```

In a second terminal, use the same `CKPT`, `EGO4WAM_CODE`, `EVAL_CODE` and source
paths, then run:

```bash
export ROBODOJO_PYTHON=/path/to/simulator-env/bin/python
export ROBODOJO_PROXY_PYTHON=/path/to/proxy-env/bin/python
export HOST=127.0.0.1 BASE_PORT=6789
export TASK_NAMES=build_tower SEED=0 NUM_EPISODES=50 NUM_ENVS=10
export EXECUTE_HORIZON=32 SAVE_VIDEO=1 CLIENT_CUDA_VISIBLE_DEVICES=2
export CLIENT_SHARED_RUN_ID=joint_build_tower_seed0
export OUTPUT_ROOT="$EGO4WAM_CODE/results/ego4wam_robodojo/$CLIENT_SHARED_RUN_ID"
bash "$EVAL_CODE/local/local_client.sh"
```

The default model server requires two GPUs; the simulator uses a third GPU.
For a functional smoke check rather than the 50-episode historical comparison,
set `NUM_EPISODES=10` with `NUM_ENVS=10` and use a distinct run ID. This runs one
parallel batch; keep the same observation, sampling and action settings. Its
success rate is not directly comparable with the historical 50-episode result.
This has been exercised on RTX 5090s, not established as a minimum-memory
configuration for every GPU. Never reuse an output/run ID for a different
experiment; reuse it only to deliberately resume the same interrupted run.

For the full 42-task historical suite, unset `TASK_NAMES`, set
`TASK_SET_NAME=FULL_TASKS` and `NUM_EPISODES=native`, and list available simulator
GPUs in `CLIENT_CUDA_VISIBLE_DEVICES`. Keep the model GPUs separate. The native
task counts yield 54 configurations / 2,100 episodes.
The migrated inventory and native-count reader have been checked against the
pinned, patched source tree: 54 runnable configurations, 42 canonical tasks,
2,100 episodes and all five capability dimensions. This verifies selection and
aggregation inputs, not full-suite rollout performance.

Outputs are stored under `OUTPUT_ROOT`: `run_manifest.json`, `results/`,
`task_status/`, `logs/`, `aggregate.json`, `aggregate.txt`, and
`video/<task>/episode_<7 digits>/{head,left_wrist,right_wrist}.mp4`.
Three nonempty videos and complete episode details are required for each task.
Incomplete results return nonzero and must not be reported as a complete
benchmark. Stop the service manually when done; shared-server shutdown is not
enabled by default.

`local_client.sh` automatically selects the five pinned IsaacLab source
extensions and the pinned cuRobo source tree on `PYTHONPATH`. This avoids
silently substituting an independently installed fork. For manual simulator
import diagnostics, source `common/source_paths.sh` after setting
`ROBODOJO_ROOT`; this changes import paths, not physics or capture timing.

## Source migration and validation

The local entrypoints are provided in
`examples/simBenchmarks/RoboDojo/eval_files_ego4wam/` in the StarVLA-based code tree:

- `prepare_runtime.py` checks all four source revisions and applies audited patches only with `--apply`; repeated checks are supported without resetting worktrees.
- `local/local_server.sh` uses the existing Ego4WAM HF loader and its dedicated historical microbatch transport.
- `local/local_client.sh` runs the local scheduler, independent Ego4WAM XPolicyLab adapter, restart wrapper, video validation and historical metric aggregation.
- `requirements-model.txt`, `requirements-simulator.txt` and `requirements-proxy.txt` record the direct runtime dependencies; `constraints-model.txt` and `constraints-simulator.txt` pin clean public-source environments. A ten-episode simulator end-to-end task smoke passed.

The proxy dependency set has now been downloaded from public PyPI and installed
into a new Python 3.11 environment with no inherited site packages. Both the
independent adapter/XPolicyLab server imports and `pip check` passed. A metadata
handshake with the running model service also passed. The public model
environment passed `pip check`, canonical strict loading and two-GPU inference.
The simulator SDK environment passes `pip check`, module-origin checks and
application startup. Its ten-episode end-to-end rollout completed normally;
all per-episode results and three-camera videos passed validation.

The training dataset export is lazy-imported so online geometry/video processing does not require the unrelated LeRobot training dependencies. Selecting Ego4WAM imports only its framework registration; other frameworks retain StarVLA's existing discovery behavior. This changes dependency loading, not model computation or data processing.

Initial checks: 16 CPU interface/source-contract tests, two framework-discovery tests and 22 standard server/configuration regression tests passed. Patches applied successfully to fresh source checkouts, repeated preparation checks passed, and the published Joint release strict-loaded on two RTX 5090 GPUs. The migrated seed-0 / 50-episode `build_tower` evaluation completed with 36 successes and successful process restart/resume between batches. Complete episode/layout and three-camera video coverage passed. Separately, the new public simulator environment completed its single-batch ten-episode functional smoke with 9 successes and 30 validated camera videos.

## RoboDojo Post-training

For the four Wan/video-pretrained × Base/Joint posttraining recipes, environment setup, data download and checkpoint format, see the [root README training guide](../README.md#robodojo-posttraining) and [train_files_ego4wam](../examples/simBenchmarks/RoboDojo/train_files_ego4wam/). Training is separate from the historical local evaluation setup described above. Do not substitute newer synchronized-data training or observation compensation when evaluating the released paper checkpoint.
