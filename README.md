<div align="center">
<h1>Ego4WAM: What Matters When Scaling Egocentric Human Data for Robot Learning?</h1>

<a href="https://sunzhihao18.github.io/Ego4WAM/"><img src="https://img.shields.io/badge/Project-Page-blue?logo=github" alt="Project Page"></a>
<a href="https://arxiv.org/abs/2609.40341"><img src="https://img.shields.io/badge/PDF-arXiv-red?logo=arxiv" alt="Paper"></a>
<a href="https://huggingface.co/HorizonRobotics/Ego4WAM"><img src="https://img.shields.io/badge/Model-HuggingFace-yellow?logo=huggingface" alt="Model"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue" alt="License"></a>
</div>

## 📖 Abstract

Egocentric human data provides a scalable source of experience for robot learning, but varies substantially in human-robot alignment, behavioral coverage, and available supervision. Existing work shows favorable scaling with increasing human data, but it remains unclear which data properties drive downstream robot gains and how to use such data throughout the training pipeline. We present a systematic study of egocentric human data with different alignment and supervision under a unified world-action model framework. With the model backbone fixed, we disentangle the effects of human-robot alignment, data duration and task diversity, action supervision, and data usage strategies. We find that aligned human demonstrations substantially improve out-of-distribution generalization and reduce target-task robot data requirements; data duration and task diversity affect downstream capabilities differently; and video-only supervision remains effective without action labels, providing a strong foundation for subsequent video-action training. We validate these findings through closed-loop policy evaluation on both real robots and RoboDojo. Rather than treating data duration as the sole scaling axis, Ego4WAM shows how alignment, task diversity, available supervision, and usage strategy jointly shape the value of egocentric human data for robot learning.

## ⭐ Key Findings

1. **Robot-aligned human data enables OOD adaptation and reduces robot data.**

   * **OOD Adaptation.** Aligned human demonstrations expose the policy to deployment conditions absent from robot training. Without any object- or scene-OOD robot demonstrations, object-OOD success increases from 10% to 60%, while scene-OOD success increases from 0% to 20% with a substantially larger gain in task progress.

   * **Reducing Robot-Data Dependence.** With 100 robot demonstrations, aligned human data recovers 80% ID, 60% object-OOD, and 80% scene-OOD success. Human data alone is insufficient when robot supervision falls to 20 demonstrations, but broad multi-task mid-training makes the same limited downstream data effective, reaching 80%, 80%, and 60% success, respectively.

2. **Scaling egocentric data does not always lead to better performance.**

   **Data duration** and **task diversity** are distinct scaling axes, and total hours alone do not characterize useful scale. Denser coverage of recurring tasks provides limited additional gains once common behaviors are sufficiently represented, while expanding aggressively into increasingly sparse long-tail tasks can introduce negative transfer and hurt precision-sensitive performance.

3. **Egocentric data provides a strong dynamics prior for world-action learning.**

   * **Learning Dynamics Priors without Action Supervision.** Video-only pre-training improves the average RoboDojo score from 6.39 to 14.13 and success rate from 3.15% to 9.45% without human action labels, extending usable data beyond the action-valid subset. The learned dynamics prior further supports video-action training, while joint future conditioning exploits it most effectively, reaching a 20.29 average score and 14.35% success rate.

   * **Competitive Performance with Limited Robot Data.** With only **approximately 90 hours of real-robot data** and **15K hours of egocentric data**, Ego4WAM achieves a competitive 20.29 average score and 14.35% success rate on RoboDojo. Because the model is not explicitly designed for long-term memory or VLM-based open-instruction following, Memory and Open remain its weaker categories; nevertheless, Ego4WAM achieves the best Precision score and success rate and the best Long-Horizon score among the reported methods.


## 🚀 Getting Started

This release provides the Ego4WAM Base and Joint models, four RoboDojo Post-training recipes, and local RoboDojo evaluation. Human-video pretraining pipelines and other internal experiments are not included. The Python package/import namespace remains `starVLA`.

Run commands from this checkout's root and use absolute paths for models, datasets and environments:

```bash
export EGO4WAM_CODE="$PWD"
export EVAL_CODE="$EGO4WAM_CODE/examples/simBenchmarks/RoboDojo/eval_files_ego4wam"
export TRAIN_CODE="$EGO4WAM_CODE/examples/simBenchmarks/RoboDojo/train_files_ego4wam"
```

### Model environment

Use Python 3.10. The exercised model runtime uses PyTorch 2.8.0 and torchvision 0.23.0 with CUDA 12.8. Keep the model, simulator and policy-proxy environments separate.

```bash
python3.10 -m venv /absolute/path/to/ego4wam-model-env
export EGO4WAM_PYTHON=/absolute/path/to/ego4wam-model-env/bin/python

"$EGO4WAM_PYTHON" -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128 \
  -c "$EVAL_CODE/constraints-model.txt"
"$EGO4WAM_PYTHON" -m pip install -r "$EVAL_CODE/requirements-model.txt" \
  -c "$EVAL_CODE/constraints-model.txt"
"$EGO4WAM_PYTHON" -m pip check
```

Do not install the repository's broad upstream `requirements.txt` into this inference environment. The pinned requirements are in [eval_files_ego4wam](examples/simBenchmarks/RoboDojo/eval_files_ego4wam/).

### Model weights

The currently published [HorizonRobotics/Ego4WAM checkpoint](https://huggingface.co/HorizonRobotics/Ego4WAM/tree/262306ba29e9a4581ac5f714b90711498917444f) is the **video-pretrained, RoboDojo-posttrained Joint policy**, named `Ego4WAM_VideoPretrained_RoboDojoJoint` in this release. At this pinned HF revision the files are at the repository root, not in a model subfolder.

```bash
export CKPT=/absolute/path/to/models/Ego4WAM_VideoPretrained_RoboDojoJoint
/absolute/path/to/ego4wam-model-env/bin/hf download HorizonRobotics/Ego4WAM \
  --revision 262306ba29e9a4581ac5f714b90711498917444f --local-dir "$CKPT"
```

Each model bundle has the same layout:

```text
<model-directory>/
├── README.md
├── LICENSE
├── inference_config.yaml
├── model.safetensors
└── model_assets/
```

The weights include the VAE and T5 text encoder; `model_assets/` provides their configurations and the tokenizer. No additional Wan download, training manifest or external normalization-statistics file is required when loading this bundle. The released code includes RoboDojo normalization and kinematics.

The video-only initialization is named `Ego4WAM_VideoPretrained`. It is prepared locally but **not yet published on HF**; do not use the posttrained Joint policy as a substitute for reproducing video-pretrained initialization. Its action/state branches are present but have not undergone robot-action posttraining. The four training recipes below do not imply that four posttrained checkpoints are already released.

<a id="robodojo-posttraining"></a>

## 🔥 RoboDojo Post-training

All Ego4WAM training files are in [train_files_ego4wam](examples/simBenchmarks/RoboDojo/train_files_ego4wam/). The only public model options are:

```yaml
framework:
  name: Ego4WAM
  interaction_mode: base  # or joint
```

Base and Joint differ only in cross-expert attention interaction. Camera layout, 32-D state/action encoding and the model geometry are fixed.

### Training environment and data

Create a separate Python 3.10 training environment. Install the CUDA PyTorch pair first, then the training-only requirements:

```bash
python3.10 -m venv /absolute/path/to/ego4wam-training-env
export TRAIN_PYTHON=/absolute/path/to/ego4wam-training-env/bin/python

"$TRAIN_PYTHON" -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
DS_BUILD_OPS=0 "$TRAIN_PYTHON" -m pip install \
  -r "$TRAIN_CODE/requirements-ego4wam-training.txt"
"$TRAIN_PYTHON" -m pip install -e "$EGO4WAM_CODE" --no-deps
export PATH="/absolute/path/to/ego4wam-training-env/bin:$PATH"
```

Download the official RoboDojo **LeRobot v2.1 video** dataset, not the v3.0 or HDF5 variant. Downloading only the selected data directory avoids other checkpoints and simulator assets:

```bash
export DATA_DOWNLOAD=/absolute/path/to/robodojo-training-data
/absolute/path/to/ego4wam-training-env/bin/hf download RoboDojo-Benchmark/RoboDojo \
  --repo-type dataset --revision 35efbc7dedfdbeeb6e95fb749bd885d73d483e41 \
  --include 'data/RoboDojo_lerobot_v21_video/**' --local-dir "$DATA_DOWNLOAD"
export robodojo_data_root="$DATA_DOWNLOAD/data"
```

Expected layout:

```text
<robodojo_data_root>/
└── RoboDojo_lerobot_v21_video/
    ├── meta/info.json
    ├── data/
    └── videos/
```

The existing LeRobot reader and Ego4WAM adapter construct the three-camera mosaic, normalized 32-D state/action and padding masks. The recipes sample episodes uniformly, then frames within each episode; they do not weight episodes by duration. Task prompts from the paper and the model's fixed prompt template are applied once.

### Four recipes

| Initialization | Attention | Output/model name | Script |
| --- | --- | --- | --- |
| Wan2.2 | Base | `Ego4WAM_Wan_RoboDojoBase` | [train_ego4wam_wan_robodojo_base.sh](examples/simBenchmarks/RoboDojo/train_files_ego4wam/train_ego4wam_wan_robodojo_base.sh) |
| Wan2.2 | Joint | `Ego4WAM_Wan_RoboDojoJoint` | [train_ego4wam_wan_robodojo_joint.sh](examples/simBenchmarks/RoboDojo/train_files_ego4wam/train_ego4wam_wan_robodojo_joint.sh) |
| Video pretrained | Base | `Ego4WAM_VideoPretrained_RoboDojoBase` | [train_ego4wam_video_pretrained_robodojo_base.sh](examples/simBenchmarks/RoboDojo/train_files_ego4wam/train_ego4wam_video_pretrained_robodojo_base.sh) |
| Video pretrained | Joint | `Ego4WAM_VideoPretrained_RoboDojoJoint` | [train_ego4wam_video_pretrained_robodojo_joint.sh](examples/simBenchmarks/RoboDojo/train_files_ego4wam/train_ego4wam_video_pretrained_robodojo_joint.sh) |

Wan recipes initialize the video expert, VAE and text encoder from `Wan-AI/Wan2.2-TI2V-5B-Diffusers`; the action expert and state projection start untrained. Either let the loader resolve that fixed HF model, or set `EGO4WAM_WAN_DIR` to a complete local Diffusers directory.

```bash
# Optional local Wan assets:
export EGO4WAM_WAN_DIR=/absolute/path/to/Wan2.2-TI2V-5B-Diffusers
export NUM_PROCESSES=8

# Choose ONE recipe:
bash "$TRAIN_CODE/train_ego4wam_wan_robodojo_base.sh" \
  --run_id Ego4WAM_Wan_RoboDojoBase_run1 --trainer.gradient_accumulation_steps 2
bash "$TRAIN_CODE/train_ego4wam_wan_robodojo_joint.sh" \
  --run_id Ego4WAM_Wan_RoboDojoJoint_run1 --trainer.gradient_accumulation_steps 2
```

Video-pretrained recipes strict-load the same complete Base initialization bundle and retain all learned weights while selecting Base or Joint interaction. Once you have that initialization:

```bash
export EGO4WAM_PRETRAINED_DIR=/absolute/path/to/models/Ego4WAM_VideoPretrained
export NUM_PROCESSES=8

# Choose ONE recipe:
bash "$TRAIN_CODE/train_ego4wam_video_pretrained_robodojo_base.sh" \
  --run_id Ego4WAM_VideoPretrained_RoboDojoBase_run1 --trainer.gradient_accumulation_steps 2
bash "$TRAIN_CODE/train_ego4wam_video_pretrained_robodojo_joint.sh" \
  --run_id Ego4WAM_VideoPretrained_RoboDojoJoint_run1 --trainer.gradient_accumulation_steps 2
```

Add `DRY_RUN=1` before a launch command to inspect it without training. The recipes use 40,000 optimizer steps, 4,000 warmup steps, per-device batch 16, AdamW LR 1e-4 with a cosine decay to 1e-6, and checkpoints every 10,000 steps. The video/action experts and state projection are trainable; VAE/T5 are frozen.

The paper's training setup used 16 devices × batch 16 × accumulation 1 = global batch 256. The eight-device examples above use accumulation 2 to retain that batch size. For two-node training, set `NUM_PROCESSES=16`, `NUM_MACHINES=2`, the node-specific `MACHINE_RANK`, and shared `MAIN_PROCESS_IP/MAIN_PROCESS_PORT`, and use accumulation 1. Full-model training requires suitable accelerator memory; the two-GPU small-model trainer smoke is not evidence that this full model fits on two RTX 5090s.

Runs write `config.full.yaml`, `metrics.jsonl`, `checkpoints/steps_N/{state,model,training_state.json}` and `final_model/` under `<run_root_dir>/<run_id>`. `final_model/` has the same five-item model-bundle format and uses the same strict inference loader. To resume, pass `--trainer.resume_from_checkpoint /absolute/path/to/run/checkpoints/steps_N` with the same data, mode, seed, batch and distributed topology. Use a unique run ID for a new experiment.

## 🤖 RoboDojo Evaluation

Evaluation files are in [eval_files_ego4wam](examples/simBenchmarks/RoboDojo/eval_files_ego4wam/). This evaluates the released Joint policy using the historical RoboDojo version used in the paper, not the current official leaderboard/submission protocol. It does not include later camera-frame compensation, synchronized-data retraining or new prompt overrides.

First follow [docs/RoboDojo.md](docs/RoboDojo.md) to prepare the pinned RoboDojo/XPolicyLab/IsaacLab/cuRobo source checkouts, bundled patches, assets and separate Python 3.11 simulator/proxy environments. Do not run the upstream installation script unchanged: it updates source revisions. Isaac Sim 5.1.0, CUDA/Vulkan support and FFmpeg with `libx264` are required.

| Component | Pinned revision |
| --- | --- |
| RoboDojo | `9b4cc885e8f530ed3ab14a30a312ae70242771c4` |
| XPolicyLab | `fe71eb54675cef495fea817a637386a4f4529153` |
| IsaacLab fork | `afca7b09d60d8beb9c1cb28b43066499940b969b` |
| cuRobo fork | `d17b54ce32cba095c0b000c4c58777075d11de0e` |
| Benchmark assets | `c037c1b3183a030724e694d69a08cb62369ed285` |

### Start the Joint policy server

After downloading the Joint checkpoint and preparing the environments, leave this running in one terminal:

```bash
cd "$EGO4WAM_CODE"
export SERVER_CUDA_VISIBLE_DEVICES=0,1 PORT=6789
bash "$EVAL_CODE/local/local_server.sh"
```

The server uses two GPUs, the canonical HF bundle loader, the fixed 20-step sampler, and asynchronous microbatching with capacity 10 / a 20-ms collection window. Keep the model GPUs separate from the simulator GPUs.

### One task or one-batch smoke

In another terminal, re-export `EGO4WAM_CODE`, `EVAL_CODE` and `CKPT` to the same paths, then:

```bash
export ROBODOJO_ROOT=/absolute/path/to/pinned-and-patched/RoboDojo
export ROBODOJO_ASSET_ROOT=/absolute/path/to/Assets
export ROBODOJO_PYTHON=/absolute/path/to/simulator-env/bin/python
export ROBODOJO_PROXY_PYTHON=/absolute/path/to/proxy-env/bin/python
export HOST=127.0.0.1 BASE_PORT=6789
export TASK_NAMES=build_tower SEED=0 NUM_EPISODES=50 NUM_ENVS=10
export EXECUTE_HORIZON=32 SAVE_VIDEO=1 CLIENT_CUDA_VISIBLE_DEVICES=2
export CLIENT_SHARED_RUN_ID=ego4wam_joint_build_tower_seed0
export OUTPUT_ROOT="$EGO4WAM_CODE/results/ego4wam_robodojo/$CLIENT_SHARED_RUN_ID"
bash "$EVAL_CODE/local/local_client.sh"
```

For a single parallel batch, set `NUM_EPISODES=10` with `NUM_ENVS=10` and choose a different run ID. This is a functional smoke, not a 50-episode performance comparison. Keep the prediction/execution horizon at 32 and the sampler at 20 steps.

### Full benchmark and outputs

Keep the same service, unset `TASK_NAMES`, set `TASK_SET_NAME=FULL_TASKS` and `NUM_EPISODES=native`, and select your simulator GPUs in `CLIENT_CUDA_VISIBLE_DEVICES`. The benchmark suite consists of 54 runnable configurations, 42 canonical tasks and 2,100 episodes.

The client saves `run_manifest.json`, `results/`, `task_status/`, `logs/`, `aggregate.json` and `aggregate.txt` under `OUTPUT_ROOT`. Videos are saved as:

```text
video/<task>/episode_<7 digits>/{head,left_wrist,right_wrist}.mp4
```

The client validates episode coverage and all three videos before marking a task complete. Standard/random configurations are episode-weighted, tasks are averaged within each capability dimension, and the five dimension means are averaged equally. Reaggregate an existing run with:

```bash
cd "$EGO4WAM_CODE"
PYTHONPATH="$EGO4WAM_CODE" "$ROBODOJO_PYTHON" \
  "$EVAL_CODE/common/summarize_results.py" --root "$OUTPUT_ROOT"
```

The paper's full-suite reference is **14.35% dimension-balanced success rate / 20.29 average score**. The migrated seed-0 `build_tower` check obtained 36/50 successes (72%, score 77.0), compared with the paper's reference 34/50 (68%, score 73.4); a separately installed public-environment ten-episode smoke obtained 9/10. These validate the local path, not a full-suite rerun or a guaranteed success rate. Environment seed 0 selects layouts, but does not pin the unseeded model RNG stream or concurrent request ordering. See the [reproduction guide](docs/RoboDojo.md) for provenance and limitations.

Stop the model service manually after evaluation; clients do not shut down a shared server by default.

## 📚 Citation

If you find Ego4WAM useful in your research, please cite:

```bibtex
@article{sun2026ego4wam,
  title   = {Ego4WAM: What Matters When Scaling Egocentric Human Data for Robot Learning?},
  author  = {Sun, Zhihao and Liu, Liu and Wang, Xinjiang and Jiang, Haoyi and Feng, Wei and Zhang, Huiqiang and Jia, Xiaosong and Su, Zhizhong and Wu, Zuxuan},
  journal = {arXiv preprint arXiv:2609.40341},
  year    = {2026}
}
```

## 🙏 Acknowledgements

Ego4WAM builds on several open-source projects and research efforts:

* **[StarVLA](https://github.com/starVLA/starVLA)** provides the modular training, data, and deployment infrastructure used by this repository.
* **[FastWAM](https://arxiv.org/abs/2603.16666)** provides the world-action modeling foundation on which the Ego4WAM architecture is built.
* **[Wan2.2](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)** provides the video world-model backbone.
* **[RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo)** provides the standardized manipulation benchmark and evaluation protocol, with **[XPolicyLab](https://github.com/JinhuiYE/XPolicyLab)** supporting policy integration and evaluation.
* **[LeRobot](https://github.com/huggingface/lerobot)** provides dataset formats and data-loading components used in the training pipeline.

We thank the authors and maintainers of these projects, as well as the creators of the public egocentric datasets used in this study.

## 📄 License

Ego4WAM-authored additions and modifications are released under the [Apache License 2.0](LICENSE). Inherited StarVLA, FastWAM and third-party code retains its original licenses and copyright notices; see [NOTICE](NOTICE) and [LICENSES](LICENSES/README.md). Third-party models, datasets, benchmarks and assets remain subject to their own licenses and terms of use.
