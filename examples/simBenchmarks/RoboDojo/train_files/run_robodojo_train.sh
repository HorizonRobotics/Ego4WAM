#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  config_yaml=<recipe.yaml> \
  robodojo_data_root=/path/to/datasets \
  bash examples/simBenchmarks/RoboDojo/train_files/run_robodojo_train.sh \
    [train_starvla.py overrides...]

Expected dataset:
  <robodojo_data_root>/RoboDojo_lerobot_v21_video

Useful variables:
  NUM_PROCESSES       Total Accelerate process count (auto-detected by default)
  NUM_MACHINES        Number of training machines (default: 1)
  MACHINE_RANK        Rank of this machine (default: 0)
  MAIN_PROCESS_IP     Required for multi-machine training
  MAIN_PROCESS_PORT   Main process port (default: 29500)
  DRY_RUN=1           Print the resolved command without starting training
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

config_yaml="${config_yaml:-examples/simBenchmarks/RoboDojo/train_files/starvla_robodojo_v21_qwenoft_h50_q99.yaml}"
robodojo_data_root="${robodojo_data_root:-playground/Datasets}"
dataset_dir="${robodojo_data_root}/RoboDojo_lerobot_v21_video"
num_machines="${NUM_MACHINES:-1}"
machine_rank="${MACHINE_RANK:-0}"
main_process_ip="${MAIN_PROCESS_IP:-}"
main_process_port="${MAIN_PROCESS_PORT:-29500}"

if [[ ! -f "${config_yaml}" ]]; then
    echo "[RoboDojo][ERROR] Training config not found: ${config_yaml}" >&2
    exit 1
fi

if [[ ! -f "${dataset_dir}/meta/info.json" || ! -f "${dataset_dir}/meta/modality.json" ]]; then
    echo "[RoboDojo][ERROR] Official LeRobot v2.1 dataset not found: ${dataset_dir}" >&2
    echo "[RoboDojo][ERROR] Download it with RoboDojo's scripts/RoboDojo/download_data.sh huggingface lerobot_v2.1." >&2
    exit 1
fi

if [[ ! "${num_machines}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[RoboDojo][ERROR] NUM_MACHINES must be a positive integer: ${num_machines}" >&2
    exit 2
fi

if [[ -n "${NUM_PROCESSES:-}" ]]; then
    num_processes="${NUM_PROCESSES}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "-1" ]]; then
    IFS=',' read -r -a visible_devices <<< "${CUDA_VISIBLE_DEVICES}"
    local_gpu_count="${#visible_devices[@]}"
    num_processes="$((local_gpu_count * num_machines))"
elif command -v nvidia-smi >/dev/null 2>&1; then
    local_gpu_count="$(nvidia-smi -L | wc -l)"
    num_processes="$((local_gpu_count * num_machines))"
else
    echo "[RoboDojo][ERROR] Cannot detect GPUs; set NUM_PROCESSES explicitly." >&2
    exit 1
fi

if [[ ! "${num_processes}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[RoboDojo][ERROR] NUM_PROCESSES must be a positive integer: ${num_processes}" >&2
    exit 2
fi

if (( num_machines > 1 )) && [[ -z "${main_process_ip}" ]]; then
    echo "[RoboDojo][ERROR] MAIN_PROCESS_IP is required for multi-machine training." >&2
    exit 2
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/playground/Cache/huggingface}"

cmd=(
    accelerate launch
    --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml
    --num_processes "${num_processes}"
)

if (( num_machines > 1 )); then
    cmd+=(
        --num_machines "${num_machines}"
        --machine_rank "${machine_rank}"
        --main_process_ip "${main_process_ip}"
        --main_process_port "${main_process_port}"
    )
fi

cmd+=(
    starVLA/training/train_starvla.py
    --config_yaml "${config_yaml}"
    --datasets.vla_data.data_root_dir "${robodojo_data_root}"
    "$@"
)

echo "[RoboDojo] config: ${config_yaml}"
echo "[RoboDojo] dataset: ${dataset_dir}"
echo "[RoboDojo] processes: ${num_processes}; machines: ${num_machines}; rank: ${machine_rank}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '[RoboDojo] DRY_RUN command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    exit 0
fi

exec "${cmd[@]}"
