#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ROBODOJO_PATH=/path/to/RoboDojo \
  bash examples/simBenchmarks/RoboDojo/eval_files/start_eval.sh \
    <oft|groot|pi_v3> <task_name> <seed> <policy_gpu> <sim_gpu> \
    <starvla_env_path> <robodojo_env_path> [episode_count|native]

RoboDojo must contain the companion checkout at:
  $ROBODOJO_PATH/XPolicyLab

The maintained evaluation scripts are provided by:
  https://github.com/JinhuiYE/XPolicyLab/tree/fix/starvla-hf-robodojo-eval

They download and verify the complete Hugging Face run directory, start the
matching policy server, check its runtime contract, and start RoboDojo. Do not
start another StarVLA server manually.
EOF
}

if [[ $# -lt 7 || $# -gt 8 ]]; then
    usage >&2
    exit 2
fi

if [[ -z "${ROBODOJO_PATH:-}" ]]; then
    echo "[RoboDojo][ERROR] Set ROBODOJO_PATH to the RoboDojo checkout." >&2
    usage >&2
    exit 2
fi

if [[ ! -d "${ROBODOJO_PATH}" ]]; then
    echo "[RoboDojo][ERROR] RoboDojo directory not found: ${ROBODOJO_PATH}" >&2
    exit 1
fi

robodojo_root="$(cd "${ROBODOJO_PATH}" && pwd)"
xpolicylab_root="${robodojo_root}/XPolicyLab"
entrypoint="${xpolicylab_root}/policy/starVLA/scripts/eval_hf_robodojo.sh"
manifest="${xpolicylab_root}/policy/starVLA/hf_robodojo_checkpoints.json"

if [[ ! -f "${robodojo_root}/scripts/robodojo.sh" ]]; then
    echo "[RoboDojo][ERROR] Invalid RoboDojo checkout: ${robodojo_root}" >&2
    echo "[RoboDojo][ERROR] Missing ${robodojo_root}/scripts/robodojo.sh" >&2
    exit 1
fi

if [[ ! -f "${entrypoint}" || ! -f "${manifest}" ]]; then
    echo "[RoboDojo][ERROR] The XPolicyLab released-checkpoint launcher is missing." >&2
    echo "[RoboDojo][ERROR] Clone the verified companion branch as:" >&2
    echo "[RoboDojo][ERROR]   ${robodojo_root}/XPolicyLab" >&2
    echo "[RoboDojo][ERROR] See https://github.com/JinhuiYE/XPolicyLab/tree/fix/starvla-hf-robodojo-eval" >&2
    exit 1
fi

xpolicylab_commit="$(git -C "${xpolicylab_root}" rev-parse HEAD 2>/dev/null || echo unknown)"
robodojo_commit="$(git -C "${robodojo_root}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[RoboDojo] XPolicyLab commit: ${xpolicylab_commit}"
echo "[RoboDojo] RoboDojo commit: ${robodojo_commit}"
echo "[RoboDojo] Delegating to ${entrypoint}"

exec bash "${entrypoint}" "$@"
