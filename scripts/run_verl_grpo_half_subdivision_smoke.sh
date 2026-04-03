#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export DATA_DIR="${DATA_DIR:-$REPO_ROOT/training/verl/data-smoke}"
export TRAIN_LIMIT="${TRAIN_LIMIT:-16}"
export VAL_LIMIT="${VAL_LIMIT:-4}"
export ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
export ROLLOUT_N="${ROLLOUT_N:-2}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"
export TEST_FREQ="${TEST_FREQ:-10}"

"$REPO_ROOT/scripts/run_verl_grpo_half_subdivision.sh"
