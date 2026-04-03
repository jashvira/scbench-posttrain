#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/training/verl/grpo_half_subdivision.yaml}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to a local Hugging Face model path}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/training/verl/runs/$(date +%Y%m%d-%H%M%S)-half-subdivision-grpo}"
if [[ -z "${N_GPUS:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    N_GPUS="$(nvidia-smi -L | wc -l | tr -d ' ')"
  else
    N_GPUS=1
  fi
fi
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
ROLLOUT_N="${ROLLOUT_N:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"

cd "$REPO_ROOT"

uv run python training/verl/prepare_half_subdivision_data.py

export HYDRA_FULL_ERROR=1
export REPO_ROOT
export MODEL_PATH
export RUN_DIR

uv run python -m verl.trainer.main_ppo \
  --config-name="$(basename "$CONFIG_PATH" .yaml)" \
  --config-path="$(dirname "$CONFIG_PATH")" \
  data.train_files="$REPO_ROOT/training/verl/data/train.parquet" \
  data.val_files="$REPO_ROOT/training/verl/data/val.parquet" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.val_batch_size="$VAL_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP_SIZE" \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
  trainer.n_gpus_per_node="$N_GPUS"
