#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to a local Hugging Face model path}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/training/verl/runs/$(date +%Y%m%d-%H%M%S)-half-subdivision-grpo}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
VERL_CONFIG_PATH="${VERL_CONFIG_PATH:-$(
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import verl

print(Path(verl.__file__).resolve().parent / "trainer" / "config")
PY
)}"
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
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-16384}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
TOTAL_TOKEN_BUDGET="${TOTAL_TOKEN_BUDGET:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
TEST_FREQ="${TEST_FREQ:-50}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/training/verl/data}"
TRAIN_LIMIT="${TRAIN_LIMIT:-}"
VAL_LIMIT="${VAL_LIMIT:-}"
XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$REPO_ROOT/.runtime-config}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_ROOT/.runtime-cache}"
MPLCONFIGDIR="${MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}"

cd "$REPO_ROOT"

prep_args=(
  training/verl/prepare_half_subdivision_data.py
  --output-dir "$DATA_DIR"
  --model-path "$MODEL_PATH"
  --max-prompt-length "$MAX_PROMPT_LENGTH"
  --max-response-length "$MAX_RESPONSE_LENGTH"
  --total-token-budget "$TOTAL_TOKEN_BUDGET"
)
if [[ -n "$TRAIN_LIMIT" ]]; then
  prep_args+=(--train-limit "$TRAIN_LIMIT")
fi
if [[ -n "$VAL_LIMIT" ]]; then
  prep_args+=(--val-limit "$VAL_LIMIT")
fi

"$PYTHON_BIN" "${prep_args[@]}"

export HYDRA_FULL_ERROR=1
export REPO_ROOT
export MODEL_PATH
export RUN_DIR
export XDG_CONFIG_HOME
export XDG_CACHE_HOME
export MPLCONFIGDIR

mkdir -p "$RUN_DIR" "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

"$PYTHON_BIN" -m verl.trainer.main_ppo \
  --config-name=ppo_trainer \
  --config-path="$VERL_CONFIG_PATH" \
  algorithm.adv_estimator=grpo \
  trainer.project_name=half_subdivision_grpo \
  trainer.experiment_name=qwen3_8b_half_subdivision \
  trainer.test_freq="$TEST_FREQ" \
  trainer.val_before_train=true \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/val.parquet" \
  data.prompt_key=prompt \
  data.truncation=error \
  data.filter_overlong_prompts=true \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.val_batch_size="$VAL_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP_SIZE" \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.prompt_length="$MAX_PROMPT_LENGTH" \
  actor_rollout_ref.rollout.response_length="$MAX_RESPONSE_LENGTH" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.use_dynamic_bsz=true \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$TOTAL_TOKEN_BUDGET" \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.ref.fsdp_config.param_offload=false \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$TOTAL_TOKEN_BUDGET" \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.max_num_batched_tokens="$TOTAL_TOKEN_BUDGET" \
  custom_reward_function.path="$REPO_ROOT/training/verl/half_subdivision_reward.py" \
  custom_reward_function.name=compute_score \
  trainer.n_gpus_per_node="$N_GPUS"
