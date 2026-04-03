#!/usr/bin/env bash
set -euo pipefail

if [[ ! -x ".venv/bin/vllm" ]]; then
  echo "vLLM is not installed in .venv. Run ./scripts/bootstrap_prime_vllm_container.sh first." >&2
  exit 1
fi

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
QUANTIZATION="${QUANTIZATION:-fp8}"
ENABLE_REASONING="${ENABLE_REASONING:-1}"

args=(
  -m
  vllm.entrypoints.openai.api_server
  --host "$HOST"
  --port "$PORT"
  --model "$MODEL_ID"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
)

if [[ -n "$QUANTIZATION" ]]; then
  args+=(--quantization "$QUANTIZATION")
fi

# Qwen3 has an explicit reasoning parser in vLLM, so we keep that enabled by
# default for the model family we want to evaluate first.
if [[ "$ENABLE_REASONING" == "1" ]]; then
  args+=(--enable-reasoning --reasoning-parser qwen3)
fi

exec .venv/bin/python "${args[@]}" "$@"
