#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRIME_RL_DIR="${PRIME_RL_DIR:-/workspace/prime-rl}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/training/prime_rl/rl.toml}"

if [[ -f "$HOME/.bash_profile" ]]; then
  # Load persisted tokens such as WANDB_API_KEY and HF_TOKEN on fresh shells.
  # shellcheck disable=SC1090
  source "$HOME/.bash_profile"
fi

if [[ ! -d "$PRIME_RL_DIR" ]]; then
  echo "Missing PRIME_RL_DIR=$PRIME_RL_DIR" >&2
  echo "Run $REPO_ROOT/scripts/bootstrap_prime_rl_workspace.sh first." >&2
  exit 1
fi

UV_BIN="${UV_BIN:-}"
if [[ -z "$UV_BIN" ]]; then
  if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
  elif [[ -x "$PRIME_RL_DIR/.venv/bin/uv" ]]; then
    UV_BIN="$PRIME_RL_DIR/.venv/bin/uv"
  elif [[ -x "$HOME/.local/bin/uv" ]]; then
    UV_BIN="$HOME/.local/bin/uv"
  else
    echo "Could not find uv. Expected it on PATH, at $PRIME_RL_DIR/.venv/bin/uv, or at $HOME/.local/bin/uv" >&2
    exit 1
  fi
fi

cd "$PRIME_RL_DIR"
"$UV_BIN" pip install --python "$PRIME_RL_DIR/.venv/bin/python" -e "$REPO_ROOT/external/VisGeomBench"
"$UV_BIN" pip install --python "$PRIME_RL_DIR/.venv/bin/python" -e "$REPO_ROOT/environments/half_subdivision_shaped"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_USE_DEEP_GEMM_E8M0="${VLLM_USE_DEEP_GEMM_E8M0:-0}"
export VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-skip}"

resolved_config_path="$CONFIG_PATH"
if [[ -n "${CURRICULUM_STAGE:-}" || -n "${CURRICULUM_MAX_STAGE:-}" ]]; then
  tmp_config="$(mktemp "${TMPDIR:-/tmp}/half-subdivision-prime-rl.XXXXXX.toml")"
  trap 'rm -f "$tmp_config"' EXIT
  cp "$CONFIG_PATH" "$tmp_config"

  curriculum_args='args = { task_name = "half_subdivision"'
  if [[ -n "${CURRICULUM_STAGE:-}" ]]; then
    curriculum_args+=", curriculum_stage = \"${CURRICULUM_STAGE}\""
  fi
  if [[ -n "${CURRICULUM_MAX_STAGE:-}" ]]; then
    curriculum_args+=", curriculum_max_stage = \"${CURRICULUM_MAX_STAGE}\""
  fi
  curriculum_args+=' }'

  CURRICULUM_ARGS="$curriculum_args" perl -0pi -e \
    's/args = \{ task_name = "half_subdivision" \}/$ENV{CURRICULUM_ARGS}/' \
    "$tmp_config"
  resolved_config_path="$tmp_config"
fi

"$UV_BIN" run --no-sync rl @ "$resolved_config_path" "$@"
