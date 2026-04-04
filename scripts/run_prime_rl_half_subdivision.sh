#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRIME_RL_DIR="${PRIME_RL_DIR:-/workspace/prime-rl}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/training/prime_rl/rl.toml}"

if [[ ! -d "$PRIME_RL_DIR" ]]; then
  echo "Missing PRIME_RL_DIR=$PRIME_RL_DIR" >&2
  echo "Run $REPO_ROOT/scripts/bootstrap_prime_rl_workspace.sh first." >&2
  exit 1
fi

cd "$PRIME_RL_DIR"
uv pip install -e "$REPO_ROOT/external/VisGeomBench"
uv pip install -e "$REPO_ROOT/environments/half_subdivision_shaped"
uv run --no-sync rl @ "$CONFIG_PATH" "$@"
