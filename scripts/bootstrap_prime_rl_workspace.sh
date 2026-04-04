#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRIME_RL_DIR="${PRIME_RL_DIR:-/workspace/prime-rl}"
PRIME_RL_REF="${PRIME_RL_REF:-bffd3103b29c60c972fd3829a74fd9c01506d855}"

git -C "$REPO_ROOT" submodule update --init --recursive external/VisGeomBench

if [[ ! -d "$PRIME_RL_DIR/.git" ]]; then
  git clone https://github.com/PrimeIntellect-ai/prime-rl.git "$PRIME_RL_DIR"
fi

git -C "$PRIME_RL_DIR" fetch --depth 1 origin main
git -C "$PRIME_RL_DIR" checkout "$PRIME_RL_REF"

cd "$PRIME_RL_DIR"
uv sync --all-extras
uv pip install -e "$REPO_ROOT/external/VisGeomBench"
uv pip install -e "$REPO_ROOT/environments/half_subdivision_shaped"

uv run --no-sync python - <<'PY'
import verifiers as vf
env = vf.load_environment("half-subdivision-shaped", task_name="half_subdivision_test", limit=1)
print(type(env).__name__)
PY
