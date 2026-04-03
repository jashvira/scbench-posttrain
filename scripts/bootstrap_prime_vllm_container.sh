#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This bootstrap is intended for Linux GPU containers." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but not installed." >&2
  exit 1
fi

uv venv --python 3.12
uv sync --group dev --extra inspect --extra rlm
uv pip install --python .venv/bin/python --torch-backend=auto "vllm>=0.8"

if command -v prime >/dev/null 2>&1; then
  uv tool upgrade prime
else
  uv tool install prime
fi

echo
echo "Bootstrapped repo env and Prime CLI."
echo "Next:"
echo "  1. source .venv/bin/activate"
echo "  2. prime login --plain"
echo "  3. export HF_TOKEN=..."
echo "  4. ./scripts/run_vllm_qwen3_8b.sh"
