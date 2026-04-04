#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/training/prime_rl/h100_8x.toml}"

"$REPO_ROOT/scripts/run_prime_rl_half_subdivision.sh" "$@"
