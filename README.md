# scbench-posttrain

## Setup

```bash
cd /Users/jashvira/code/scbench-posttrain
uv venv
uv sync --group dev --extra inspect --extra rlm
```

## Prime GPU Container

Use this path once you have Linux GPUs and want a copy-pasteable bring-up for
Qwen on vLLM:

```bash
cd /Users/jashvira/code/scbench-posttrain
./scripts/bootstrap_prime_vllm_container.sh
source .venv/bin/activate
prime login --plain
export HF_TOKEN=...
./scripts/run_vllm_qwen3_8b.sh
```

The server comes up on `http://0.0.0.0:8000/v1`. Point the eval harness at it:

```bash
export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
uv run inspect eval evals/vgb.py@vgb_task --model openai/Qwen/Qwen3-8B -T task_name=delaunay --limit 5
```

This mirrors Prime's documented self-hosted pattern: run a local vLLM
OpenAI-compatible server, then point the evaluator at `http://localhost:8000/v1`.
If you want to use Prime's own evaluator against the same server:

```bash
prime eval gsm8k \
  -m Qwen/Qwen3-8B \
  --api-base-url http://127.0.0.1:8000/v1 \
  --api-key-var OPENAI_API_KEY \
  -n 5 -r 1
```

## Env

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...
```

## Tasks

```text
topology_enumeration
enumerate_edges
classify_behaviour
convex_hull
delaunay
two_segments
shikaku
half_subdivision
half_subdivision_test
```

## Run

```bash
uv run --with openai inspect eval evals/vgb.py@vgb_task --model openai/gpt-5.4-2026-03-05 -T task_name=delaunay
uv run --extra rlm --with openai inspect eval evals/vgb.py@vgb_task --model openai/gpt-5.4-2026-03-05 -T task_name=delaunay --solver vgb_rlm_repl
uv run --extra rlm --with openai inspect eval evals/vgb.py@vgb_task --model openai/gpt-5.4-2026-03-05 -T task_name=delaunay --solver vgb_rlm_full
uv run --with openai inspect eval evals/vgb.py@vgb_task --model openai/gpt-5.4-2026-03-05 -T task_name=topology_enumeration
uv run inspect view
```

## Half-Subdivision Shaped Env

Keep benchmark eval binary and load the shaped training env separately:

```python
from scbench_posttrain.half_subdivision_env import load_half_subdivision_shaped_env

env = load_half_subdivision_shaped_env(task_name="half_subdivision_test")
```

## Check

```bash
uv run pytest
uv run ruff check
```
