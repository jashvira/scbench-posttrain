# scbench-posttrain

## Setup

```bash
cd /Users/jashvira/code/scbench-posttrain
uv venv
uv sync --group dev --extra inspect
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

The Linux GPU bootstrap now uses pinned versions for the fragile serving stack:

- `numpy==1.26.4`
- `huggingface-hub==0.36.2`
- `transformers==4.57.6`
- `vllm==0.14.0`
- `antlr4-python3-runtime==4.9.3`
- `omegaconf==2.3.0`
- `verl==0.7.1`

Known-good recovery note for the rented Prime box:

- the remote env had drifted to `transformers==5.5.0.dev0`
- restoring this pinned stack was enough to get vLLM booting again: `transformers==4.57.6`, `huggingface-hub==0.36.2`, `numpy==1.26.4`, `fsspec==2025.9.0`, `packaging==25.0`, `anthropic==0.71.0`

Current known-good server setup on the 2x A100 box:

- model: `Qwen/Qwen3-8B`
- `tensor_parallel_size=2`
- `quantization=fp8`
- endpoint: `http://127.0.0.1:8000/v1/models`

Important caveat:

- these A100s do not have native FP8 support in this path
- vLLM reports weight-only FP8 via Marlin
- vLLM also warns that this may degrade performance

Other stability notes that mattered in practice:

- when starting vLLM over SSH, use a fully detached launch (`setsid`, redirected stdin, log file). A plain backgrounded SSH command let the parent exit and took the server down with it
- for local vLLM evals, set `OPENAI_API_KEY=dummy` and `OPENAI_BASE_URL=http://127.0.0.1:8000/v1`
- the Inspect task arg is `task_name`, not `name`
- `record_indices` can arrive as a repeated list argument under Inspect; `--limit` is the safer quick path unless you need an exact fixed slice
- the half-subdivision prompt/parser now expects a final line of the form `Final answer: ...`; this fixed valid long-form explanations being mis-scored by literal extraction
- if you serve with a small `--max-model-len`, large half-subdivision prompts plus large `--max-tokens` will trip context errors. The current server uses `32768` to avoid that

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
uv run --with openai inspect eval evals/vgb.py@vgb_task --model openai/gpt-5.4-2026-03-05 -T task_name=topology_enumeration
uv run inspect view
```

## Half-Subdivision Shaped Env

Keep benchmark eval binary and use the separate Verifiers env package for
training:

```bash
prime env install environments/half_subdivision_shaped
```

```python
from half_subdivision_shaped import load_environment

env = load_environment(task_name="half_subdivision_test")
```

## PRIME-RL

The repo now has a native PRIME-RL path for half-subdivision:

- bootstrap PRIME-RL workspace:
  [bootstrap_prime_rl_workspace.sh](/Users/jashvira/code/scbench-posttrain/scripts/bootstrap_prime_rl_workspace.sh)
- full run:
  [run_prime_rl_half_subdivision.sh](/Users/jashvira/code/scbench-posttrain/scripts/run_prime_rl_half_subdivision.sh)
- fail-fast smoke run:
  [run_prime_rl_half_subdivision_smoke.sh](/Users/jashvira/code/scbench-posttrain/scripts/run_prime_rl_half_subdivision_smoke.sh)
- configs/docs:
  [training/prime_rl](/Users/jashvira/code/scbench-posttrain/training/prime_rl)

The PRIME-RL setup uses the local
[half_subdivision_shaped](/Users/jashvira/code/scbench-posttrain/environments/half_subdivision_shaped)
environment, defaults to LoRA on `Qwen/Qwen3-8B`, and keeps a smoke config for
fast bring-up on `2x A100 80GB`.

## Check

```bash
uv run pytest
uv run ruff check
```
