# scbench-posttrain

## Setup

```bash
cd /Users/jashvira/code/scbench-posttrain
uv venv
uv sync --group dev --extra inspect
uv pip install git+https://github.com/alexzhang13/rlm.git
```

## Env

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...
```

## Run

```bash
uv run --with openai inspect eval evals/delaunay.py --model openai/gpt-5-nano-2025-08-07
uv run --with openai inspect eval evals/delaunay.py --model openai/gpt-5-nano-2025-08-07 --solver delaunay_rlm_repl
uv run --with openai inspect eval evals/delaunay.py --model openai/gpt-5-nano-2025-08-07 --solver delaunay_rlm_full
uv run inspect view
```

## Check

```bash
uv run pytest
uv run ruff check
```
