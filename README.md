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
inspect eval evals/delaunay.py
inspect eval evals/delaunay.py --solver delaunay_rlm_repl
inspect eval evals/delaunay.py --solver delaunay_rlm_full
inspect view
```

## Check

```bash
uv run pytest
uv run ruff check
```
