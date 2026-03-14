# scbench-posttrain

Small SCBench post-training scaffold.

The repo is intentionally flat for now:

- `tasks.py`: shared task/verifier types and a tiny registry
- `inspect.py`: Inspect adapter
- `tinker.py`: single-turn Tinker env adapter

## Getting Started

```bash
cd /Users/jashvira/code/scbench-posttrain
uv venv
uv sync --group dev
```

If you need the optional integrations later:

```bash
uv sync --group dev --extra inspect --extra tinker
```

## Next

- add the first concrete SCBench task family
- add one eval entrypoint
- add training code only once the first task exists
