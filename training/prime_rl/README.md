# PRIME-RL half-subdivision

This is the repo-native PRIME-RL path for the local
`half-subdivision-shaped` Verifiers environment.

Use the official PRIME-RL stack shape:

- install the local environment editable
- run `uv run rl @ ...`
- let PRIME-RL manage trainer, orchestrator, and inference

## Bootstrap

On a Prime RL image, `prime-rl` already lives in `/workspace/prime-rl`.

If not, use:

```bash
cd /Users/jashvira/code/scbench-posttrain
PRIME_RL_DIR=/workspace/prime-rl \
./scripts/bootstrap_prime_rl_workspace.sh
```

That will:

- clone `prime-rl` if missing
- pin it to the repo-tested commit
- `uv sync --all-extras`
- install [half_subdivision_shaped](/Users/jashvira/code/scbench-posttrain/environments/half_subdivision_shaped) editable into the PRIME-RL workspace

## Configs

- [rl.toml](/Users/jashvira/code/scbench-posttrain/training/prime_rl/rl.toml)
  - default 2-GPU LoRA GRPO run
  - `1` train GPU, `1` infer GPU
  - full curriculum train, 10-example test eval

- [smoke.toml](/Users/jashvira/code/scbench-posttrain/training/prime_rl/smoke.toml)
  - fail-fast bring-up
  - first `16` train examples
  - first `4` test examples
  - shorter run, eval every `10` steps

Both configs use:

- `Qwen/Qwen3-8B`
- LoRA (`rank=32`, `alpha=64`)
- `max_tokens = 2048`

## Run

Smoke:

```bash
cd /Users/jashvira/code/scbench-posttrain
PRIME_RL_DIR=/workspace/prime-rl \
./scripts/run_prime_rl_half_subdivision_smoke.sh
```

Full:

```bash
cd /Users/jashvira/code/scbench-posttrain
PRIME_RL_DIR=/workspace/prime-rl \
./scripts/run_prime_rl_half_subdivision.sh
```

Both wrappers:

- install the local env into the PRIME-RL workspace first
- run from the PRIME-RL checkout
- accept extra CLI overrides after the script name

Example:

```bash
./scripts/run_prime_rl_half_subdivision_smoke.sh \
  --wandb \
  --wandb.project half-subdivision-prime-rl \
  --wandb.name qwen3-8b-smoke \
  --max-steps 20
```

## Notes

- use LoRA first on `2x A100 80GB`; full finetune is the wrong starting point
- `seq_len` is full prompt+completion budget in PRIME-RL
- smoke uses env-level `limit` args, so no separate parquet prep path is needed
