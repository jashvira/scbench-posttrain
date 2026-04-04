# PRIME-RL half-subdivision

This is the repo-native PRIME-RL path for the local
`half-subdivision-shaped` Verifiers environment.

Use the official PRIME-RL stack shape:

- install the local environment editable
- run `uv run rl @ ...`
- let PRIME-RL manage trainer, orchestrator, and inference

This setup follows the practical PRIME-RL defaults that matter here:

- LoRA first, not full finetune
- smoke run before the full run
- benchmark before long training if hardware usage is unclear
- periodic checkpointing on longer runs
- periodic eval on the held-out 10-example slice
- no difficulty buffer on this task, because baseline reward can be all-zero

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
  - `seq_len = 12288`
  - `batch_size = 64`
  - `rollouts_per_example = 4`

- [smoke.toml](/Users/jashvira/code/scbench-posttrain/training/prime_rl/smoke.toml)
  - fail-fast bring-up
  - first `16` train examples
  - first `4` test examples
  - shorter run, eval every `10` steps

- [h100_8x.toml](/Users/jashvira/code/scbench-posttrain/training/prime_rl/h100_8x.toml)
  - single-node `8x H100 80GB` setup
  - `2` train GPUs, `6` infer GPUs
  - `seq_len = 16384`
  - `batch_size = 256`
  - `rollouts_per_example = 8`

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

8x H100:

```bash
cd /Users/jashvira/code/scbench-posttrain
PRIME_RL_DIR=/workspace/prime-rl \
./scripts/run_prime_rl_half_subdivision_h100_8x.sh
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

Benchmark-first pass:

```bash
./scripts/run_prime_rl_half_subdivision_smoke.sh --bench
```

Sanity-check pass before long RL:

- run the smoke config first
- confirm the environment loads and rollouts complete
- check that validation is producing non-zero parseable behavior on at least some samples before committing to a long run

## Notes

- use LoRA first on `2x A100 80GB`; full finetune is the wrong starting point
- `seq_len` is full prompt+completion budget in PRIME-RL
- smoke uses env-level `limit` args, so no separate parquet prep path is needed
- full config checkpoints every `50` steps; smoke checkpoints every `10`
- on Hopper boxes, install FlashAttention3 in the PRIME-RL workspace:

```bash
cd /workspace/prime-rl
uv pip install "flash-attn-3 @ git+https://github.com/Dao-AILab/flash-attention.git@main#subdirectory=hopper" --no-build-isolation
```

- after installing FlashAttention3, keep using the repo wrappers here because they run `uv run --no-sync`, which avoids uninstalling the package again
