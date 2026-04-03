# VeRL half-subdivision

This is the minimal VeRL surface for GRPO on the half-subdivision task.

Use the pinned training stack from the repo bootstrap on GPU boxes:

```bash
cd /Users/jashvira/code/scbench-posttrain
./scripts/bootstrap_prime_vllm_container.sh
```

## Prepare data

```bash
cd /Users/jashvira/code/scbench-posttrain
uv run python training/verl/prepare_half_subdivision_data.py \
  --model-path /path/to/Qwen/Qwen3-8B \
  --max-prompt-length 16384 \
  --max-response-length 16384 \
  --total-token-budget 32768
```

This writes:

- `training/verl/data/train.parquet`
- `training/verl/data/val.parquet`

The parquet rows follow VeRL's expected fields:

- `data_source`
- `prompt`
- `ability`
- `reward_model.ground_truth`
- `extra_info`

## Reward hook

Use [half_subdivision_reward.py](/Users/jashvira/code/scbench-posttrain/training/verl/half_subdivision_reward.py) as the custom reward function path in VeRL.

It exposes:

- `compute_score(data_source, solution_str, ground_truth, extra_info)`

The reward matches the local shaped env:

- face neighbor: `1.0`
- edge/corner near-miss: `0.25`
- unrelated: `0.0`
- normalized by `max(len(truth), len(valid_predictions))`

The VeRL reward hook also returns subscore fields for W&B validation graphs:

- `parseable`
- `face_credit`
- `near_contact_credit`
- `valid_prediction_fraction`
- `pred_count`
- `truth_count`

## VeRL config fields

Point your trainer config at the local parquet files and reward hook:

```yaml
data:
  train_files: ${oc.env:REPO_ROOT}/training/verl/data/train.parquet
  val_files: ${oc.env:REPO_ROOT}/training/verl/data/val.parquet
  prompt_key: prompt

algorithm:
  adv_estimator: grpo

actor_rollout_ref:
  rollout:
    n: 4

custom_reward_function:
  path: ${oc.env:REPO_ROOT}/training/verl/half_subdivision_reward.py
  name: compute_score
```

## Launch

Use the provided script on the GPU box:

```bash
cd /Users/jashvira/code/scbench-posttrain
MODEL_PATH=/path/to/Qwen/Qwen3-8B \
N_GPUS=2 \
./scripts/run_verl_grpo_half_subdivision.sh
```

The launcher now:

- regenerates parquet before launch using the actual model tokenizer
- exports `REPO_ROOT`, `MODEL_PATH`, and `RUN_DIR` for the Hydra config
- auto-detects GPU count if `N_GPUS` is unset
- starts with conservative defaults for first bring-up: `ROLLOUT_N=4`, `TRAIN_BATCH_SIZE=32`, `VAL_BATCH_SIZE=64`, `MAX_PROMPT_LENGTH=16384`, `MAX_RESPONSE_LENGTH=16384`
- sets `TOTAL_TOKEN_BUDGET=MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH` for actor PPO and rollout log-prob passes, which VeRL needs because those paths see full prompt-plus-response sequences
- writes already-filtered parquet rows, instead of relying only on VeRL's prompt-only overlength filter
- runs validation on the 10-example `half_subdivision_test` slice every `50` training steps by default
- creates writable runtime dirs for `XDG_CONFIG_HOME`, `XDG_CACHE_HOME`, and `MPLCONFIGDIR` under the repo so vLLM and matplotlib do not crash on locked-down GPU boxes

Override as needed with:

- `RUN_DIR`
- `N_GPUS`
- `ROLLOUT_TP_SIZE`
- `ROLLOUT_N`
- `TRAIN_BATCH_SIZE`
- `VAL_BATCH_SIZE`
- `MAX_PROMPT_LENGTH`
- `MAX_RESPONSE_LENGTH`
- `TOTAL_TOKEN_BUDGET`
- `GPU_MEMORY_UTILIZATION`
- `TEST_FREQ`

## Stability Notes

These were the fixes that actually mattered for getting a dummy GRPO run through
first-step startup on the 2x A100 box:

- use VeRL's base `ppo_trainer` config directly; the local task setup is applied
  as CLI overrides by the launcher
- keep `flash-attn` installed and use `bfloat16` for actor/ref model dtype
- kill any manual eval `vllm` server before training; VeRL starts its own rollout
  backend
- budget tokens for full sequence length, not just prompt length:
  `TOTAL_TOKEN_BUDGET = MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH`
- pass that total budget into:
  - `actor_rollout_ref.actor.ppo_max_token_len_per_gpu`
  - `actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu`
  - `actor_rollout_ref.rollout.max_num_batched_tokens`
- create writable runtime dirs on the box:
  - `XDG_CONFIG_HOME`
  - `XDG_CACHE_HOME`
  - `MPLCONFIGDIR`
- the old failure was:
  - `AssertionError: max_token_len must be greater than the sequence length`
  - this happened because VeRL log-prob saw prompt plus response, while the old
    launcher only budgeted prompt length

The current launcher has already baked these fixes in.

## Validation Cadence

Current VeRL setup validates on the 10-example `half_subdivision_test` slice:

- once before training starts
- every `TEST_FREQ` steps during training

Default:

- `TEST_FREQ=50`

Those validation metrics are what drive the W&B `val-core/...` and `val-aux/...`
graphs over time.
