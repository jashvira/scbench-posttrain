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
uv run python training/verl/prepare_half_subdivision_data.py
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

- regenerates parquet before launch
- exports `REPO_ROOT`, `MODEL_PATH`, and `RUN_DIR` for the Hydra config
- auto-detects GPU count if `N_GPUS` is unset
- starts with conservative defaults for first bring-up: `ROLLOUT_N=4`, `TRAIN_BATCH_SIZE=32`, `VAL_BATCH_SIZE=64`, `MAX_PROMPT_LENGTH=8192`, `MAX_RESPONSE_LENGTH=1024`

Override as needed with:

- `RUN_DIR`
- `N_GPUS`
- `ROLLOUT_TP_SIZE`
- `ROLLOUT_N`
- `TRAIN_BATCH_SIZE`
- `VAL_BATCH_SIZE`
- `MAX_PROMPT_LENGTH`
- `MAX_RESPONSE_LENGTH`
- `GPU_MEMORY_UTILIZATION`
