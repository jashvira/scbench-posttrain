# VeRL half-subdivision

This is the minimal VeRL surface for GRPO on the half-subdivision task.

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
  train_files: /Users/jashvira/code/scbench-posttrain/training/verl/data/train.parquet
  val_files: /Users/jashvira/code/scbench-posttrain/training/verl/data/val.parquet
  prompt_key: prompt

algorithm:
  adv_estimator: grpo

actor_rollout_ref:
  rollout:
    n: 8

custom_reward_function:
  path: /Users/jashvira/code/scbench-posttrain/training/verl/half_subdivision_reward.py
  name: compute_score
```
