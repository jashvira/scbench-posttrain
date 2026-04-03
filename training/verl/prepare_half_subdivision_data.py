"""Prepare half-subdivision parquet files for VeRL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "half_subdivision": REPO_ROOT / "data" / "half_subdivision_curriculum.jsonl",
    "half_subdivision_test": REPO_ROOT / "data" / "half_subdivision_test.jsonl",
}


def load_jsonl(path: Path) -> list[dict]:
    """Load records from a JSONL file."""

    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def to_verl_rows(records: list[dict], *, data_source: str, split: str) -> list[dict]:
    """Convert half-subdivision records into VeRL parquet rows."""

    rows: list[dict] = []
    for index, record in enumerate(records):
        rows.append(
            {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": record["prompt"]}],
                "ability": "spatial",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps(record["ground_truth"]),
                },
                "extra_info": {
                    "split": split,
                    "index": index,
                    "record_id": record["id"],
                    "datagen_args": record["datagen_args"],
                    "runtime": record["runtime"],
                },
            }
        )
    return rows


def count_prompt_tokens(tokenizer: Any, prompt: list[dict]) -> int:
    """Count prompt tokens using the model's chat template when available."""

    try:
        token_ids = tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
        )
    except Exception:
        text = "\n".join(str(message.get("content", "")) for message in prompt)
        token_ids = tokenizer(text, add_special_tokens=True)["input_ids"]
    return len(token_ids)


def filter_rows(
    rows: list[dict],
    *,
    model_path: str | None,
    max_prompt_length: int | None,
    max_response_length: int | None,
    total_token_budget: int | None,
) -> list[dict]:
    """Filter rows against the actual tokenizer and sequence budget."""

    if model_path is None or max_prompt_length is None:
        return rows

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    filtered_rows: list[dict] = []

    for row in rows:
        prompt_tokens = count_prompt_tokens(tokenizer, row["prompt"])
        total_tokens = prompt_tokens + (max_response_length or 0)
        if prompt_tokens > max_prompt_length:
            continue
        if total_token_budget is not None and total_tokens > total_token_budget:
            continue
        row["extra_info"]["prompt_tokens"] = prompt_tokens
        row["extra_info"]["max_response_length"] = max_response_length
        row["extra_info"]["total_token_budget"] = total_token_budget
        filtered_rows.append(row)

    return filtered_rows


def write_parquet(rows: list[dict], path: Path) -> None:
    """Write rows to parquet."""

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def main() -> None:
    """Create VeRL train/val parquet files for half-subdivision."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=REPO_ROOT / "training" / "verl" / "data",
        type=Path,
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--max-prompt-length", type=int, default=None)
    parser.add_argument("--max-response-length", type=int, default=None)
    parser.add_argument("--total-token-budget", type=int, default=None)
    args = parser.parse_args()

    train_records = load_jsonl(DATASETS["half_subdivision"])
    val_records = load_jsonl(DATASETS["half_subdivision_test"])

    train_rows = filter_rows(
        to_verl_rows(train_records, data_source="half_subdivision", split="train"),
        model_path=args.model_path,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        total_token_budget=args.total_token_budget,
    )
    val_rows = filter_rows(
        to_verl_rows(val_records, data_source="half_subdivision", split="val"),
        model_path=args.model_path,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        total_token_budget=args.total_token_budget,
    )

    write_parquet(
        train_rows,
        args.output_dir / "train.parquet",
    )
    write_parquet(
        val_rows,
        args.output_dir / "val.parquet",
    )

    print(
        f"train: kept {len(train_rows)}/{len(train_records)} | "
        f"val: kept {len(val_rows)}/{len(val_records)}"
    )


if __name__ == "__main__":
    main()
