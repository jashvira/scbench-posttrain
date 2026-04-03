"""Prepare half-subdivision parquet files for VeRL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    args = parser.parse_args()

    train_records = load_jsonl(DATASETS["half_subdivision"])
    val_records = load_jsonl(DATASETS["half_subdivision_test"])

    write_parquet(
        to_verl_rows(train_records, data_source="half_subdivision", split="train"),
        args.output_dir / "train.parquet",
    )
    write_parquet(
        to_verl_rows(val_records, data_source="half_subdivision", split="val"),
        args.output_dir / "val.parquet",
    )


if __name__ == "__main__":
    main()
