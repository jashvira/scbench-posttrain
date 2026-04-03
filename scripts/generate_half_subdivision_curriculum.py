#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VGB_ROOT = ROOT / "external" / "VisGeomBench"
OUTPUT_PATH = ROOT / "data" / "half_subdivision_curriculum.jsonl"
TEST_OUTPUT_PATH = ROOT / "data" / "half_subdivision_test.jsonl"
BASE_CONFIG_PATH = VGB_ROOT / "configs" / "half_subdivision.toml"
TOTAL_RECORDS = 300

TEST_STAGE_QUOTAS: tuple[tuple[str, int], ...] = (
    ("stage_01_2d_intro", 1),
    ("stage_02_2d_easy", 1),
    ("stage_03_2d_medium", 1),
    ("stage_04_2d_hard", 2),
    ("stage_00_curated", 1),
    ("stage_05_3d_intro", 1),
    ("stage_06_3d_medium", 1),
    ("stage_07_3d_hard", 2),
)

sys.path.insert(0, str(VGB_ROOT))


@dataclass(frozen=True)
class Profile:
    name: str
    dimension: str
    count: int
    max_depth_range: tuple[int, int]
    min_depth_range: tuple[int, int]
    split_prob_range: tuple[float, float]
    axis_cycles: tuple[tuple[str, ...], ...]
    difficulty: str


PROFILES: tuple[Profile, ...] = (
    Profile(
        name="stage_01_2d_intro",
        dimension="2D",
        count=44,
        max_depth_range=(3, 4),
        min_depth_range=(1, 2),
        split_prob_range=(0.20, 0.35),
        axis_cycles=(("x", "y"), ("y", "x"), ("x", "y", "x"), ("y", "x", "y")),
        difficulty="easy",
    ),
    Profile(
        name="stage_02_2d_easy",
        dimension="2D",
        count=44,
        max_depth_range=(4, 6),
        min_depth_range=(2, 4),
        split_prob_range=(0.35, 0.50),
        axis_cycles=(("x", "y"), ("y", "x"), ("x", "x", "y"), ("y", "y", "x")),
        difficulty="easy",
    ),
    Profile(
        name="stage_03_2d_medium",
        dimension="2D",
        count=44,
        max_depth_range=(6, 7),
        min_depth_range=(4, 5),
        split_prob_range=(0.50, 0.68),
        axis_cycles=(("x", "y", "x"), ("y", "x", "y"), ("x", "y", "y", "x")),
        difficulty="medium",
    ),
    Profile(
        name="stage_04_2d_hard",
        dimension="2D",
        count=44,
        max_depth_range=(7, 9),
        min_depth_range=(5, 7),
        split_prob_range=(0.68, 0.88),
        axis_cycles=(("x", "x", "y", "y"), ("y", "x", "x", "y"), ("x", "y", "x", "y", "x")),
        difficulty="hard",
    ),
    Profile(
        name="stage_05_3d_intro",
        dimension="3D",
        count=36,
        max_depth_range=(4, 6),
        min_depth_range=(2, 4),
        split_prob_range=(0.35, 0.52),
        axis_cycles=(("x", "y", "z"), ("z", "y", "x"), ("x", "z", "y")),
        difficulty="medium",
    ),
    Profile(
        name="stage_06_3d_medium",
        dimension="3D",
        count=36,
        max_depth_range=(6, 8),
        min_depth_range=(4, 6),
        split_prob_range=(0.52, 0.72),
        axis_cycles=(("x", "y", "z"), ("y", "z", "x"), ("z", "x", "y"), ("x", "z", "z", "y")),
        difficulty="hard",
    ),
    Profile(
        name="stage_07_3d_hard",
        dimension="3D",
        count=35,
        max_depth_range=(8, 10),
        min_depth_range=(6, 8),
        split_prob_range=(0.72, 0.92),
        axis_cycles=(
            ("x", "y", "z", "x"),
            ("z", "y", "x", "z"),
            ("x", "z", "y", "x", "y"),
            ("y", "x", "z", "z", "y"),
        ),
        difficulty="hard",
    ),
)


def _lerp(start: float, end: float, t: float) -> float:
    return start + (end - start) * t


def _lerp_int(start: int, end: int, t: float) -> int:
    return round(_lerp(float(start), float(end), t))


def _tree_line_count(prompt: str) -> int:
    match = re.search(r"```(?:\n)?(.*?)```", prompt, re.S)
    if match is None:
        return 0
    return sum(1 for line in match.group(1).splitlines() if line.strip())


def _target_depth(label: str) -> int:
    return 0 if label == '""' else len(label)


# The score is a proxy for actual reasoning burden: larger trees, deeper targets,
# longer axis schedules, more neighbours to enumerate, and 3D face adjacency all
# make the task meaningfully harder.
def _difficulty_score(record: dict[str, Any]) -> int:
    runtime = record["runtime"]
    return (
        (10_000 if runtime["dimension"] == "3D" else 0)
        + _tree_line_count(record["prompt"]) * 12
        + _target_depth(runtime["target_label"]) * 80
        + len(runtime["axis_cycle"]) * 25
        + int(runtime["neighbour_count"]) * 35
        + int(runtime["max_depth"]) * 60
        + int(runtime["min_depth"]) * 30
        + round(float(runtime["split_prob"]) * 100)
    )


def _load_base_records() -> list[dict[str, Any]]:
    from visual_geometry_bench.dataset import build_records_from_config, load_config

    config = load_config(BASE_CONFIG_PATH)
    records = build_records_from_config(config)
    for record in records:
        metadata = record.setdefault("metadata", {})
        metadata["curriculum_source"] = "base_curated"
        metadata["curriculum_stage"] = "stage_00_curated"
        metadata["curriculum_score"] = _difficulty_score(record)
    return records


def _build_profile_records(profile: Profile, profile_index: int) -> list[dict[str, Any]]:
    from visual_geometry_bench.datagen.half_subdivision_neighbours import generate_dataset_record

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    attempt = 0

    while len(records) < profile.count:
        span = max(profile.count - 1, 1)
        t = len(records) / span
        max_depth = _lerp_int(*profile.max_depth_range, t)
        min_depth = min(max_depth, _lerp_int(*profile.min_depth_range, t))
        split_prob = round(_lerp(*profile.split_prob_range, t), 3)
        axis_cycle = list(profile.axis_cycles[attempt % len(profile.axis_cycles)])
        seed = (profile_index + 1) * 100_000 + attempt * 7_919 + len(records) * 131

        record = generate_dataset_record(
            {
                "max_depth": max_depth,
                "min_depth": min_depth,
                "split_prob": split_prob,
                "seed": seed,
                "dimension": profile.dimension,
                "axis_cycle": axis_cycle,
            },
            tags=["curriculum", "half_subdivision", profile.name, profile.dimension.lower()],
            difficulty=profile.difficulty,
        )
        attempt += 1

        if record["id"] in seen_ids or record["prompt"] in seen_prompts:
            continue

        seen_ids.add(record["id"])
        seen_prompts.add(record["prompt"])
        metadata = record.setdefault("metadata", {})
        metadata["curriculum_source"] = "generated"
        metadata["curriculum_stage"] = profile.name
        metadata["curriculum_score"] = _difficulty_score(record)
        records.append(record)

    records.sort(
        key=lambda record: (
            int(record["metadata"]["curriculum_score"]),
            record["runtime"]["target_label"],
            record["id"],
        )
    )
    return records


def _build_test_slice(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    test_records: list[dict[str, Any]] = []
    source_index_by_id = {record["id"]: index for index, record in enumerate(records)}
    bucket = 0

    for stage_name, quota in TEST_STAGE_QUOTAS:
        stage_records = [
            record
            for record in records
            if record["metadata"].get("curriculum_stage") == stage_name
        ]
        if len(stage_records) < quota:
            raise SystemExit(f"Stage {stage_name} has {len(stage_records)} records, need {quota}")

        for rank in range(quota):
            index = ((2 * rank + 1) * len(stage_records)) // (2 * quota)
            index = min(index, len(stage_records) - 1)
            record = json.loads(json.dumps(stage_records[index]))
            metadata = record.setdefault("metadata", {})
            metadata["slice"] = "test"
            metadata["slice_bucket"] = bucket
            metadata["slice_source_index"] = source_index_by_id[record["id"]]
            test_records.append(record)
            bucket += 1

    test_records.sort(key=lambda record: int(record["metadata"]["curriculum_score"]))
    for rank, record in enumerate(test_records):
        record["metadata"]["slice_rank"] = rank
    return test_records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    records = _load_base_records()
    seen_ids = {record["id"] for record in records}
    seen_prompts = {record["prompt"] for record in records}

    for profile_index, profile in enumerate(PROFILES):
        for record in _build_profile_records(profile, profile_index):
            if record["id"] in seen_ids or record["prompt"] in seen_prompts:
                continue
            seen_ids.add(record["id"])
            seen_prompts.add(record["prompt"])
            records.append(record)

    if len(records) < TOTAL_RECORDS:
        topoff_profile = Profile(
            name="stage_08_3d_topoff",
            dimension="3D",
            count=TOTAL_RECORDS - len(records),
            max_depth_range=(9, 10),
            min_depth_range=(7, 8),
            split_prob_range=(0.84, 0.94),
            axis_cycles=PROFILES[-1].axis_cycles,
            difficulty="hard",
        )
        for record in _build_profile_records(topoff_profile, len(PROFILES)):
            if record["id"] in seen_ids or record["prompt"] in seen_prompts:
                continue
            seen_ids.add(record["id"])
            seen_prompts.add(record["prompt"])
            records.append(record)

    if len(records) != TOTAL_RECORDS:
        raise SystemExit(f"Expected {TOTAL_RECORDS} records, got {len(records)}")

    records.sort(
        key=lambda record: (
            int(record["metadata"]["curriculum_score"]),
            record["metadata"]["curriculum_stage"],
            record["id"],
        )
    )
    for index, record in enumerate(records):
        record["metadata"]["curriculum_index"] = index

    test_records = _build_test_slice(records)
    _write_jsonl(OUTPUT_PATH, records)
    _write_jsonl(TEST_OUTPUT_PATH, test_records)

    print(f"Wrote {len(records)} records to {OUTPUT_PATH}")
    print(f"Wrote {len(test_records)} records to {TEST_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
