#!/usr/bin/env python3
from __future__ import annotations

import json
import random
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


def _target_depth(label: str) -> int:
    return 0 if label == '""' else len(label)


# The score is a proxy for actual reasoning burden: larger leaf sets, deeper
# targets, longer axis schedules, more neighbours to enumerate, and 3D face
# adjacency all make the task meaningfully harder.
def _difficulty_score(record: dict[str, Any]) -> int:
    runtime = record["runtime"]
    return (
        (10_000 if runtime["dimension"] == "3D" else 0)
        + int(runtime["leaf_count"]) * 28
        + _target_depth(runtime["target_label"]) * 80
        + len(runtime["axis_cycle"]) * 25
        + int(runtime["neighbour_count"]) * 35
        + int(runtime["max_depth"]) * 60
        + int(runtime["min_depth"]) * 30
        + round(float(runtime["split_prob"]) * 100)
    )


def _label_sort_key(label: str) -> tuple[int, str]:
    raw = "" if label == '""' else label
    return (len(raw), raw)


def _format_leaf_block(labels: list[str], *, row_width: int = 12) -> str:
    rows = [
        ", ".join(json.dumps(label) for label in labels[index : index + row_width])
        for index in range(0, len(labels), row_width)
    ]
    return "[\n  " + ",\n  ".join(rows) + "\n]"


def _prepare_compact_case(datagen_args: dict) -> tuple[list[Any], Any, list[Any], dict[str, Any]]:
    from treelib import Tree
    from visual_geometry_bench.datagen.half_subdivision_neighbours import (
        Dimension,
        _are_adjacent,
        _build_subdivision,
        _normalise_label,
        _resolve_axis_cycle,
    )

    if not isinstance(datagen_args, dict):
        raise TypeError("datagen_args must be a dictionary")

    max_depth = int(datagen_args["max_depth"])
    min_depth = int(datagen_args.get("min_depth", 0))
    split_prob = float(datagen_args["split_prob"])
    seed = int(datagen_args["seed"])
    dim_name = str(datagen_args.get("dimension", "2D")).upper()
    if dim_name == "2D":
        dim = Dimension.D2
    elif dim_name == "3D":
        dim = Dimension.D3
    else:
        raise ValueError(f"Invalid dimension {dim_name!r}")
    axis_cycle = _resolve_axis_cycle(
        dim,
        axis_cycle=datagen_args.get("axis_cycle"),
        start_axis=datagen_args.get("start_axis"),
    )
    target_label = _normalise_label(datagen_args.get("target_label"))

    rng = random.Random(seed)
    tree = Tree()
    bounds = {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0}
    if dim == Dimension.D3:
        bounds.update({"z0": 0.0, "z1": 1.0})

    leaves = _build_subdivision(
        tree=tree,
        parent_id=None,
        label="",
        depth=0,
        max_depth=max_depth,
        min_depth=min_depth,
        split_prob=split_prob,
        axis_cycle=axis_cycle,
        dim=dim,
        rng=rng,
        **bounds,
    )
    leaves_by_label = {leaf.label: leaf for leaf in leaves}
    target = rng.choice(leaves) if target_label is None else leaves_by_label[target_label]
    neighbours = sorted(
        (leaf for leaf in leaves if leaf is not target and _are_adjacent(leaf, target, dim)),
        key=lambda leaf: leaf.label,
    )
    runtime_info = {
        "max_depth": max_depth,
        "min_depth": min_depth,
        "split_prob": split_prob,
        "seed": seed,
        "start_axis": axis_cycle[0],
        "axis_cycle": list(axis_cycle),
        "dimension": dim_name,
        "target_label": target.display_label(),
        "leaf_count": len(leaves),
    }
    return leaves, target, neighbours, runtime_info


def _format_compact_prompt(leaves: list[Any], target: Any, runtime_info: dict[str, Any]) -> str:
    dim_name = runtime_info["dimension"]
    shape = "unit cube" if dim_name == "3D" else "unit square"
    contact = "shares a face with the target voxel" if dim_name == "3D" else (
        "shares a boundary segment with the target"
    )
    axis_cycle_text = " -> ".join(runtime_info["axis_cycle"])
    leaf_labels = sorted((leaf.display_label() for leaf in leaves), key=_label_sort_key)
    leaf_block = _format_leaf_block(leaf_labels)
    intro = (
        "You are given the terminal leaves of a binary-tree description of an axis-aligned "
        f"half subdivision of the {shape}."
    )
    return (
        f"{intro}\n\n"
        "Each node splits its parent cell into two children by bisecting along axes in the "
        f"repeating cycle {axis_cycle_text} (repeating).\n\n"
        "Instead of the full tree, you are given the terminal leaves only. Each label is the "
        "root-to-leaf bitstring for a terminal cell: at each depth, bit 0 selects the lower "
        "half and bit 1 selects the upper half along that split axis.\n\n"
        f"Here are the terminal leaves of the subdivision:\n\n{leaf_block}\n\n"
        f"Target leaf: {target.display_label()}\n\n"
        "Before presenting the final list, begin your response with <thinking>...</thinking> "
        "containing your full chain of thought or reasoning for your answer.\n"
        f"List every terminal leaf that {contact}. Return the labels as a comma-separated "
        "list of strings (quotes optional)."
    )


def _build_compact_record(
    datagen_args: dict,
    *,
    tags: list[str] | None = None,
    difficulty: str | None = None,
) -> dict[str, Any]:
    from visual_geometry_bench.datagen.utils import compute_content_hash

    leaves, target, neighbours, runtime_info = _prepare_compact_case(datagen_args)
    axis_cycle = tuple(runtime_info["axis_cycle"])
    prompt = _format_compact_prompt(leaves, target, runtime_info)
    ground_truth = sorted((leaf.display_label() for leaf in neighbours), key=_label_sort_key)

    stored_datagen_args = {**datagen_args, "axis_cycle": list(axis_cycle)}
    metadata = {"problem_type": "half_subdivision_neighbours"}
    if tags:
        metadata["tags"] = list(tags)
    if difficulty:
        metadata["difficulty"] = difficulty

    content_id = compute_content_hash(
        problem_type="half_subdivision_neighbours",
        datagen_args=stored_datagen_args,
        prompt=prompt,
        ground_truth=ground_truth,
    )
    return {
        "id": content_id,
        "prompt": prompt,
        "ground_truth": ground_truth,
        "metadata": metadata,
        "datagen_args": stored_datagen_args,
        "runtime": {
            "target_label": target.display_label(),
            "neighbour_count": len(ground_truth),
            **runtime_info,
        },
    }


def _load_base_records() -> list[dict[str, Any]]:
    from visual_geometry_bench.dataset import build_records_from_config, load_config

    config = load_config(BASE_CONFIG_PATH)
    source_records = build_records_from_config(config)
    records = [
        _build_compact_record(
            record["datagen_args"],
            tags=record.get("metadata", {}).get("tags"),
            difficulty=record.get("metadata", {}).get("difficulty"),
        )
        for record in source_records
    ]
    for record in records:
        metadata = record.setdefault("metadata", {})
        metadata["curriculum_source"] = "base_curated"
        metadata["curriculum_stage"] = "stage_00_curated"
        metadata["curriculum_score"] = _difficulty_score(record)
    return records


def _build_profile_records(profile: Profile, profile_index: int) -> list[dict[str, Any]]:
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

        record = _build_compact_record(
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
