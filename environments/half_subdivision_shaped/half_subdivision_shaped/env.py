"""Verifiers env assembly for half-subdivision rewards."""

from __future__ import annotations

import json
from pathlib import Path

import verifiers as vf
from datasets import Dataset

from .geometry import (
    GeometryCase,
    build_geometry_case,
    exact_match,
    resolve_case,
)
from .parser import make_parser, parse_labels

REPO_ROOT = Path(__file__).resolve().parents[3]
TASK_SOURCES = {
    "half_subdivision": REPO_ROOT / "data" / "half_subdivision_curriculum.jsonl",
    "half_subdivision_test": REPO_ROOT / "data" / "half_subdivision_test.jsonl",
}
CURRICULUM_STAGE_ORDER = (
    "stage_01_2d_intro",
    "stage_02_2d_easy",
    "stage_03_2d_medium",
    "stage_04_2d_hard",
    "stage_00_curated",
    "stage_05_3d_intro",
    "stage_06_3d_medium",
    "stage_07_3d_hard",
    "stage_08_3d_topoff",
)
CURRICULUM_STAGE_RANK = {stage: index for index, stage in enumerate(CURRICULUM_STAGE_ORDER)}


def load_environment(
    task_name: str = "half_subdivision",
    *,
    limit: int | None = None,
    curriculum_stage: str | None = None,
    curriculum_max_stage: str | None = None,
    system_prompt: str | None = None,
):
    """Build a single-turn environment for half-subdivision tasks."""

    records = load_records(
        task_name,
        limit=limit,
        curriculum_stage=curriculum_stage,
        curriculum_max_stage=curriculum_max_stage,
    )
    rows, cases = build_dataset(records)
    parser = make_parser()

    rubric = vf.Rubric(parser=parser)
    rubric.add_reward_func(make_reward(cases))
    rubric.add_metric(parseable)
    rubric.add_metric(make_exact_match_metric(cases))

    return vf.SingleTurnEnv(
        dataset=Dataset.from_list(rows),
        rubric=rubric,
        parser=parser,
        system_prompt=system_prompt,
    )


def load_records(
    task_name: str,
    limit: int | None = None,
    curriculum_stage: str | None = None,
    curriculum_max_stage: str | None = None,
) -> list[dict]:
    """Load one of the local half-subdivision datasets."""

    try:
        source_path = TASK_SOURCES[task_name]
    except KeyError as exc:
        raise ValueError(f"Unknown half-subdivision task {task_name!r}") from exc

    with source_path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    records = filter_records(
        records,
        curriculum_stage=curriculum_stage,
        curriculum_max_stage=curriculum_max_stage,
    )

    if limit is not None:
        records = records[:limit]

    if any(
        record.get("metadata", {}).get("problem_type") != "half_subdivision_neighbours"
        for record in records
    ):
        raise ValueError(f"{task_name!r} contains non-half-subdivision records")
    return records


def filter_records(
    records: list[dict],
    *,
    curriculum_stage: str | None,
    curriculum_max_stage: str | None,
) -> list[dict]:
    """Apply optional curriculum-stage filters before slicing."""

    if curriculum_stage is not None:
        validate_curriculum_stage(curriculum_stage)
        records = [
            record
            for record in records
            if record.get("metadata", {}).get("curriculum_stage") == curriculum_stage
        ]

    if curriculum_max_stage is not None:
        max_rank = validate_curriculum_stage(curriculum_max_stage)
        records = [
            record
            for record in records
            if CURRICULUM_STAGE_RANK.get(record.get("metadata", {}).get("curriculum_stage"), -1) <= max_rank
        ]

    return records


def validate_curriculum_stage(stage: str) -> int:
    """Validate a curriculum stage name and return its cumulative rank."""

    try:
        return CURRICULUM_STAGE_RANK[stage]
    except KeyError as exc:
        expected = ", ".join(CURRICULUM_STAGE_ORDER)
        raise ValueError(f"Unknown curriculum stage {stage!r}. Expected one of: {expected}") from exc


def build_dataset(records: list[dict]) -> tuple[list[dict], dict[str, GeometryCase]]:
    """Convert records into a Verifiers dataset and geometry lookup."""

    rows: list[dict] = []
    cases: dict[str, GeometryCase] = {}

    for index, record in enumerate(records):
        token = str(record.get("id", index))
        rows.append(
            {
                "id": token,
                "prompt": [{"role": "user", "content": record["prompt"]}],
                "answer": "",
                # Reward funcs get this `info` field back with the completion.
                "info": json.dumps({"record_token": token}),
            }
        )
        cases[token] = build_geometry_case(record)

    return rows, cases


def make_reward(cases: dict[str, GeometryCase]):
    """Create the main reward function."""

    def reward(parser, completion, *, info=None, **_kwargs) -> float:
        labels, case = parse_completion(parser, completion, info, cases)
        if labels is None or case is None:
            return 0.0

        return exact_match(labels, case)

    reward.__name__ = "reward"
    return reward


def parseable(parser, completion, *, info=None, **_kwargs) -> float:
    """Zero-weight parseability metric."""

    _ = info
    return 1.0 if parse_labels(parser.parse_answer(completion)) is not None else 0.0


def make_exact_match_metric(cases: dict[str, GeometryCase]):
    """Create a zero-weight exact-match metric."""

    def exact_match_metric(parser, completion, *, info=None, **_kwargs) -> float:
        labels, case = parse_completion(parser, completion, info, cases)
        if labels is None or case is None:
            return 0.0
        return exact_match(labels, case)

    exact_match_metric.__name__ = "exact_match"
    return exact_match_metric


def parse_completion(
    parser,
    completion,
    info,
    cases: dict[str, GeometryCase],
) -> tuple[list[str] | None, GeometryCase | None]:
    """Parse the model output and resolve its backing geometry case."""

    return parse_labels(parser.parse_answer(completion)), resolve_case(info, cases)
