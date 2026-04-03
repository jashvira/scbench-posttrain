"""Verifiers env assembly for half-subdivision shaped rewards."""

from __future__ import annotations

import json
from pathlib import Path

import verifiers as vf
from datasets import Dataset

from .geometry import (
    GeometryCase,
    build_geometry_case,
    geometric_credit_sum,
    resolve_case,
    shaped_score,
    valid_predictions,
)
from .parser import make_parser, parse_labels

NEAR_CONTACT_CREDIT = 0.25

REPO_ROOT = Path(__file__).resolve().parents[3]
TASK_SOURCES = {
    "half_subdivision": REPO_ROOT / "data" / "half_subdivision_curriculum.jsonl",
    "half_subdivision_test": REPO_ROOT / "data" / "half_subdivision_test.jsonl",
}


def load_environment(
    task_name: str = "half_subdivision",
    *,
    system_prompt: str | None = None,
    near_contact_credit: float = NEAR_CONTACT_CREDIT,
):
    """Build a shaped single-turn environment for half-subdivision tasks."""

    records = load_records(task_name)
    rows, cases = format_dataset(records)
    parser = make_parser()

    rubric = vf.Rubric(parser=parser)
    rubric.add_reward_func(
        make_shaped_reward(
            cases,
            near_contact_credit=near_contact_credit,
        )
    )
    rubric.add_metric(parseable)
    rubric.add_metric(make_valid_labels_metric(cases))
    rubric.add_metric(make_geometric_credit_metric(cases, near_contact_credit=near_contact_credit))

    return vf.SingleTurnEnv(
        dataset=Dataset.from_list(rows),
        rubric=rubric,
        parser=parser,
        system_prompt=system_prompt,
    )


def load_records(task_name: str) -> list[dict]:
    """Load one of the local half-subdivision datasets."""

    try:
        path = TASK_SOURCES[task_name]
    except KeyError as exc:
        raise ValueError(f"Unknown half-subdivision task {task_name!r}") from exc

    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    if any(
        record.get("metadata", {}).get("problem_type") != "half_subdivision_neighbours"
        for record in records
    ):
        raise ValueError(f"{task_name!r} contains non-half-subdivision records")
    return records


def format_dataset(
    records: list[dict],
) -> tuple[list[dict], dict[str, GeometryCase]]:
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
                "info": json.dumps({"record_token": token}),
            }
        )
        cases[token] = build_geometry_case(record)

    return rows, cases


def make_shaped_reward(
    cases: dict[str, GeometryCase],
    *,
    near_contact_credit: float,
):
    """Create the main shaped reward function."""

    def shaped_reward(parser, completion, *, info=None, **_kwargs) -> float:
        labels = parse_labels(parser.parse_answer(completion))
        if labels is None:
            return 0.0

        case = resolve_case(info, cases)
        if case is None:
            return 0.0

        return shaped_score(labels, case, near_contact_credit)

    shaped_reward.__name__ = "shaped_reward"
    return shaped_reward


def parseable(parser, completion, *, info=None, **_kwargs) -> float:
    """Zero-weight parseability metric."""

    _ = info
    return 1.0 if parse_labels(parser.parse_answer(completion)) is not None else 0.0


def make_valid_labels_metric(cases: dict[str, GeometryCase]):
    """Create a zero-weight valid-label metric."""

    def valid_labels(parser, completion, *, info=None, **_kwargs) -> float:
        labels = parse_labels(parser.parse_answer(completion))
        case = resolve_case(info, cases)
        if labels is None or case is None:
            return 0.0
        if not labels:
            return 1.0
        return len(valid_predictions(labels, case)) / len(labels)

    valid_labels.__name__ = "valid_labels"
    return valid_labels


def make_geometric_credit_metric(cases: dict[str, GeometryCase], *, near_contact_credit: float):
    """Create a zero-weight near-miss geometry metric."""

    def geometric_credit(parser, completion, *, info=None, **_kwargs) -> float:
        labels = parse_labels(parser.parse_answer(completion))
        case = resolve_case(info, cases)
        if labels is None or case is None:
            return 0.0

        valid_labels = valid_predictions(labels, case)
        if not valid_labels:
            return 0.0

        total = geometric_credit_sum(valid_labels, case, near_contact_credit)
        return min(total / len(valid_labels), 1.0)

    geometric_credit.__name__ = "geometric_credit"
    return geometric_credit
