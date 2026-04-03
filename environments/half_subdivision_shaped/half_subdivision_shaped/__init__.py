"""Shaped Verifiers environment for half-subdivision neighbour tasks."""

from __future__ import annotations

import ast
import json
import os
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Sequence

import verifiers as vf
from datasets import Dataset
from treelib import Tree

PARSE_BONUS = 0.05
VALID_BONUS = 0.05
NEAR_CONTACT_CREDIT = 0.25
EPS = 1e-9

REPO_ROOT = Path(__file__).resolve().parents[3]
VGB_ROOT = REPO_ROOT / "external" / "VisGeomBench"
VGB_LOCK = RLock()
TASK_SOURCES = {
    "half_subdivision": REPO_ROOT / "data" / "half_subdivision_curriculum.jsonl",
    "half_subdivision_test": REPO_ROOT / "data" / "half_subdivision_test.jsonl",
}


@dataclass(frozen=True)
class Cell:
    """Axis-aligned leaf cell."""

    label: str
    x0: float
    y0: float
    x1: float
    y1: float
    z0: float = 0.0
    z1: float = 1.0


@dataclass(frozen=True)
class GeometryCase:
    """Precomputed geometry for one half-subdivision record."""

    cells: dict[str, Cell]
    target_label: str
    truth_labels: frozenset[str]
    dimension_count: int


@contextmanager
def vgb_runtime():
    if not VGB_ROOT.exists():
        raise RuntimeError(f"Missing VisGeomBench submodule at {VGB_ROOT}")

    with VGB_LOCK:
        previous_cwd = Path.cwd()
        if str(VGB_ROOT) not in sys.path:
            sys.path.insert(0, str(VGB_ROOT))
        try:
            os.chdir(VGB_ROOT)
            yield
        finally:
            os.chdir(previous_cwd)


def load_environment(
    task_name: str = "half_subdivision",
    *,
    system_prompt: str | None = None,
    parse_bonus: float = PARSE_BONUS,
    valid_bonus: float = VALID_BONUS,
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
            parse_bonus=parse_bonus,
            valid_bonus=valid_bonus,
            near_contact_credit=near_contact_credit,
        )
    )
    rubric.add_metric(make_parseable_metric())
    rubric.add_metric(make_valid_labels_metric(cases))
    rubric.add_metric(make_geometric_credit_metric(cases, near_contact_credit=near_contact_credit))

    return vf.SingleTurnEnv(
        dataset=Dataset.from_list(rows),
        rubric=rubric,
        parser=parser,
        system_prompt=system_prompt,
    )


def load_records(task_name: str) -> list[dict[str, Any]]:
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
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, GeometryCase]]:
    """Convert records into a Verifiers dataset and geometry lookup."""

    rows: list[dict[str, Any]] = []
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


def make_parser():
    """Reuse the upstream VGB literal parser."""

    with vgb_runtime():
        from visual_geometry_bench.evaluation.answer_parser import PythonLiteralParser

        return PythonLiteralParser()


def make_shaped_reward(
    cases: dict[str, GeometryCase],
    *,
    parse_bonus: float,
    valid_bonus: float,
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

        valid_labels = valid_predictions(labels, case)
        valid_fraction = len(valid_labels) / len(labels) if labels else 1.0
        geometric_credit = geometric_credit_sum(valid_labels, case, near_contact_credit)
        denominator = max(len(case.truth_labels), len(valid_labels), 1)
        geometric_score = min(geometric_credit / denominator, 1.0)
        score = (
            parse_bonus
            + valid_bonus * valid_fraction
            + (1.0 - parse_bonus - valid_bonus) * geometric_score
        )
        return 1.0 if score > 1.0 - EPS else min(score, 1.0)

    shaped_reward.__name__ = "shaped_reward"
    return shaped_reward


def make_parseable_metric():
    """Create a zero-weight parseability metric."""

    def parseable(parser, completion, *, info=None, **_kwargs) -> float:
        _ = info
        return 1.0 if parse_labels(parser.parse_answer(completion)) is not None else 0.0

    parseable.__name__ = "parseable"
    return parseable


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


def resolve_case(info: Any, cases: dict[str, GeometryCase]) -> GeometryCase | None:
    """Resolve a geometry case from Verifiers rollout info."""

    if isinstance(info, str):
        try:
            info = json.loads(info)
        except json.JSONDecodeError:
            return None
    if isinstance(info, dict):
        token = info.get("record_token")
        if isinstance(token, str):
            return cases.get(token)
    return None


def parse_labels(extracted: str | None) -> list[str] | None:
    """Parse a model answer into a deduplicated list of labels."""

    if extracted is None:
        return None

    text = extracted.strip()
    if not text:
        return []

    parsed = parse_sequence_like(text)
    if parsed is None:
        return None

    labels: list[str] = []
    seen: set[str] = set()
    for item in parsed:
        label = normalize_label(item)
        if label is None or label in seen:
            return None
        seen.add(label)
        labels.append(label)
    return labels


def parse_sequence_like(text: str) -> Sequence[Any] | None:
    """Parse JSON, Python literals, or comma-separated labels."""

    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
        except (json.JSONDecodeError, SyntaxError, ValueError, TypeError):
            continue
        if isinstance(parsed, Sequence) and not isinstance(parsed, (str, bytes)):
            return parsed

    parts = [part.strip() for part in text.replace("\n", ",").split(",") if part.strip()]
    return parts or None


def normalize_label(token: Any) -> str | None:
    """Normalize one label token into canonical string form."""

    if token is None:
        return None
    if isinstance(token, (int, float)) and not isinstance(token, bool):
        if isinstance(token, float) and not token.is_integer():
            return None
        token = str(int(token))
    elif isinstance(token, str):
        token = token.strip()
    else:
        return None

    if token in {"", '""'}:
        return ""
    return token if token and all(ch in "01" for ch in token) else None


def build_geometry_case(record: dict[str, Any]) -> GeometryCase:
    """Rebuild the target cell and all leaves for one record."""

    datagen_args = record["datagen_args"]
    target_label = normalize_label(record["runtime"]["target_label"])
    if target_label is None:
        raise ValueError("Missing target label")

    with vgb_runtime():
        from visual_geometry_bench.datagen.half_subdivision_neighbours import (
            Dimension,
            _build_subdivision,
            _resolve_axis_cycle,
        )

        dim_name = str(datagen_args.get("dimension", "2D")).upper()
        dim = Dimension.D2 if dim_name == "2D" else Dimension.D3
        axis_cycle = _resolve_axis_cycle(
            dim,
            axis_cycle=datagen_args.get("axis_cycle"),
            start_axis=datagen_args.get("start_axis"),
        )
        rng = random.Random(int(datagen_args["seed"]))
        tree = Tree()
        bounds = {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0}
        if dim == Dimension.D3:
            bounds.update({"z0": 0.0, "z1": 1.0})

        leaves = _build_subdivision(
            tree=tree,
            parent_id=None,
            label="",
            depth=0,
            max_depth=int(datagen_args["max_depth"]),
            min_depth=int(datagen_args.get("min_depth", 0)),
            split_prob=float(datagen_args["split_prob"]),
            axis_cycle=axis_cycle,
            dim=dim,
            rng=rng,
            **bounds,
        )

    cells = {
        leaf.label: Cell(
            label=leaf.label,
            x0=leaf.x0,
            y0=leaf.y0,
            x1=leaf.x1,
            y1=leaf.y1,
            z0=leaf.z0,
            z1=leaf.z1,
        )
        for leaf in leaves
    }
    return GeometryCase(
        cells=cells,
        target_label=target_label,
        truth_labels=frozenset(
            label for label in (normalize_label(item) for item in record["ground_truth"]) if label is not None
        ),
        dimension_count=2 if dim_name == "2D" else 3,
    )


def contact_credit(a: Cell, b: Cell, dimension_count: int, near_contact_credit: float) -> float:
    """Score one predicted cell against the target."""

    touch = 0
    overlap = 0
    axes = [
        ((a.x0, a.x1), (b.x0, b.x1)),
        ((a.y0, a.y1), (b.y0, b.y1)),
    ]
    if dimension_count == 3:
        axes.append(((a.z0, a.z1), (b.z0, b.z1)))

    for first, second in axes:
        relation = axis_relation(*first, *second)
        if relation == "separate":
            return 0.0
        if relation == "touch":
            touch += 1
        else:
            overlap += 1

    if touch == 1 and overlap == dimension_count - 1:
        return 1.0
    if touch >= 2 and touch + overlap == dimension_count:
        return near_contact_credit
    return 0.0


def axis_relation(a0: float, a1: float, b0: float, b1: float) -> str:
    """Classify one-dimensional interval relation."""

    if max(a0, b0) < min(a1, b1) - EPS:
        return "overlap"
    if abs(a1 - b0) < EPS or abs(a0 - b1) < EPS:
        return "touch"
    return "separate"


def valid_predictions(labels: list[str], case: GeometryCase) -> list[str]:
    """Keep only valid non-target leaf labels."""

    return [label for label in labels if label in case.cells and label != case.target_label]


def geometric_credit_sum(
    labels: list[str],
    case: GeometryCase,
    near_contact_credit: float,
) -> float:
    """Sum contact credit over predicted labels."""

    return sum(
        contact_credit(
            case.cells[label],
            case.cells[case.target_label],
            case.dimension_count,
            near_contact_credit,
        )
        for label in labels
    )
