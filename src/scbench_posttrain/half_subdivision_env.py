from __future__ import annotations

import ast
import json
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

import verifiers as vf
from datasets import Dataset
from treelib import Tree

from scbench_posttrain.vgb import VGBTask, load_vgb_task, vgb_runtime

PARSE_BONUS = 0.05
VALID_BONUS = 0.05
NEAR_CONTACT_CREDIT = 0.25
_EPS = 1e-9


@dataclass(frozen=True)
class Cell:
    label: str
    x0: float
    y0: float
    x1: float
    y1: float
    z0: float = 0.0
    z1: float = 1.0


def load_half_subdivision_shaped_env(
    *,
    task_name: str = "half_subdivision",
    system_prompt: str | None = None,
    parse_bonus: float = PARSE_BONUS,
    valid_bonus: float = VALID_BONUS,
    near_contact_credit: float = NEAR_CONTACT_CREDIT,
):
    task = load_vgb_task(task_name)
    if any(
        record.get("metadata", {}).get("problem_type") != "half_subdivision_neighbours"
        for record in task.records
    ):
        raise ValueError(f"{task_name!r} is not a half_subdivision task")

    dataset_rows, record_lookup = _format_records(task)

    # Keep the benchmark verifier separate; this env only adds a shaped reward for training.
    parser = _make_parser()
    rubric = vf.Rubric(
        funcs=[
            _make_reward_func(
                record_lookup,
                parse_bonus=parse_bonus,
                valid_bonus=valid_bonus,
                near_contact_credit=near_contact_credit,
            )
        ],
        parser=parser,
    )

    return vf.SingleTurnEnv(
        dataset=Dataset.from_list(dataset_rows),
        rubric=rubric,
        parser=parser,
        system_prompt=system_prompt,
    )


def _format_records(task: VGBTask) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    lookup: dict[str, dict[str, Any]] = {}

    for index, record in enumerate(task.records):
        token = str(record.get("id", index))
        rows.append(
            {
                "id": token,
                "prompt": [{"role": "user", "content": record["prompt"]}],
                "answer": json.dumps(record.get("ground_truth", [])),
                "info": json.dumps({"record_token": token}),
            }
        )
        lookup[token] = record

    return rows, lookup


def _make_parser():
    with vgb_runtime():
        from visual_geometry_bench.evaluation.answer_parser import PythonLiteralParser

        return PythonLiteralParser()


def _make_reward_func(
    record_lookup: dict[str, dict[str, Any]],
    *,
    parse_bonus: float,
    valid_bonus: float,
    near_contact_credit: float,
):
    def reward_func(parser, completion, answer, *, info=None, **_kwargs):
        extracted = parser.parse_answer(completion)
        record = _resolve_record(info, record_lookup)
        return score_half_subdivision_completion(
            extracted,
            record,
            parse_bonus=parse_bonus,
            valid_bonus=valid_bonus,
            near_contact_credit=near_contact_credit,
        )

    return reward_func


def _resolve_record(info: Any, record_lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if isinstance(info, str):
        try:
            info = json.loads(info)
        except json.JSONDecodeError:
            return {}
    if isinstance(info, dict):
        token = info.get("record_token")
        if isinstance(token, str):
            return dict(record_lookup.get(token, {}))
    return {}


def score_half_subdivision_completion(
    extracted: str | None,
    record: dict[str, Any],
    *,
    parse_bonus: float = PARSE_BONUS,
    valid_bonus: float = VALID_BONUS,
    near_contact_credit: float = NEAR_CONTACT_CREDIT,
) -> float:
    labels = _parse_labels(extracted)
    if labels is None:
        return 0.0

    cells, target, truth, dimension_count = _load_geometry(record)
    if not cells or target is None:
        return 0.0

    valid_labels = [label for label in labels if label in cells]
    valid_fraction = len(valid_labels) / len(labels) if labels else 1.0

    core_denominator = max(len(truth), len(valid_labels), 1)
    geom_credit = sum(
        _contact_credit(cells[label], target, dimension_count, near_contact_credit)
        for label in valid_labels
        if label != target.label
    )
    core_score = min(geom_credit / core_denominator, 1.0)
    score = parse_bonus + valid_bonus * valid_fraction + (1.0 - parse_bonus - valid_bonus) * core_score
    return 1.0 if score > 1.0 - _EPS else min(score, 1.0)


def _parse_labels(extracted: str | None) -> list[str] | None:
    if extracted is None:
        return None

    text = extracted.strip()
    if not text:
        return []

    parsed = _parse_sequence_like(text)
    if parsed is None:
        return None

    labels: list[str] = []
    seen: set[str] = set()
    for item in parsed:
        label = _normalize_label(item)
        if label is None or label in seen:
            return None
        seen.add(label)
        labels.append(label)
    return labels


def _parse_sequence_like(text: str) -> Sequence[Any] | None:
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
        except (json.JSONDecodeError, SyntaxError, ValueError, TypeError):
            continue
        if isinstance(parsed, Sequence) and not isinstance(parsed, (str, bytes)):
            return parsed

    parts = [part.strip() for part in text.replace("\n", ",").split(",") if part.strip()]
    return parts or None


def _normalize_label(token: Any) -> str | None:
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


@lru_cache(maxsize=None)
def _cached_geometry(record_key: str) -> tuple[dict[str, Cell], Cell | None, frozenset[str], int]:
    payload = json.loads(record_key)
    datagen_args = payload["datagen_args"]
    target_label = _normalize_label(payload["target_label"])
    truth = frozenset(
        label for label in (_normalize_label(item) for item in payload["ground_truth"]) if label is not None
    )

    with vgb_runtime():
        from visual_geometry_bench.datagen.half_subdivision_neighbours import (
            Dimension,
            _build_subdivision,
            _resolve_axis_cycle,
        )

        dim_name = str(datagen_args.get("dimension", "2D")).upper()
        dim = Dimension.D2 if dim_name == "2D" else Dimension.D3
        dimension_count = 2 if dim == Dimension.D2 else 3
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
    return cells, cells.get(target_label), truth, dimension_count


def _load_geometry(record: dict[str, Any]) -> tuple[dict[str, Cell], Cell | None, frozenset[str], int]:
    record_key = json.dumps(
        {
            "datagen_args": record.get("datagen_args", {}),
            "target_label": record.get("runtime", {}).get("target_label", ""),
            "ground_truth": record.get("ground_truth", []),
        },
        sort_keys=True,
    )
    return _cached_geometry(record_key)


def _contact_credit(a: Cell, b: Cell, dimension_count: int, near_contact_credit: float) -> float:
    touch = 0
    overlap = 0

    for first, second in (
        ((a.x0, a.x1), (b.x0, b.x1)),
        ((a.y0, a.y1), (b.y0, b.y1)),
        *([((a.z0, a.z1), (b.z0, b.z1))] if dimension_count == 3 else []),
    ):
        relation = _axis_relation(*first, *second)
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


def _axis_relation(a0: float, a1: float, b0: float, b1: float) -> str:
    if max(a0, b0) < min(a1, b1) - _EPS:
        return "overlap"
    if abs(a1 - b0) < _EPS or abs(a0 - b1) < _EPS:
        return "touch"
    return "separate"
