"""Geometry helpers for half-subdivision shaped rewards."""

from __future__ import annotations

import json
import os
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from treelib import Tree

from .parser import normalize_label

EPS = 1e-9
REPO_ROOT = Path(__file__).resolve().parents[3]
VGB_ROOT = REPO_ROOT / "external" / "VisGeomBench"
VGB_LOCK = RLock()


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
    """Expose the upstream VGB package in-process."""

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


def build_geometry_case(record: dict) -> GeometryCase:
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

    return GeometryCase(
        cells={
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
        },
        target_label=target_label,
        truth_labels=frozenset(
            label
            for label in (normalize_label(item) for item in record["ground_truth"])
            if label is not None
        ),
        dimension_count=2 if str(datagen_args.get("dimension", "2D")).upper() == "2D" else 3,
    )


def resolve_case(info, cases: dict[str, GeometryCase]) -> GeometryCase | None:
    """Resolve a geometry case from rollout info."""

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


def shaped_score(labels: list[str], case: GeometryCase, near_contact_credit: float) -> float:
    """Compute the normalized shaped score for one parsed prediction."""

    valid_labels = valid_predictions(labels, case)
    denominator = max(len(case.truth_labels), 1)
    total_credit = geometric_credit_sum(valid_labels, case, near_contact_credit)
    return min(total_credit / denominator, 1.0)


def exact_match(labels: list[str], case: GeometryCase) -> float:
    """Return 1.0 when the predicted neighbour set matches exactly."""

    return 1.0 if frozenset(valid_predictions(labels, case)) == case.truth_labels else 0.0


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
