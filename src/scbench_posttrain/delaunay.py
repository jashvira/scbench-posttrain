"""Minimal Delaunay task logic extracted from VisGeomBench.

This module intentionally copies only the Delaunay-specific generation,
literal extraction, and exact verification behavior needed by the Inspect
pilot. The upstream `VisGeomBench` submodule remains a source reference, not a
runtime dependency.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Literal, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.spatial import Delaunay as SciPyDelaunay

EPSILON = 1e-12


class DelaunayRecordMetadata(BaseModel):
    """Typed record metadata for frozen dataset rows."""

    model_config = ConfigDict(frozen=True)

    problem_type: Literal["delaunay_triangulation"] = "delaunay_triangulation"
    tags: list[str]
    difficulty: str = ""


class DelaunayRecord(BaseModel):
    """Typed dataset record aligned with the VisGeomBench JSONL shape."""

    model_config = ConfigDict(frozen=True)

    id: str
    prompt: str
    ground_truth: list[list[int]]
    metadata: DelaunayRecordMetadata
    datagen_args: dict[str, Any]


class DelaunaySampleMetadata(BaseModel):
    """Typed metadata stored on Inspect samples."""

    model_config = ConfigDict(frozen=True)

    record_id: str
    problem_type: Literal["delaunay_triangulation"] = "delaunay_triangulation"
    difficulty: str = ""
    tags: list[str]
    ground_truth: list[list[int]]
    datagen_args: dict[str, Any]


@dataclass(frozen=True)
class DelaunayEvaluation:
    """Structured verification result for one model completion."""

    score: float
    passed: bool
    parsed_ok: bool
    error_type: str | None
    extracted_literal: str | None
    missing: list[list[int]]
    extra: list[list[int]]


def compute_content_hash(
    *,
    problem_type: str,
    datagen_args: dict[str, Any],
    prompt: str,
    ground_truth: list[list[int]],
    prefix_len: int = 8,
) -> str:
    """Compute a deterministic short content hash for a record."""

    payload = {
        "problem_type": problem_type,
        "datagen_args": datagen_args,
        "prompt": prompt,
        "ground_truth": ground_truth,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()[:prefix_len]


def validate_point_array(
    points: Sequence[Sequence[float]], *, min_points: int = 0
) -> list[tuple[float, float]]:
    """Validate 2D point arrays and coerce to float tuples."""

    if len(points) < min_points:
        raise AssertionError(f"requires at least {min_points} points; got {len(points)}")

    coerced: list[tuple[float, float]] = []
    for index, point in enumerate(points):
        if len(point) != 2:
            raise ValueError(f"point {index} is not length 2: {point!r}")

        x = float(point[0])
        y = float(point[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"point {index} has non-finite coordinates: {(x, y)!r}")
        coerced.append((x, y))

    return coerced


def _orientation_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return twice the signed area of triangle ABC."""

    u = b - a
    v = c - a
    return float(u[0] * v[1] - u[1] * v[0])


def _incircle_det(points: np.ndarray) -> float:
    """Return the 4x4 incircle determinant for four points."""

    x = points[:, 0]
    y = points[:, 1]
    matrix = np.column_stack((x, y, x * x + y * y, np.ones(4)))
    return float(np.linalg.det(matrix))


def sample_unique_delaunay_points(
    n: int,
    *,
    box: tuple[float, float] = (0.0, 1.0),
    eps: float = EPSILON,
    max_tries: int = 10_000,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample points in general position with a unique Delaunay triangulation."""

    if rng is not None and seed is not None:
        raise ValueError("Specify either seed or rng, not both")
    if n < 3:
        raise ValueError("Delaunay triangulation requires at least 3 points")

    if rng is None:
        rng = np.random.default_rng(seed)

    lo, hi = box
    scale = hi - lo
    tolerance = eps * scale * scale

    for _ in range(max_tries):
        points = rng.uniform(lo, hi, size=(n, 2))

        if any(
            abs(_orientation_area(points[i], points[j], points[k])) <= tolerance
            for i, j, k in combinations(range(n), 3)
        ):
            continue

        if any(
            abs(_incircle_det(points[np.array(indices, dtype=int)])) <= tolerance
            for indices in combinations(range(n), 4)
        ):
            continue

        return points

    raise RuntimeError(f"Failed to find general-position sample after {max_tries} attempts")


def points_from_datagen_args(datagen_args: dict[str, Any]) -> list[tuple[float, float]]:
    """Generate points from Delaunay datagen args."""

    try:
        num_points = int(datagen_args["num_points"])
        seed = int(datagen_args["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("datagen_args must contain 'num_points' and 'seed'") from exc

    if num_points < 3:
        raise ValueError("num_points must be at least 3")

    box = datagen_args.get("box", (0.0, 1.0))
    try:
        lo, hi = map(float, box)
    except (TypeError, ValueError) as exc:
        raise ValueError("box must be an iterable of two numeric bounds") from exc
    if hi <= lo:
        raise ValueError("box upper bound must exceed lower bound")

    points = sample_unique_delaunay_points(
        num_points,
        seed=seed,
        box=(lo, hi),
        eps=float(datagen_args.get("eps", EPSILON)),
        max_tries=int(datagen_args.get("max_tries", 10_000)),
    )
    return validate_point_array(points.tolist(), min_points=3)


def compute_delaunay_triangulation(points: Sequence[Sequence[float]]) -> list[list[int]]:
    """Compute the canonical Delaunay triangulation for a point set."""

    triangulation = SciPyDelaunay(np.array(points, dtype=float))
    simplices = [sorted(map(int, simplex)) for simplex in triangulation.simplices]
    return sorted(simplices)


def make_prompt(datagen_args: dict[str, Any]) -> str:
    """Build the frozen text prompt for one Delaunay instance."""

    points = points_from_datagen_args(datagen_args)
    display_points = np.round(points, 3)
    points_text = ",\n".join(f"  {list(map(float, point))}" for point in display_points)
    return "\n".join(
        [
            "You are given a set of 2D points in general position (indices correspond to the order shown):",
            "[",
            points_text,
            "]",
            "",
            "Return the Delaunay triangulation as a list of triangles.",
            "Each triangle is a list of three point indices (sorted in ascending order).",
            "Strict output: return only a Python list of lists of integers.",
        ]
    )


def get_solutions(datagen_args: dict[str, Any]) -> list[list[int]]:
    """Return the canonical ground-truth Delaunay triangulation."""

    return compute_delaunay_triangulation(points_from_datagen_args(datagen_args))


def generate_dataset_record(
    datagen_args: dict[str, Any],
    *,
    tags: list[str] | None = None,
    difficulty: str = "",
    record_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a JSON-serialisable Delaunay dataset record."""

    prompt = make_prompt(datagen_args)
    ground_truth = get_solutions(datagen_args)

    merged_tags = ["geometry", "triangulation", "delaunay"]
    for tag in tags or []:
        if tag not in merged_tags:
            merged_tags.append(tag)

    record_metadata: dict[str, Any] = {
        "problem_type": "delaunay_triangulation",
        "tags": merged_tags,
        "difficulty": difficulty,
    }
    if metadata:
        for key, value in metadata.items():
            if key != "requires_visual":
                record_metadata[key] = value

    content_id = record_id or compute_content_hash(
        problem_type="delaunay_triangulation",
        datagen_args=datagen_args,
        prompt=prompt,
        ground_truth=ground_truth,
    )

    return {
        "id": content_id,
        "prompt": prompt,
        "ground_truth": ground_truth,
        "metadata": record_metadata,
        "datagen_args": dict(datagen_args),
    }


def _normalise_text(text: str) -> str:
    """Normalize smart quotes to ASCII quotes before parsing."""

    translation_table = str.maketrans(
        {
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
        }
    )
    return text.translate(translation_table)


def _strip_thinking_blocks(text: str) -> str:
    """Remove `<thinking>` blocks before literal extraction."""

    return re.sub(
        re.compile(r"<thinking>.*?</thinking>", re.IGNORECASE | re.DOTALL),
        "",
        text,
    )


def _extract_from_code_fence(text: str) -> str | None:
    """Return the last fenced code block, if present."""

    matches = re.findall(r"```(?:[^\n`]*\n)?(.*?)```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def _backscan_literal(text: str) -> str | None:
    """Backscan for the last balanced list or dict literal."""

    last_bracket = -1
    closing_bracket = ""
    for index in range(len(text) - 1, -1, -1):
        if text[index] in ("]", "}"):
            last_bracket = index
            closing_bracket = text[index]
            break

    if last_bracket == -1:
        return None

    opening_bracket = "[" if closing_bracket == "]" else "{"
    bracket_count = 1
    for index in range(last_bracket - 1, -1, -1):
        if text[index] == closing_bracket:
            bracket_count += 1
        elif text[index] == opening_bracket:
            bracket_count -= 1
            if bracket_count == 0:
                return text[index : last_bracket + 1]
    return None


def _is_valid_literal(text: str) -> bool:
    """Return whether `ast.literal_eval` accepts the text."""

    try:
        ast.literal_eval(text)
        return True
    except (ValueError, SyntaxError):
        return False


def extract_python_literal(text: str) -> str | None:
    """Extract the final Python/JSON literal from a verbose model response."""

    cleaned = _strip_thinking_blocks(_normalise_text(text)).strip()
    if not cleaned:
        return None

    fenced = _extract_from_code_fence(cleaned)
    if fenced and _is_valid_literal(fenced):
        return fenced.strip()

    if _is_valid_literal(cleaned):
        return cleaned

    backscanned = _backscan_literal(cleaned)
    if backscanned and _is_valid_literal(backscanned):
        return backscanned.strip()

    return None


def _parse_literal(text: str) -> Any:
    """Parse a literal as JSON first, then as a Python literal."""

    try:
        return json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return ast.literal_eval(text)


def _validate_triangulation_format(triangles: Any) -> tuple[bool, str | None]:
    """Validate that a triangulation is a list of non-negative integer triples."""

    if not isinstance(triangles, list):
        return False, "not_a_list"

    for index, triangle in enumerate(triangles):
        if not isinstance(triangle, list):
            return False, f"triangle_{index}_not_list"
        if len(triangle) != 3:
            return False, f"triangle_{index}_wrong_length"
        if not all(isinstance(point_index, int) and point_index >= 0 for point_index in triangle):
            return False, f"triangle_{index}_invalid_indices"

    return True, None


def canonicalise_triangulation(triangles: Sequence[Sequence[int]]) -> list[list[int]]:
    """Sort triangle indices and sort the list lexicographically."""

    return sorted([sorted(map(int, triangle)) for triangle in triangles])


def verify_delaunay_answer(
    answer: Sequence[Sequence[int]],
    ground_truth: Sequence[Sequence[int]],
) -> tuple[bool, list[list[int]], list[list[int]]]:
    """Verify a parsed triangulation against exact ground truth."""

    predicted = canonicalise_triangulation(answer)
    expected = canonicalise_triangulation(ground_truth)

    predicted_counter = Counter(map(tuple, predicted))
    expected_counter = Counter(map(tuple, expected))

    missing = sorted(
        [list(triangle) for triangle, count in (expected_counter - predicted_counter).items() for _ in range(count)]
    )
    extra = sorted(
        [list(triangle) for triangle, count in (predicted_counter - expected_counter).items() for _ in range(count)]
    )
    return (not missing and not extra), missing, extra


def score_delaunay_answer(raw_output: str, record: DelaunayRecord | dict[str, Any]) -> DelaunayEvaluation:
    """Parse, validate, and exactly score a Delaunay model response."""

    validated_record = (
        record if isinstance(record, DelaunayRecord) else DelaunayRecord.model_validate(record)
    )
    literal = extract_python_literal(raw_output)
    if literal is None:
        return DelaunayEvaluation(
            score=0.0,
            passed=False,
            parsed_ok=False,
            error_type="parse_failure",
            extracted_literal=None,
            missing=[],
            extra=[],
        )

    try:
        parsed = _parse_literal(literal)
    except (ValueError, SyntaxError, json.JSONDecodeError):
        return DelaunayEvaluation(
            score=0.0,
            passed=False,
            parsed_ok=False,
            error_type="parse_failure",
            extracted_literal=literal,
            missing=[],
            extra=[],
        )

    format_ok, error_type = _validate_triangulation_format(parsed)
    if not format_ok:
        return DelaunayEvaluation(
            score=0.0,
            passed=False,
            parsed_ok=True,
            error_type=error_type,
            extracted_literal=literal,
            missing=[],
            extra=[],
        )

    passed, missing, extra = verify_delaunay_answer(parsed, validated_record.ground_truth)
    return DelaunayEvaluation(
        score=1.0 if passed else 0.0,
        passed=passed,
        parsed_ok=True,
        error_type=None if passed else "exact_mismatch",
        extracted_literal=literal,
        missing=missing,
        extra=extra,
    )
