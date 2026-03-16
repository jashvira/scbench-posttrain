"""Delaunay helpers for the Inspect pilot."""

from __future__ import annotations

import ast
import base64
import html
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

VGB_ROOT = Path(__file__).resolve().parents[2] / "external" / "VisGeomBench"
if str(VGB_ROOT) not in sys.path:
    sys.path.insert(0, str(VGB_ROOT))

from visual_geometry_bench.datagen import delaunay_tasks as vgb_delaunay_datagen  # noqa: E402
from visual_geometry_bench.verification import delaunay_tasks as vgb_delaunay_verification  # noqa: E402

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
    return vgb_delaunay_datagen.sample_unique_delaunay_points(
        n,
        box=box,
        eps=eps,
        max_tries=max_tries,
        seed=seed,
        rng=rng,
    )


def points_from_datagen_args(datagen_args: dict[str, Any]) -> list[tuple[float, float]]:
    """Generate points from Delaunay datagen args."""
    return vgb_delaunay_datagen._to_points(datagen_args)


def make_prompt(datagen_args: dict[str, Any]) -> str:
    """Build the frozen text prompt for one Delaunay instance."""
    return vgb_delaunay_datagen.make_prompt(datagen_args)


def render_delaunay_prompt_image(datagen_args: dict[str, Any]) -> str:
    """Render a Delaunay prompt image as an SVG data URL for Inspect logs."""

    points = np.array(points_from_datagen_args(datagen_args), dtype=float)
    width = 560
    height = 560
    pad = 44
    plot_size = width - 2 * pad

    def scale_x(x: float) -> float:
        return pad + x * plot_size

    def scale_y(y: float) -> float:
        return height - pad - y * plot_size

    grid_lines = []
    for fraction in np.linspace(0.0, 1.0, 6):
        offset = pad + fraction * plot_size
        grid_lines.append(
            f'<line x1="{offset:.1f}" y1="{pad}" x2="{offset:.1f}" y2="{height - pad}" '
            'stroke="#D7D7D7" stroke-width="1" stroke-dasharray="4 6" />'
        )
        grid_lines.append(
            f'<line x1="{pad}" y1="{offset:.1f}" x2="{width - pad}" y2="{offset:.1f}" '
            'stroke="#D7D7D7" stroke-width="1" stroke-dasharray="4 6" />'
        )

    points_svg = []
    labels_svg = []
    for index, (x, y) in enumerate(points):
        cx = scale_x(float(x))
        cy = scale_y(float(y))
        points_svg.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="8" fill="#2A7FFF" stroke="white" stroke-width="2" />'
        )
        labels_svg.append(
            f'<text x="{cx + 12:.2f}" y="{cy - 12:.2f}" font-size="14" font-weight="700" '
            f'font-family="Helvetica, Arial, sans-serif" fill="#111111">{html.escape(str(index))}</text>'
        )

    svg = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#FFFFFF" />',
            f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-size="24" font-weight="700" font-family="Helvetica, Arial, sans-serif" fill="#111111">Delaunay Task</text>',
            f'<rect x="{pad}" y="{pad}" width="{plot_size}" height="{plot_size}" fill="#FAFAFA" stroke="#BDBDBD" stroke-width="1.5" />',
            *grid_lines,
            *points_svg,
            *labels_svg,
            "</svg>",
        ]
    )
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def get_solutions(datagen_args: dict[str, Any]) -> list[list[int]]:
    """Return the canonical ground-truth Delaunay triangulation."""
    return vgb_delaunay_datagen.get_solutions(datagen_args)


def generate_dataset_record(
    datagen_args: dict[str, Any],
    *,
    tags: list[str] | None = None,
    difficulty: str = "",
    record_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a JSON-serialisable Delaunay dataset record."""
    return vgb_delaunay_datagen.generate_dataset_record(
        datagen_args,
        tags=tags,
        difficulty=difficulty,
        record_id=record_id,
        metadata=metadata,
    )


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


def _simple_sequence_candidates(text: str) -> list[str]:
    """Generate candidate substrings likely containing a scalar list."""

    candidates: list[str] = []
    seen: set[str] = set()

    def add(candidate: str) -> None:
        candidate = candidate.strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)

    add(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        add(lines[-1])
    for line in reversed(lines):
        for separator in ("->", "=>", ":", "="):
            if separator in line:
                add(line.rsplit(separator, 1)[-1])
                break
    for separator in ("->", "=>", ":", "="):
        if separator in text:
            add(text.rsplit(separator, 1)[-1])
    return candidates


def _normalise_csv_token(token: str) -> str:
    """Coerce a simple token into a Python literal element."""

    if token in {'""', "''"}:
        return '""'
    if (token.startswith('"') and token.endswith('"') and len(token) >= 2) or (
        token.startswith("'") and token.endswith("'") and len(token) >= 2
    ):
        return token

    try:
        value = ast.literal_eval(token)
    except Exception:
        value = None

    if isinstance(value, (int, float)):
        stripped = token.lstrip("-+")
        if stripped.startswith("0") and len(stripped) > 1:
            return repr(token)
        return token

    return repr(token)


def _coerce_simple_sequence(text: str, *, min_tokens: int) -> str | None:
    """Turn a plain scalar sequence into a list literal when possible."""

    if any(bracket in text for bracket in "[]{}()"):
        return None

    candidate = text.replace("\n", ",")
    tokens = [token.strip() for token in candidate.split(",") if token.strip()]
    if len(tokens) < min_tokens:
        return None

    allowed = re.compile(r"^[A-Za-z0-9_.\-\"']+$")
    if not all(allowed.fullmatch(token) for token in tokens):
        return None

    try:
        normalised = [_normalise_csv_token(token) for token in tokens]
    except ValueError:
        return None

    literal = "[" + ", ".join(normalised) + "]"
    return literal if _is_valid_literal(literal) else None


def _parse_simple_sequence(text: str, *, min_tokens: int) -> str | None:
    """Convert comma/newline scalar sequences into Python list literals."""

    if not text:
        return None

    candidates = _simple_sequence_candidates(text)
    ordered = sorted(enumerate(candidates), key=lambda item: (-len(item[1]), item[0]))
    for _, candidate in ordered:
        literal = _coerce_simple_sequence(candidate, min_tokens=min_tokens)
        if literal is not None:
            return literal
    return None


def extract_python_literal(text: str) -> str | None:
    """Extract Python literals using the upstream VisGeomBench parser logic."""

    text = _normalise_text(text)
    text = _strip_thinking_blocks(text)
    if not text:
        return None

    code_fence = _extract_from_code_fence(text)
    if code_fence and _is_valid_literal(code_fence):
        return code_fence.strip()

    stripped = text.strip()
    if _is_valid_literal(stripped):
        return stripped

    simple_list = _parse_simple_sequence(stripped, min_tokens=2)
    if simple_list is not None:
        return simple_list

    backscanned = _backscan_literal(text)
    if backscanned and _is_valid_literal(backscanned):
        return backscanned.strip()

    simple_list = _parse_simple_sequence(stripped, min_tokens=1)
    if simple_list is not None:
        return simple_list

    return None

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

    verification = vgb_delaunay_verification.verify_delaunay_triangulation(
        literal,
        validated_record.model_dump(mode="json"),
        return_diff=True,
    )
    errors = verification["errors"]
    if errors:
        return DelaunayEvaluation(
            score=0.0,
            passed=False,
            parsed_ok=True,
            error_type=errors[0],
            extracted_literal=literal,
            missing=[],
            extra=[],
        )

    passed = bool(verification["passed"])
    missing = verification["missing"]
    extra = verification["extra"]
    return DelaunayEvaluation(
        score=1.0 if passed else 0.0,
        passed=passed,
        parsed_ok=True,
        error_type=None if passed else "exact_mismatch",
        extracted_literal=literal,
        missing=missing,
        extra=extra,
    )
