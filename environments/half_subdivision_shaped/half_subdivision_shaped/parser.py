"""Parsing helpers for half-subdivision answers."""

from __future__ import annotations

import ast
import json
from typing import Any, Sequence


def make_parser():
    """Reuse the upstream VGB literal parser."""

    from .geometry import vgb_runtime

    with vgb_runtime():
        from visual_geometry_bench.evaluation.answer_parser import PythonLiteralParser

        return PythonLiteralParser()


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
