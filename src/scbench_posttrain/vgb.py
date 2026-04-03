from __future__ import annotations

import ast
import base64
import io
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import transcript

REPO_ROOT = Path(__file__).resolve().parents[2]
VGB_ROOT = REPO_ROOT / "external" / "VisGeomBench"
VGB_LOCK = RLock()
VGB_TASKS: dict[str, tuple[str, str]] = {
    "topology_enumeration": ("topology_enumeration_curated.toml", "Topology Enumeration"),
    "enumerate_edges": ("topology_edge_enumerate_curated.toml", "Enumerate Edges"),
    "classify_behaviour": ("topology_edge_classify_curated.toml", "Classify Behaviour"),
    "convex_hull": ("convex_hull_curated.toml", "Convex Hull Ordering"),
    "delaunay": ("delaunay_dataset.toml", "Delaunay Triangulation"),
    "two_segments": ("two_segments_curated.toml", "Two Segments"),
    "shikaku": ("shikaku_curated.toml", "Shikaku Rectangles"),
    "half_subdivision": ("data/half_subdivision_curriculum.jsonl", "Half Subdivision Neighbours"),
    "half_subdivision_test": ("data/half_subdivision_test.jsonl", "Half Subdivision Test Slice"),
}


@dataclass(frozen=True)
class VGBTask:
    name: str
    title: str
    dataset: MemoryDataset
    records: tuple[dict[str, Any], ...]


@contextmanager
def vgb_runtime():
    if not VGB_ROOT.exists():
        raise RuntimeError(
            f"Missing VisGeomBench submodule at {VGB_ROOT}. Initialize it before running VGB evals."
        )

    with VGB_LOCK:
        previous_cwd = Path.cwd()
        if str(VGB_ROOT) not in sys.path:
            sys.path.insert(0, str(VGB_ROOT))
        try:
            os.chdir(VGB_ROOT)
            yield
        finally:
            os.chdir(previous_cwd)


def _build_dataset(name: str, title: str, records: list[dict[str, Any]]) -> MemoryDataset:
    samples = [
        Sample(
            input=record["prompt"],
            id=str(record.get("id", f"{name}:{index}")),
            metadata={
                "name": name,
                "title": title,
                "record_id": str(record.get("id", f"{name}:{index}")),
                "record_index": index,
                "source_record_index": index,
                "problem_type": str(record.get("metadata", {}).get("problem_type", "")),
            },
        )
        for index, record in enumerate(records)
    ]
    return MemoryDataset(samples=samples, name=f"vgb_{name}")


def _resolve_task_source(source_name: str) -> Path:
    local_path = REPO_ROOT / source_name
    if local_path.exists():
        return local_path
    return VGB_ROOT / "configs" / source_name


def _load_records(source_name: str) -> list[dict[str, Any]]:
    source_path = _resolve_task_source(source_name)
    if source_path.suffix == ".jsonl":
        with source_path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    with vgb_runtime():
        from visual_geometry_bench.dataset import build_records_from_config, load_config

        config = load_config(source_path)
        return build_records_from_config(config)


@lru_cache(maxsize=None)
def _load_vgb_task_cached(name: str) -> VGBTask:
    try:
        source_name, title = VGB_TASKS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown VGB task {name!r}. Available tasks: {', '.join(sorted(VGB_TASKS))}"
        ) from exc

    records = _load_records(source_name)
    return VGBTask(
        name=name,
        title=title,
        dataset=_build_dataset(name, title, records),
        records=tuple(records),
    )


def load_vgb_task(name: str) -> VGBTask:
    return _load_vgb_task_cached(name)


load_vgb_task.cache_clear = _load_vgb_task_cached.cache_clear


def slice_vgb_task(task: VGBTask, indices: list[int]) -> VGBTask:
    source_count = len(task.records)
    if not indices:
        raise ValueError("record_indices must not be empty")

    selected_records: list[dict[str, Any]] = []
    samples: list[Sample] = []
    for local_index, source_index in enumerate(indices):
        if source_index < 0 or source_index >= source_count:
            raise ValueError(
                f"record index {source_index} out of range for {task.name!r} (0..{source_count - 1})"
            )
        record = task.records[source_index]
        selected_records.append(record)
        samples.append(
            Sample(
                input=record["prompt"],
                id=str(record.get("id", f"{task.name}:{source_index}")),
                metadata={
                    "name": task.name,
                    "title": task.title,
                    "record_id": str(record.get("id", f"{task.name}:{source_index}")),
                    "record_index": local_index,
                    "source_record_index": source_index,
                    "problem_type": str(record.get("metadata", {}).get("problem_type", "")),
                },
            )
        )

    return VGBTask(
        name=task.name,
        title=task.title,
        dataset=MemoryDataset(samples=samples, name=f"vgb_{task.name}"),
        records=tuple(selected_records),
    )


def extract_vgb_answer(completion: str) -> str | None:
    with vgb_runtime():
        from visual_geometry_bench.evaluation.answer_parser import PythonLiteralParser

        return PythonLiteralParser().parse_answer(completion)


def parse_vgb_answer(literal: str | None) -> Any | None:
    if literal is None:
        return None
    try:
        return json.loads(literal)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(literal)
        except (SyntaxError, ValueError):
            return None


def grade_vgb_answer(record: dict[str, Any], completion: str) -> tuple[float, dict[str, Any], Any | None]:
    extracted = extract_vgb_answer(completion)
    parsed_answer = parse_vgb_answer(extracted)

    if extracted is None:
        return 0.0, {"grading_text": "parse_failure", "errors": ["parse_failure"]}, None

    problem_type = record.get("metadata", {}).get("problem_type", "")
    with vgb_runtime():
        from visual_geometry_bench.registry import get_verifier

        verifier = get_verifier(problem_type)

    diagnostics = verifier(extracted, record, return_diff=True)
    if not isinstance(diagnostics, dict):
        passed = bool(diagnostics)
        diagnostics = {"passed": passed, "errors": [], "missing": [], "extra": []}

    metadata: dict[str, Any] = {
        "grading_text": _format_grading_text(diagnostics),
    }
    for key in ("errors", "missing", "extra", "details"):
        value = diagnostics.get(key)
        if value not in (None, [], {}, ""):
            metadata[key] = value

    return (1.0 if diagnostics.get("passed") else 0.0), metadata, parsed_answer


def log_prompt_artifacts(record: dict[str, Any]) -> None:
    markdown = _render_record_markdown(
        record,
        answer=None,
        mode="ground_truth",
        heading="Task visual",
        alt_text="Task visual",
    )
    if markdown is not None:
        transcript().info(markdown, source="solver")


def log_score_artifacts(record: dict[str, Any], answer: Any | None) -> None:
    markdown = _render_record_markdown(
        record,
        answer=answer,
        mode="both",
        heading="Ground truth vs model answer",
        alt_text="Ground truth vs model answer",
        answer_label="Model answer",
    )
    if markdown is not None:
        transcript().info(markdown, source="scorer")


def _render_record_markdown(
    record: dict[str, Any],
    *,
    answer: Any | None,
    mode: str,
    heading: str,
    alt_text: str,
    answer_label: str | None = None,
) -> str | None:
    with vgb_runtime():
        from visualisations import visualise_record
        import matplotlib.pyplot as plt

        try:
            result = visualise_record(
                record,
                answer=answer,
                mode=mode,
                answer_label=answer_label,
                show=False,
            )
        except NotImplementedError:
            return None

        if isinstance(result, dict):
            figures = result
        else:
            figures = {"main": result}

        sections = [f"### {heading}"]
        for key, figure in figures.items():
            if figure is None:
                continue
            image = _figure_to_data_url(figure)
            plt.close(figure)
            label = alt_text if key == "main" else f"{alt_text} ({key})"
            sections.append(f"![{label}]({image})")
        if len(sections) == 1:
            return None
        return "\n\n".join(sections)


def _figure_to_data_url(figure: Any) -> str:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _format_grading_text(diagnostics: dict[str, Any]) -> str:
    if diagnostics.get("passed"):
        return "verified"

    errors = diagnostics.get("errors")
    if errors:
        return ", ".join(str(error) for error in errors)

    missing = diagnostics.get("missing") or []
    extra = diagnostics.get("extra") or []
    parts: list[str] = []
    if missing:
        parts.append(f"missing={missing}")
    if extra:
        parts.append(f"extra={extra}")
    return "; ".join(parts) if parts else "mismatch"
