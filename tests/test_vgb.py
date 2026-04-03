from __future__ import annotations

import ast
import asyncio
import re
import subprocess
from pathlib import Path

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser, ModelOutput
from inspect_ai.solver import TaskState

from evals.vgb import vgb_score, vgb_task
from scbench_posttrain.vgb import VGBTask, log_prompt_artifacts

ROOT = Path(__file__).resolve().parents[1]


class FakeTranscript:
    def __init__(self) -> None:
        self.events: list[tuple[str, str | None]] = []

    def info(self, data, *, source=None) -> None:
        self.events.append((data, source))


def _make_state(*, sample, completion: str) -> TaskState:
    return TaskState(
        model="openai/test-model",
        sample_id=sample.id,
        epoch=1,
        input=sample.input,
        messages=[ChatMessageUser(content=sample.input)],
        output=ModelOutput.from_content(model="openai/test-model", content=completion),
        completed=bool(completion),
        metadata=sample.metadata,
        store={},
    )


def test_vgb_task_builds_memory_dataset():
    inspect_task = vgb_task("topology_enumeration")

    assert len(inspect_task.dataset) == 10
    assert inspect_task.name == "vgb_topology_enumeration"
    assert inspect_task.display_name == "Topology Enumeration"
    assert inspect_task.dataset[0].metadata == {
        "name": "topology_enumeration",
        "title": "Topology Enumeration",
        "record_id": inspect_task.dataset[0].id,
        "record_index": 0,
        "source_record_index": 0,
        "problem_type": "topology_enumeration",
    }


def test_vgb_task_can_slice_explicit_record_indices():
    inspect_task = vgb_task("half_subdivision", record_indices="0,47,192")

    assert len(inspect_task.dataset) == 3
    assert [sample.metadata["source_record_index"] for sample in inspect_task.dataset] == [0, 47, 192]
    assert [sample.metadata["record_index"] for sample in inspect_task.dataset] == [0, 1, 2]


def test_vgb_task_accepts_keyword_task_name():
    inspect_task = vgb_task(task_name="half_subdivision_test")

    assert len(inspect_task.dataset) == 10
    assert inspect_task.name == "vgb_half_subdivision_test"


def test_delaunay_prompt_uses_exact_points():
    from visual_geometry_bench.datagen.delaunay_tasks import _to_points, generate_dataset_record

    record = generate_dataset_record({"num_points": 58, "seed": 1063})
    prompt_points = ast.literal_eval(
        re.search(
            r"\n(\[.*\])\n\nReturn the Delaunay triangulation",
            record["prompt"],
            re.S,
        ).group(1)
    )

    assert prompt_points == [list(map(float, pt)) for pt in _to_points(record["datagen_args"])]


def test_delaunay_visuals_log_to_transcript(monkeypatch):
    fake_transcript = FakeTranscript()
    monkeypatch.setattr("scbench_posttrain.vgb.transcript", lambda: fake_transcript)
    monkeypatch.setattr(
        "scbench_posttrain.vgb._render_record_markdown",
        lambda *args, **kwargs: "### Task visual\n\n![Task visual](data:image/png;base64,AAAA)",
    )

    from visual_geometry_bench.datagen.delaunay_tasks import generate_dataset_record

    record = generate_dataset_record({"num_points": 8, "seed": 33})
    task = VGBTask(
        name="delaunay",
        title="Delaunay Triangulation",
        dataset=MemoryDataset(
            samples=[
                Sample(
                    input=record["prompt"],
                    id=record["id"],
                    metadata={
                        "name": "delaunay",
                        "title": "Delaunay Triangulation",
                        "record_id": record["id"],
                        "record_index": 0,
                        "problem_type": "delaunay_triangulation",
                    },
                )
            ]
        ),
        records=(record,),
    )
    log_prompt_artifacts(task.records[0])

    sample = task.dataset[0]
    state = _make_state(sample=sample, completion=str(task.records[0]["ground_truth"]))
    result = asyncio.run(vgb_score(task)(state, None))

    assert result.value == 1.0
    assert len(fake_transcript.events) == 2
    assert fake_transcript.events[0] == (
        "### Task visual\n\n![Task visual](data:image/png;base64,AAAA)",
        "solver",
    )
    assert fake_transcript.events[1] == (
        "### Task visual\n\n![Task visual](data:image/png;base64,AAAA)",
        "scorer",
    )


def test_repo_no_longer_references_scbench_surface():
    result = subprocess.run(
        [
            "rg",
            "-n",
            "SpatialCompetenceBenchmark|evals/scbench.py|scbench_task|task_id=|load_scbench_task",
            "README.md",
            "src",
            "evals",
            ".gitmodules",
            "pyproject.toml",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1, result.stdout
