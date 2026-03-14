from __future__ import annotations

import asyncio
import json
import sys
import types

from inspect_ai.model import ChatMessageUser, ModelOutput
from inspect_ai.solver import TaskState

from evals.delaunay import (
    DEFAULT_DATASET_PATH,
    DelaunayRunStore,
    delaunay_exact,
    delaunay_rlm_full,
    delaunay_rlm_repl,
    load_delaunay_dataset,
    record_to_sample,
)
from scbench_posttrain.delaunay import DelaunaySampleMetadata, generate_dataset_record


def _make_state(*, metadata: dict, completion: str = "") -> TaskState:
    """Build a minimal Inspect task state for unit tests."""

    return TaskState(
        model="openai/test-model",
        sample_id="sample-1",
        epoch=1,
        input="prompt",
        messages=[ChatMessageUser(content="prompt")],
        output=ModelOutput.from_content(model="openai/test-model", content=completion),
        completed=bool(completion),
        metadata=metadata,
        store={},
    )


def _demo_metadata() -> dict:
    """Build shared sample metadata for RLM solver tests."""

    return DelaunaySampleMetadata(
        record_id="demo",
        difficulty="easy",
        tags=["geometry"],
        ground_truth=[[0, 1, 2]],
        datagen_args={"num_points": 3, "seed": 1},
    ).model_dump(mode="json")


async def _unexpected_generate(*args, **kwargs):
    """Fail if a custom solver unexpectedly calls Inspect generation."""

    del args, kwargs
    raise AssertionError("custom RLM solver should not call Inspect generate()")


def _install_fake_rlm(monkeypatch):
    """Install a fake `rlm` module so solver tests stay fully local."""

    fake_rlm_module = types.ModuleType("rlm")
    fake_logger_module = types.ModuleType("rlm.logger")

    class FakeUsageSummary:
        def to_dict(self) -> dict[str, object]:
            """Return a small fake usage payload."""

            return {"fake-model": {"total_calls": 1, "total_input_tokens": 10, "total_output_tokens": 5}}

    class FakeCompletion:
        def __init__(self, response: str):
            """Capture a fake RLM completion payload."""

            self.response = response
            self.usage_summary = FakeUsageSummary()
            self.execution_time = 0.25
            self.metadata = {"iterations": [{"iteration": 1}]}

    class FakeLogger:
        def __init__(self, *args, **kwargs):
            """Ignore logger construction arguments in tests."""

            del args, kwargs

    class FakeRLM:
        init_calls: list[dict[str, object]] = []

        def __init__(self, **kwargs):
            """Record solver configuration for later assertions."""

            FakeRLM.init_calls.append(kwargs)

        def completion(self, prompt, root_prompt=None):
            """Return a fixed completion while recording call details."""

            del prompt, root_prompt
            return FakeCompletion("[[0, 1, 2]]")

    fake_rlm_module.RLM = FakeRLM
    fake_logger_module.RLMLogger = FakeLogger

    monkeypatch.setitem(sys.modules, "rlm", fake_rlm_module)
    monkeypatch.setitem(sys.modules, "rlm.logger", fake_logger_module)

    return FakeRLM


def test_record_to_sample_maps_frozen_record():
    """Dataset records should map into plain Inspect samples."""

    record = generate_dataset_record({"num_points": 6, "seed": 10}, difficulty="easy")

    sample = record_to_sample(record)

    assert sample.id == record["id"]
    assert sample.input == record["prompt"]
    assert sample.target == json.dumps(record["ground_truth"])
    assert sample.metadata["record_id"] == record["id"]
    assert sample.metadata["difficulty"] == "easy"


def test_load_delaunay_dataset_reads_json_dataset():
    """The frozen JSONL should load through Inspect's dataset reader."""

    dataset = load_delaunay_dataset(DEFAULT_DATASET_PATH)

    assert len(dataset) == 2
    assert [sample.metadata["datagen_args"] for sample in dataset] == [
        {"num_points": 8, "seed": 33},
        {"num_points": 20, "seed": 44},
    ]


def test_shared_scorer_includes_arm_metadata():
    """The shared scorer should preserve the selected arm in metadata."""

    record = generate_dataset_record({"num_points": 6, "seed": 10}, difficulty="easy")
    sample = record_to_sample(record)
    state = _make_state(metadata=sample.metadata, completion=json.dumps(record["ground_truth"]))
    store = state.store_as(DelaunayRunStore)
    store.arm = "rlm_full"

    result = asyncio.run(delaunay_exact()(state, None))

    assert result.value == 1.0
    assert result.metadata["parsed_ok"] is True
    assert result.metadata["passed"] is True
    assert result.metadata["arm"] == "rlm_full"


def test_delaunay_rlm_repl_solver_uses_native_shallow_settings(monkeypatch):
    """The shallow RLM arm should use native non-recursive settings."""

    fake_rlm = _install_fake_rlm(monkeypatch)
    state = _make_state(metadata=_demo_metadata())

    solved = asyncio.run(delaunay_rlm_repl(rlm_model_name="fake-model")(state, _unexpected_generate))
    store = solved.store_as(DelaunayRunStore)

    assert solved.output.completion == "[[0, 1, 2]]"
    assert store.arm == "rlm_repl"
    assert fake_rlm.init_calls[0]["max_depth"] == 1
    assert "other_backends" not in fake_rlm.init_calls[0]


def test_delaunay_rlm_full_solver_uses_recursive_backend(monkeypatch):
    """The recursive RLM arm should register the extra backend."""

    fake_rlm = _install_fake_rlm(monkeypatch)
    state = _make_state(metadata=_demo_metadata())

    solved = asyncio.run(
        delaunay_rlm_full(rlm_model_name="fake-model")(state, _unexpected_generate)
    )
    store = solved.store_as(DelaunayRunStore)

    assert solved.output.completion == "[[0, 1, 2]]"
    assert store.arm == "rlm_full"
    assert fake_rlm.init_calls[0]["max_depth"] == 2
    assert fake_rlm.init_calls[0]["other_backends"] == ["openai"]
