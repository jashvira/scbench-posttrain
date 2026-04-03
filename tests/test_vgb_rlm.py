from __future__ import annotations

import asyncio
import sys
import types

from inspect_ai.model import ChatMessageUser, ModelOutput
from inspect_ai.solver import TaskState

from evals.vgb import VGBRunStore, vgb_rlm_full, vgb_rlm_repl
from scbench_posttrain.vgb import load_vgb_task


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


async def _unexpected_generate(*args, **kwargs):
    del args, kwargs
    raise AssertionError("custom RLM solver should not call Inspect generate()")


def _install_fake_rlm(monkeypatch):
    fake_rlm_module = types.ModuleType("rlm")
    fake_logger_module = types.ModuleType("rlm.logger")

    class FakeUsageSummary:
        def to_dict(self) -> dict[str, object]:
            return {
                "model_usage_summaries": {
                    "fake-model": {
                        "total_calls": 1,
                        "total_input_tokens": 10,
                        "total_output_tokens": 5,
                    }
                }
            }

    class FakeCompletion:
        def __init__(self, response: str):
            self.response = response
            self.usage_summary = FakeUsageSummary()
            self.execution_time = 0.25
            self.metadata = {
                "run_metadata": {"root_model": "fake-model", "max_depth": 2},
                "iterations": [
                    {
                        "iteration": 1,
                        "response": "thinking",
                        "code_blocks": [],
                        "final_answer": response,
                    }
                ],
            }

    class FakeLogger:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    class FakeRLM:
        init_calls: list[dict[str, object]] = []
        responses = ["[]"]
        completion_calls: list[dict[str, object]] = []

        def __init__(self, **kwargs):
            FakeRLM.init_calls.append(kwargs)

        def completion(self, prompt, root_prompt=None):
            FakeRLM.completion_calls.append({"prompt": prompt, "root_prompt": root_prompt})
            return FakeCompletion(FakeRLM.responses.pop(0))

    fake_rlm_module.RLM = FakeRLM
    fake_logger_module.RLMLogger = FakeLogger
    FakeRLM.init_calls = []
    FakeRLM.responses = ["[]"]
    FakeRLM.completion_calls = []

    monkeypatch.setitem(sys.modules, "rlm", fake_rlm_module)
    monkeypatch.setitem(sys.modules, "rlm.logger", fake_logger_module)
    return FakeRLM


def test_vgb_rlm_repl_solver_uses_native_shallow_settings(monkeypatch):
    fake_rlm = _install_fake_rlm(monkeypatch)
    task = load_vgb_task("topology_enumeration")
    state = _make_state(sample=task.dataset[0], completion="")

    solved = asyncio.run(vgb_rlm_repl(rlm_model_name="fake-model")(state, _unexpected_generate))
    store = solved.store_as(VGBRunStore)

    assert solved.output.completion == "[]"
    assert store.name == "topology_enumeration"
    assert store.arm == "rlm_repl"
    assert len(store.rlm_trace["attempts"]) == 1
    assert store.rlm_trace["attempts"][0]["run_metadata"]["root_model"] == "fake-model"
    assert store.trajectory_present is True
    assert store.trajectory_iterations == 1
    assert store.rlm_run_config["arm"] == "rlm_repl"
    assert store.rlm_run_config["backend"] == "openai"
    assert store.rlm_run_config["environment"] == "local"
    assert store.rlm_run_config["rlm_model_name"] == "fake-model"
    assert store.rlm_run_config["max_iterations"] == 12
    assert store.rlm_run_config["max_depth"] == 1
    assert store.rlm_run_config["other_backends"] == []
    assert "Recursive child RLM calls are not enabled" in store.rlm_run_config["root_prompt"]
    assert "Return only the final answer in the exact literal format requested in `context`" in (
        store.rlm_run_config["root_prompt"]
    )
    assert "api_key" not in str(store.rlm_run_config)
    assert fake_rlm.init_calls[0]["max_depth"] == 1
    assert "other_backends" not in fake_rlm.init_calls[0]


def test_vgb_rlm_retries_malformed_meta_final(monkeypatch):
    fake_rlm = _install_fake_rlm(monkeypatch)
    fake_rlm.responses = [
        '"Please run the REPL inspection first; I have requested printing `context` to read the task before solving it."',
        '"I’m unable to access the REPL output in this interface, so I can’t derive the answer from `context` here."',
        "[]",
    ]
    task = load_vgb_task("topology_enumeration")
    state = _make_state(sample=task.dataset[0], completion="")

    solved = asyncio.run(vgb_rlm_full(rlm_model_name="fake-model")(state, _unexpected_generate))
    store = solved.store_as(VGBRunStore)

    assert solved.output.completion == "[]"
    assert store.arm == "rlm_full"
    assert store.rlm_run_config["repair_attempted"] is True
    assert store.rlm_execution_time_seconds == 0.75
    assert store.usage_summary == {
        "model_usage_summaries": {
            "fake-model": {
                "total_calls": 3,
                "total_input_tokens": 30,
                "total_output_tokens": 15,
            }
        }
    }
    assert len(store.rlm_trace["attempts"]) == 3
    assert store.trajectory_iterations == 3
    assert len(fake_rlm.completion_calls) == 3
    assert "There is no human in the loop" in fake_rlm.completion_calls[0]["root_prompt"]
    assert "must not ask the user to run code" in fake_rlm.completion_calls[1]["root_prompt"]
    assert "wait for the resulting REPL output in the next iteration" in fake_rlm.completion_calls[2]["root_prompt"]
