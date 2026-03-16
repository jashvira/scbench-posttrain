from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.model import (
    ChatMessageAssistant,
    GenerateConfig,
    ModelOutput,
)
from inspect_ai.scorer import Score, scorer
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver
from inspect_ai.util import StoreModel
from pydantic import Field

from scbench_posttrain.delaunay import (
    DelaunayRecord,
    DelaunaySampleMetadata,
    render_delaunay_prompt_image,
    score_delaunay_answer,
)

DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "delaunay_pilot.jsonl"
DEFAULT_GENERATE_CONFIG = GenerateConfig(
    max_retries=0,
    timeout=960,
    attempt_timeout=900,
    reasoning_summary="concise",
    reasoning_effort="high",
    max_tokens=32768,
    verbosity="low",
)
DEFAULT_GENERATE_CONFIG_METADATA = DEFAULT_GENERATE_CONFIG.model_dump(exclude_none=True)


class DelaunayRunStore(StoreModel):
    arm: str = "direct"
    rlm_execution_time_seconds: float | None = None
    usage_summary: dict[str, Any] = Field(default_factory=dict)
    trajectory_present: bool = False
    trajectory_iterations: int = 0
    rlm_model_name: str | None = None
    rlm_trace: dict[str, Any] = Field(default_factory=dict)
    rlm_run_config: dict[str, Any] = Field(default_factory=dict)


DelaunayRunStore.model_rebuild()


def _json_safe(value: Any) -> Any:
    """Convert nested values into a JSON-serializable shape with minimal coercion."""

    return json.loads(json.dumps(value, default=str))


def _merge_usage_summaries(usages: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge multiple RLM usage summaries into one aggregate payload."""

    merged: dict[str, dict[str, float | int]] = {}
    total_cost = 0.0
    has_cost = False

    for usage in usages:
        for model, summary in usage.get("model_usage_summaries", {}).items():
            bucket = merged.setdefault(
                model,
                {
                    "total_calls": 0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                },
            )
            bucket["total_calls"] += int(summary.get("total_calls", 0))
            bucket["total_input_tokens"] += int(summary.get("total_input_tokens", 0))
            bucket["total_output_tokens"] += int(summary.get("total_output_tokens", 0))
            if "total_cost" in summary and summary["total_cost"] is not None:
                bucket["total_cost"] = float(bucket.get("total_cost", 0.0)) + float(
                    summary["total_cost"]
                )

        if "total_cost" in usage and usage["total_cost"] is not None:
            total_cost += float(usage["total_cost"])
            has_cost = True

    result: dict[str, Any] = {"model_usage_summaries": merged}
    if has_cost:
        result["total_cost"] = total_cost
    return result


def _delaunay_record(state: TaskState, metadata: DelaunaySampleMetadata) -> dict[str, Any]:
    """Rebuild the verifier record shape for a sample from Inspect state."""

    return {
        "id": metadata.record_id,
        "prompt": state.input_text,
        "ground_truth": metadata.ground_truth,
        "metadata": {
            "problem_type": metadata.problem_type,
            "difficulty": metadata.difficulty,
            "tags": metadata.tags,
        },
        "datagen_args": metadata.datagen_args,
    }


def record_to_sample(record: dict[str, Any]) -> Sample:
    """Map a frozen JSONL record into a normal Inspect sample."""

    validated = DelaunayRecord.model_validate(record)
    metadata = DelaunaySampleMetadata(
        record_id=validated.id,
        problem_type=validated.metadata.problem_type,
        difficulty=validated.metadata.difficulty,
        tags=validated.metadata.tags,
        ground_truth=validated.ground_truth,
        datagen_args=validated.datagen_args,
    )
    sample_metadata = metadata.model_dump(mode="json")
    sample_metadata["prompt_image"] = render_delaunay_prompt_image(validated.datagen_args)
    return Sample(
        input=validated.prompt,
        target=json.dumps(validated.ground_truth),
        id=validated.id,
        metadata=sample_metadata,
    )


def load_delaunay_dataset(dataset_path: str | Path = DEFAULT_DATASET_PATH):
    """Load the frozen pilot dataset with Inspect's native JSON reader."""

    return json_dataset(str(dataset_path), sample_fields=record_to_sample, name="delaunay_pilot")


def _resolve_rlm_model_name(state: TaskState, override: str | None) -> str:
    """Resolve the model name to pass through to RLM."""

    if override:
        return override

    model_name = str(state.model)
    if "/" not in model_name:
        return model_name

    provider, resolved = model_name.split("/", 1)
    if provider != "openai":
        raise ValueError(
            "RLM solvers currently support only OpenAI-compatible Inspect models. "
            "Pass `rlm_model_name=` explicitly when using a provider-prefixed model string."
        )
    return resolved


def _rlm_root_prompt(arm: str) -> str:
    """Return the control prompt for an RLM solver arm."""

    guidance = (
        "Use the local Python REPL and built-in query helpers if useful. "
        "Recursive child RLM calls are not enabled in this arm."
        if arm == "rlm_repl"
        else "Use the local Python REPL and recursive LM helpers if useful."
    )
    return (
        "Solve the Delaunay task in `context`. "
        f"{guidance} "
        "Do not call FINAL or FINAL_VAR until you have the actual triangulation. "
        "If you are still inspecting `context` or testing code, continue iterating. "
        "Never finalize with a request for more work, a next-step note, or any other non-answer text. "
        "When you have the final triangulation, assign it to a variable and return it with FINAL_VAR."
    )


async def _run_rlm_arm(
    state: TaskState,
    *,
    arm: str,
    rlm_model_name: str | None,
    max_iterations: int,
    max_depth: int,
) -> TaskState:
    """Run one RLM-backed solver arm and write its completion into Inspect state."""

    try:
        from rlm import RLM
        from rlm.logger import RLMLogger
    except ImportError as exc:
        raise RuntimeError(
            "RLM solvers require the `rlm` package in the current environment. "
            "Install it before using `delaunay_rlm_repl` or `delaunay_rlm_full`."
        ) from exc

    resolved_model_name = _resolve_rlm_model_name(state, rlm_model_name)
    backend_kwargs: dict[str, Any] = {"model_name": resolved_model_name}
    if api_key := os.getenv("OPENAI_API_KEY"):
        backend_kwargs["api_key"] = api_key
    if base_url := os.getenv("OPENAI_BASE_URL"):
        backend_kwargs["base_url"] = base_url
    logger = RLMLogger()
    store = state.store_as(DelaunayRunStore)
    store.arm = arm
    store.rlm_model_name = resolved_model_name
    record = _delaunay_record(state, state.metadata_as(DelaunaySampleMetadata))

    rlm_kwargs: dict[str, Any] = {
        "backend": "openai",
        "backend_kwargs": backend_kwargs,
        "environment": "local",
        "logger": logger,
        "max_iterations": max_iterations,
        "max_depth": max(max_depth, 1),
    }
    if arm == "rlm_full":
        rlm_kwargs["other_backends"] = ["openai"]
        rlm_kwargs["other_backend_kwargs"] = [backend_kwargs.copy()]

    prompt_context = [{"role": message.role, "content": message.text} for message in state.messages]
    root_prompt = _rlm_root_prompt(arm)
    store.rlm_run_config = _json_safe(
        {
            "arm": arm,
            "backend": rlm_kwargs["backend"],
            "environment": rlm_kwargs["environment"],
            "rlm_model_name": resolved_model_name,
            "max_iterations": max_iterations,
            "max_depth": rlm_kwargs["max_depth"],
            "root_prompt": root_prompt,
            "other_backends": rlm_kwargs.get("other_backends", []),
            "repair_attempted": False,
        }
    )
    runner = RLM(**rlm_kwargs)
    completion = runner.completion(prompt_context, root_prompt=root_prompt)
    completions = [completion]
    evaluation = score_delaunay_answer(completion.response, record)
    if not evaluation.passed and evaluation.error_type != "exact_mismatch":
        repair_prompt = (
            f"{root_prompt} "
            "Your previous completion was not a valid triangulation. "
            "Do not ask for more work or describe your next step. "
            "Continue using the REPL and only finalize with the triangulation itself."
        )
        completion = runner.completion(prompt_context, root_prompt=repair_prompt)
        completions.append(completion)
        store.rlm_run_config["repair_attempted"] = True

    traces = [_json_safe(item.metadata or {}) for item in completions if item.metadata]
    store.rlm_execution_time_seconds = sum(item.execution_time for item in completions)
    store.usage_summary = _merge_usage_summaries(
        [item.usage_summary.to_dict() for item in completions]
    )
    store.rlm_trace = {"attempts": traces} if traces else {}
    store.trajectory_present = bool(store.rlm_trace)
    store.trajectory_iterations = sum(len(trace.get("iterations", [])) for trace in traces)

    model_name = str(state.model)
    state.output = ModelOutput.from_content(model=model_name, content=completion.response)
    state.messages.append(
        ChatMessageAssistant(content=completion.response, source="generate", model=model_name)
    )
    state.completed = True
    return state


@solver
def delaunay_direct():
    """Run the default Inspect generation path for Delaunay."""

    direct_generate = generate()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        """Tag the run as direct generation before delegating to Inspect."""

        store = state.store_as(DelaunayRunStore)
        store.arm = "direct"
        return await direct_generate(state, generate)

    return solve


@solver
def delaunay_rlm_repl(
    max_iterations: int = 12,
    rlm_model_name: str | None = None,
):
    """Run Delaunay through native shallow RLM with a local REPL."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        """Execute the shallow RLM arm for one sample."""

        del generate
        return await _run_rlm_arm(
            state,
            arm="rlm_repl",
            rlm_model_name=rlm_model_name,
            max_iterations=max_iterations,
            max_depth=1,
        )

    return solve


@solver
def delaunay_rlm_full(
    max_iterations: int = 12,
    max_depth: int = 2,
    rlm_model_name: str | None = None,
):
    """Run Delaunay through native recursive RLM."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        """Execute the full RLM arm for one sample."""

        del generate
        return await _run_rlm_arm(
            state,
            arm="rlm_full",
            rlm_model_name=rlm_model_name,
            max_iterations=max_iterations,
            max_depth=max_depth,
        )

    return solve


@scorer(metrics=[])
def delaunay_exact():
    """Score one completion with the shared exact Delaunay verifier."""

    async def score(state: TaskState, target: Any) -> Score:
        """Parse and score one sample output, then attach run metadata."""

        del target
        metadata = state.metadata_as(DelaunaySampleMetadata)
        store = state.store_as(DelaunayRunStore)
        evaluation = score_delaunay_answer(
            state.output.completion if state.output is not None else "",
            _delaunay_record(state, metadata),
        )

        return Score(
            value=evaluation.score,
            answer=evaluation.extracted_literal,
            metadata={
                "parsed_ok": evaluation.parsed_ok,
                "passed": evaluation.passed,
                "error_type": evaluation.error_type,
                "problem_type": metadata.problem_type,
                "difficulty": metadata.difficulty,
                "arm": store.arm,
                "missing": evaluation.missing,
                "extra": evaluation.extra,
                "rlm_execution_time_seconds": store.rlm_execution_time_seconds,
                "trajectory_present": store.trajectory_present,
                "trajectory_iterations": store.trajectory_iterations,
                "usage_summary": store.usage_summary,
                "rlm_model_name": store.rlm_model_name,
            },
        )

    return score


@task
def delaunay_pilot(
    solver: Solver | None = None,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
) -> Task:
    """Build the Inspect task for the frozen Delaunay pilot split."""

    return Task(
        dataset=load_delaunay_dataset(dataset_path),
        solver=solver or delaunay_direct(),
        scorer=delaunay_exact(),
        config=DEFAULT_GENERATE_CONFIG,
        metadata={"generate_config": DEFAULT_GENERATE_CONFIG_METADATA},
    )
