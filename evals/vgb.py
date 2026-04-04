import json
import os
from typing import Any

from inspect_ai import Task, task
from inspect_ai.model import ChatMessageAssistant, GenerateConfig, ModelOutput
from inspect_ai.scorer import Score, mean, scorer
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver
from inspect_ai.util import StoreModel
from pydantic import Field

from scbench_posttrain.vgb import (
    VGBTask,
    extract_vgb_answer,
    grade_vgb_answer,
    load_vgb_task,
    log_prompt_artifacts,
    log_score_artifacts,
    slice_vgb_task,
)

DEFAULT_GENERATE_CONFIG = GenerateConfig(
    max_retries=0,
    timeout=960,
    attempt_timeout=900,
    max_tokens=8192,
)


class VGBRunStore(StoreModel):
    name: str | None = None
    arm: str = "direct"
    rlm_execution_time_seconds: float | None = None
    usage_summary: dict[str, Any] = Field(default_factory=dict)
    trajectory_present: bool = False
    trajectory_iterations: int = 0
    rlm_model_name: str | None = None
    rlm_trace: dict[str, Any] = Field(default_factory=dict)
    rlm_run_config: dict[str, Any] = Field(default_factory=dict)


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _merge_usage_summaries(usages: list[dict[str, Any]]) -> dict[str, Any]:
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


def _resolve_rlm_model_name(state: TaskState, override: str | None) -> str:
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
    guidance = (
        "Use the local Python REPL and built-in query helpers if useful. "
        "Recursive child RLM calls are not enabled in this arm."
        if arm == "rlm_repl"
        else "Use the local Python REPL and recursive LM helpers if useful."
    )
    return (
        "Solve the VGB task in `context`. "
        f"{guidance} "
        "`context` already contains the full task prompt text. "
        "Any `repl` block you emit is executed automatically by the environment. "
        "Its stdout/stderr is returned to you in later iterations. "
        "There is no human in the loop during execution. "
        "Never ask the user to run code, provide REPL output, or continue the interaction. "
        "Do not call FINAL or FINAL_VAR until you have the actual final answer. "
        "If you are still inspecting `context` or testing code, continue iterating. "
        "Do not emit an inspection `repl` block and then finalize in the same response. "
        "Never finalize with a request for more work, a next-step note, or any other non-answer text. "
        "Return only the final answer in the exact literal format requested in `context`, with no markdown fences or surrounding prose. "
        "When you have the final answer, assign it to a variable and return it with FINAL_VAR."
    )


def _needs_repair(completion: str) -> bool:
    if extract_vgb_answer(completion) is None:
        return True

    lowered = completion.lower()
    return any(
        pattern in lowered
        for pattern in (
            "please run the repl",
            "run the repl inspection",
            "unable to access the repl output",
            "can't access the repl output",
            "cannot access the repl output",
            "provide repl output",
            "continue the interaction",
        )
    )


async def _run_rlm_arm(
    state: TaskState,
    *,
    arm: str,
    rlm_model_name: str | None,
    max_iterations: int,
    max_depth: int,
) -> TaskState:
    try:
        from rlm import RLM
        from rlm.logger import RLMLogger
    except ImportError as exc:
        raise RuntimeError(
            "RLM solvers require the `rlm` package in the current environment. "
            "Install it before using `vgb_rlm_repl` or `vgb_rlm_full`."
        ) from exc

    loaded_task = load_vgb_task(str(state.metadata["name"]))
    record = loaded_task.records[int(state.metadata["record_index"])]
    log_prompt_artifacts(record)
    resolved_model_name = _resolve_rlm_model_name(state, rlm_model_name)
    backend_kwargs: dict[str, Any] = {"model_name": resolved_model_name}
    if api_key := os.getenv("OPENAI_API_KEY"):
        backend_kwargs["api_key"] = api_key
    if base_url := os.getenv("OPENAI_BASE_URL"):
        backend_kwargs["base_url"] = base_url

    logger = RLMLogger()
    store = state.store_as(VGBRunStore)
    store.name = loaded_task.name
    store.arm = arm
    store.rlm_model_name = resolved_model_name

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
            "name": loaded_task.name,
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
    completions = []
    prompt = root_prompt
    max_repairs = 2
    repairs = 0

    while True:
        completion = runner.completion(prompt_context, root_prompt=prompt)
        completions.append(completion)
        if not _needs_repair(completion.response) or repairs >= max_repairs:
            break

        repairs += 1
        store.rlm_run_config["repair_attempted"] = True
        prompt = (
            f"{root_prompt} "
            "Your previous completion was not a valid final answer for this task. "
            "The environment already executed any `repl` blocks you emitted. "
            "You must not ask the user to run code or provide REPL output. "
            "Do not finalize in the same response as a first-pass inspection block. "
            "Inspect `context`, wait for the resulting REPL output in the next iteration, "
            "and only finalize once you have the actual final answer in the required format."
        )

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
def vgb_direct():
    direct_generate = generate()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        loaded_task = load_vgb_task(str(state.metadata["name"]))
        record = loaded_task.records[int(state.metadata["record_index"])]
        store = state.store_as(VGBRunStore)
        store.name = loaded_task.name
        store.arm = "direct"
        log_prompt_artifacts(record)
        return await direct_generate(state, generate)

    return solve


@solver
def vgb_rlm_repl(
    max_iterations: int = 12,
    rlm_model_name: str | None = None,
):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
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
def vgb_rlm_full(
    max_iterations: int = 12,
    max_depth: int = 2,
    rlm_model_name: str | None = None,
):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        del generate
        return await _run_rlm_arm(
            state,
            arm="rlm_full",
            rlm_model_name=rlm_model_name,
            max_iterations=max_iterations,
            max_depth=max_depth,
        )

    return solve


@scorer(metrics=[mean()])
def vgb_score(vgb_task: VGBTask):
    async def score(state: TaskState, target: Any) -> Score:
        del target
        completion = state.output.completion if state.output is not None else ""
        record = vgb_task.records[int(state.metadata["record_index"])]
        value, score_metadata, parsed_answer = grade_vgb_answer(record, completion)
        if parsed_answer is not None:
            log_score_artifacts(record, parsed_answer)
        store = state.store_as(VGBRunStore)

        return Score(
            value=float(value),
            answer=completion,
            metadata={
                "name": state.metadata["name"],
                "title": state.metadata["title"],
                "record_id": state.metadata["record_id"],
                "record_index": state.metadata["record_index"],
                "problem_type": state.metadata["problem_type"],
                "arm": store.arm,
                "rlm_execution_time_seconds": store.rlm_execution_time_seconds,
                "trajectory_present": store.trajectory_present,
                "trajectory_iterations": store.trajectory_iterations,
                "usage_summary": store.usage_summary,
                "rlm_model_name": store.rlm_model_name,
                **score_metadata,
            },
        )

    return score


@task
def vgb_task(
    task_name: str,
    record_indices: str | list[int] | list[str] | None = None,
    solver: Solver | None = None,
) -> Task:
    loaded_task = load_vgb_task(task_name)
    selected_indices: list[int] | None = None
    if record_indices is not None:
        if isinstance(record_indices, str):
            raw_indices = [part.strip() for part in record_indices.split(",") if part.strip()]
        else:
            raw_indices = []
            for item in record_indices:
                text = str(item).strip()
                if text:
                    raw_indices.append(text)
        selected_indices = [int(part) for part in raw_indices]
        loaded_task = slice_vgb_task(loaded_task, selected_indices)
    inspect_task_name = f"vgb_{loaded_task.name}"
    return Task(
        dataset=loaded_task.dataset,
        solver=solver or vgb_direct(),
        scorer=vgb_score(loaded_task),
        config=DEFAULT_GENERATE_CONFIG,
        name=inspect_task_name,
        display_name=loaded_task.title,
        metadata={
            "name": loaded_task.name,
            "title": loaded_task.title,
            "task_name": inspect_task_name,
            "record_indices": selected_indices,
            "generate_config": DEFAULT_GENERATE_CONFIG.model_dump(exclude_none=True),
        },
    )
