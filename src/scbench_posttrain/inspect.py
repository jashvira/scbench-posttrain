from __future__ import annotations

from typing import Iterable

from scbench_posttrain.tasks import TaskFamily, TaskInstance


def build_inspect_task(
    family: TaskFamily,
    instances: Iterable[TaskInstance],
    system_prompt: str | None = None,
):
    from inspect_ai import Task
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.scorer import Score, scorer
    from inspect_ai.solver import generate, system_message

    samples = [
        Sample(
            input=instance.prompt,
            target=str(instance.answer) if instance.answer is not None else "",
            metadata={"instance": instance},
            id=f"{instance.task_name}-{instance.split}-{instance.seed}",
        )
        for instance in instances
    ]

    @scorer(metrics=[])
    def deterministic_verifier():
        async def score(state, target):
            instance = state.metadata["instance"]
            parsed = family.parse_answer(state.output.completion, instance)
            result = family.verify(parsed, instance)
            metadata = {**result.details, **result.metrics}
            return Score(value=result.score, answer=str(target), metadata=metadata)

        return score

    solver = [generate()]
    if system_prompt:
        solver.insert(0, system_message(system_prompt))

    return Task(
        dataset=MemoryDataset(samples),
        solver=solver,
        scorer=deterministic_verifier(),
    )
