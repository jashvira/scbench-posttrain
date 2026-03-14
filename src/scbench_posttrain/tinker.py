from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Sequence

from scbench_posttrain.tasks import TaskFamily, TaskInstance


class SingleTurnTaskEnv:
    def __init__(self, family: TaskFamily, instance: TaskInstance, renderer: Any):
        self.family = family
        self.instance = instance
        self.renderer = renderer

    async def initial_observation(self):
        messages = [{"role": "user", "content": self.instance.prompt}]
        return (
            self.renderer.build_generation_prompt(messages),
            self.renderer.get_stop_sequences(),
        )

    async def step(self, action):
        import tinker
        from tinker_cookbook.rl.types import StepResult

        message, valid = self.renderer.parse_response(action)
        parsed = self.family.parse_answer(str(message["content"]), self.instance)
        result = self.family.verify(parsed, self.instance)

        metrics = {"format_valid": float(valid), "passed": float(result.passed), **result.metrics}

        return StepResult(
            reward=float(result.score),
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.renderer.get_stop_sequences(),
            metrics=metrics,
        )


@dataclass(frozen=True)
class _SingleTurnGroupBuilder:
    env_thunk: Any
    num_envs: int
    tags: tuple[str, ...]

    async def make_envs(self) -> Sequence[Any]:
        return [self.env_thunk() for _ in range(self.num_envs)]

    def logging_tags(self) -> list[str]:
        return list(self.tags)


def make_single_turn_group_builder(
    family: TaskFamily,
    instance: TaskInstance,
    renderer: Any,
    num_envs: int,
):
    env_thunk = partial(SingleTurnTaskEnv, family, instance, renderer)
    return _SingleTurnGroupBuilder(
        env_thunk=env_thunk,
        num_envs=num_envs,
        tags=(family.name, instance.split),
    )
