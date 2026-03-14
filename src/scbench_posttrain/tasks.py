from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class TaskInstance:
    task_name: str
    split: str
    seed: int
    prompt: str
    answer: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationResult:
    score: float
    passed: bool
    metrics: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


class TaskFamily(Protocol):
    name: str

    def sample_instance(self, split: str, seed: int, difficulty: str | None = None) -> TaskInstance:
        ...

    def parse_answer(self, raw_output: str, instance: TaskInstance) -> Any:
        ...

    def verify(self, parsed_answer: Any, instance: TaskInstance) -> VerificationResult:
        ...


@dataclass
class TaskRegistry:
    _families: dict[str, TaskFamily] = field(default_factory=dict)

    def register(self, family: TaskFamily) -> None:
        if family.name in self._families:
            raise ValueError(f"Task family already registered: {family.name}")
        self._families[family.name] = family

    def get(self, name: str) -> TaskFamily:
        try:
            return self._families[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._families))
            raise KeyError(f"Unknown task family: {name}. Known: {known}") from exc

    def names(self) -> list[str]:
        return sorted(self._families)
