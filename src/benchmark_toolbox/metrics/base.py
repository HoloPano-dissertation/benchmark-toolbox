from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from benchmark_toolbox.domain import SceneOutput
from benchmark_toolbox.registry import ComponentRegistry


class BaseMetric(ABC):
    registry: ComponentRegistry[BaseMetric] = ComponentRegistry("metric")
    name: str
    requires_ground_truth: bool = True
    dataset_level: bool = False

    @abstractmethod
    def compute(
        self, prediction: SceneOutput, ground_truth: SceneOutput | None
    ) -> float:
        """Compute one scalar metric value for a scene."""

    @classmethod
    def create(
        cls, name: str, parameters: dict[str, Any] | None = None
    ) -> BaseMetric:
        return cls.registry.create(name, parameters)
