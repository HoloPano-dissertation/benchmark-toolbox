from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Sequence

from benchmark_toolbox.domain import SceneOutput, SceneSample
from benchmark_toolbox.registry import ComponentRegistry


class BaseSceneEstimator(ABC):
    registry: ComponentRegistry[BaseSceneEstimator] = ComponentRegistry(
        "scene estimator"
    )

    @abstractmethod
    def predict(self, sample: SceneSample) -> SceneOutput:
        """Return the normalized scene representation."""

    def predict_many(self, samples: Sequence[SceneSample]) -> Iterable[SceneOutput]:
        for sample in samples:
            yield self.predict(sample)

    @classmethod
    def create(
        cls, name: str, parameters: dict[str, Any] | None = None
    ) -> BaseSceneEstimator:
        return cls.registry.create(name, parameters)

    def provenance(self) -> dict[str, Any]:
        return {}
