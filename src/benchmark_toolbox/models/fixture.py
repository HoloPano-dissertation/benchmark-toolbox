from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark_toolbox.domain import SceneOutput, SceneSample
from benchmark_toolbox.models.base import BaseSceneEstimator


@BaseSceneEstimator.registry.register("fixture")
class FixtureEstimator(BaseSceneEstimator):
    def __init__(self, parameters: dict[str, Any]) -> None:
        try:
            self.prediction_dir = Path(parameters["prediction_dir"])
        except KeyError as error:
            raise ValueError("FixtureEstimator requires 'prediction_dir'") from error

    def predict(self, sample: SceneSample) -> SceneOutput:
        prediction_path = self.prediction_dir / f"{sample.sample_id}.json"
        with prediction_path.open(encoding="utf-8") as prediction_file:
            return SceneOutput.from_dict(json.load(prediction_file))
