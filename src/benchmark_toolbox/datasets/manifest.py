from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Mapping

from benchmark_toolbox.datasets.base import BaseDatasetLoader
from benchmark_toolbox.domain import SceneOutput, SceneSample


@BaseDatasetLoader.registry.register("manifest")
class ManifestDatasetLoader(BaseDatasetLoader):
    def __init__(self, parameters: dict[str, Any]) -> None:
        try:
            self.manifest_path = Path(parameters["manifest"]).resolve()
        except KeyError as error:
            raise ValueError("ManifestDatasetLoader requires 'manifest'") from error

    def _resolve(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else (self.manifest_path.parent / path).resolve()

    def _load_ground_truth(
        self, record: Mapping[str, Any]
    ) -> SceneOutput | None:
        ground_truth = record.get("ground_truth")
        if ground_truth is None:
            return None
        if isinstance(ground_truth, Mapping):
            return SceneOutput.from_dict(ground_truth)
        ground_truth_path = self._resolve(str(ground_truth))
        with ground_truth_path.open(encoding="utf-8") as ground_truth_file:
            return SceneOutput.from_dict(json.load(ground_truth_file))

    def samples(self) -> Iterator[SceneSample]:
        with self.manifest_path.open(encoding="utf-8") as manifest:
            for line_number, line in enumerate(manifest, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                try:
                    sample_id = str(record["sample_id"])
                    input_path = self._resolve(str(record["input"]))
                except KeyError as error:
                    raise ValueError(
                        f"Invalid manifest record at line {line_number}: {error}"
                    ) from error
                if not input_path.exists():
                    raise FileNotFoundError(
                        f"Input for sample '{sample_id}' does not exist: {input_path}"
                    )
                yield SceneSample(
                    sample_id=sample_id,
                    input_path=input_path,
                    ground_truth=self._load_ground_truth(record),
                    metadata=dict(record.get("metadata", {})),
                )

BaseDatasetLoader.registry.alias("manifest", "igibson")
