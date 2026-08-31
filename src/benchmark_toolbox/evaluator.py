from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark_toolbox.config import ExperimentConfig
from benchmark_toolbox.datasets.base import BaseDatasetLoader
from benchmark_toolbox.metrics.base import BaseMetric
from benchmark_toolbox.models.base import BaseSceneEstimator
from benchmark_toolbox.provenance import git_revision


@dataclass(frozen=True)
class EvaluationResult:
    output_dir: Path
    scene_count: int
    summary: dict[str, dict[str, Any]]


def _bootstrap_ci(values: list[float], seed: int, samples: int = 2000) -> tuple[float, float]:
    count = len(values)
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(samples):
        total = 0.0
        for _ in range(count):
            total += values[rng.randrange(count)]
        means.append(total / count)
    means.sort()
    low = means[int(0.025 * samples)]
    high = means[min(samples - 1, int(0.975 * samples))]
    return low, high


def _aggregate(values: list[float], seed: int = 0) -> dict[str, float | int]:
    count = len(values)
    if count == 0:
        return {"count": 0}
    mean = statistics.fmean(values)
    median = statistics.median(values)
    if count == 1:
        return {"count": 1, "mean": mean, "median": median, "ci95_low": mean, "ci95_high": mean}
    low, high = _bootstrap_ci(values, seed)
    return {
        "count": count,
        "mean": mean,
        "median": median,
        "ci95_low": low,
        "ci95_high": high,
    }


class Evaluator:
    def __init__(
        self,
        config: ExperimentConfig,
        estimator: BaseSceneEstimator,
        dataset: BaseDatasetLoader,
        metrics: list[BaseMetric],
    ) -> None:
        self.config = config
        self.estimator = estimator
        self.dataset = dataset
        names = [metric.name for metric in metrics]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                "Metric names must be unique; give the repeated entries a 'name' "
                f"parameter. Duplicates: {', '.join(duplicates)}"
            )
        self.metrics = metrics

    def _write_report(
        self, output_dir: Path, summary: dict[str, dict[str, Any]]
    ) -> None:
        lines = [
            f"# Experiment: {self.config.experiment_name}",
            "",
            "| Metric | Count | Mean | Median | 95% CI (bootstrap) |",
            "|---|---:|---:|---:|---:|",
        ]
        for metric_name, values in summary.items():
            if values.get("dataset_level"):
                lines.append(
                    f"| {metric_name} (dataset) | {values['count']} | "
                    f"{values['value']:.6f} | — | — |"
                )
            elif not values["count"]:
                lines.append(f"| {metric_name} | 0 | n/a | n/a | n/a |")
            else:
                lines.append(
                    f"| {metric_name} | {values['count']} | "
                    f"{values['mean']:.6f} | {values['median']:.6f} | "
                    f"[{values['ci95_low']:.6f}, {values['ci95_high']:.6f}] |"
                )
        (output_dir / "report.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    def run(self) -> EvaluationResult:
        random.seed(self.config.seed)
        output_dir = self.config.output_dir
        predictions_dir = output_dir / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)

        per_scene_metrics = [
            metric for metric in self.metrics if not getattr(metric, "dataset_level", False)
        ]
        dataset_metrics = [
            metric for metric in self.metrics if getattr(metric, "dataset_level", False)
        ]
        metric_values: dict[str, list[float]] = {
            metric.name: [] for metric in per_scene_metrics
        }
        records: list[dict[str, Any]] = []
        predictions: list[Any] = []
        ground_truths: list[Any] = []

        samples = list(self.dataset)
        for sample, prediction in zip(samples, self.estimator.predict_many(samples)):
            prediction_path = predictions_dir / f"{sample.sample_id}.json"
            prediction_path.write_text(
                json.dumps(
                    prediction.to_dict(),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            predictions.append(prediction)
            ground_truths.append(sample.ground_truth)
            scene_metrics: dict[str, float] = {}
            for metric in per_scene_metrics:
                if sample.ground_truth is None and getattr(
                    metric, "requires_ground_truth", True
                ):
                    continue
                value = float(metric.compute(prediction, sample.ground_truth))
                if math.isnan(value):
                    continue
                scene_metrics[metric.name] = value
                metric_values[metric.name].append(value)
            records.append(
                {"sample_id": sample.sample_id, "metrics": scene_metrics}
            )

        metrics_path = output_dir / "metrics.jsonl"
        metrics_path.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )

        summary: dict[str, dict[str, Any]] = {
            metric_name: _aggregate(values, self.config.seed)
            for metric_name, values in metric_values.items()
        }
        labelled = [
            (prediction, ground_truth)
            for prediction, ground_truth in zip(predictions, ground_truths)
            if ground_truth is not None
        ]
        for metric in dataset_metrics:
            value = float(
                metric.compute_dataset(
                    [prediction for prediction, _ in labelled],
                    [ground_truth for _, ground_truth in labelled],
                )
            )
            summary[metric.name] = {
                "count": len(labelled),
                "value": value,
                "dataset_level": True,
            }

        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        run_manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_revision": git_revision(),
            "seed": self.config.seed,
            "scene_count": len(records),
            "config": self.config.to_dict(),
            "estimator": self.estimator.provenance(),
        }
        (output_dir / "run.json").write_text(
            json.dumps(run_manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self._write_report(output_dir, summary)
        return EvaluationResult(
            output_dir=output_dir,
            scene_count=len(records),
            summary=summary,
        )
