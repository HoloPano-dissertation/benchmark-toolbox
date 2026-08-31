import math
import tempfile
import unittest
from pathlib import Path

from benchmark_toolbox.config import ComponentConfig, ExperimentConfig
from benchmark_toolbox.datasets.base import BaseDatasetLoader
from benchmark_toolbox.domain import BoundingBox3D, SceneObject, SceneOutput, SceneSample
from benchmark_toolbox.evaluator import Evaluator, _aggregate
from benchmark_toolbox.metrics.base import BaseMetric
from benchmark_toolbox.metrics.geometry import (
    CollisionRate,
    LayoutIoU3D,
    ObjectDetectionMAPDataset,
)
from benchmark_toolbox.models.base import BaseSceneEstimator


class _NaNWhenEmptyMetric(BaseMetric):
    name = "nan_when_empty"
    requires_ground_truth = False

    def compute(self, prediction, ground_truth):
        return 1.0 if prediction.objects else math.nan


class _SequenceDataset(BaseDatasetLoader):
    def __init__(self, samples):
        self._samples = samples

    def samples(self):
        return iter(self._samples)


class _PerSampleEstimator(BaseSceneEstimator):
    def __init__(self, outputs):
        self._outputs = outputs

    def predict(self, sample):
        return self._outputs[sample.sample_id]


class _FixedEstimator(BaseSceneEstimator):
    def __init__(self, output: SceneOutput) -> None:
        self._output = output

    def predict(self, sample: SceneSample) -> SceneOutput:
        return self._output


class _ListDataset(BaseDatasetLoader):
    def __init__(self, samples: list[SceneSample]) -> None:
        self._samples = samples

    def samples(self):
        return iter(self._samples)


def _config(tmp: str) -> ExperimentConfig:
    dummy = ComponentConfig(type="x", parameters={})
    return ExperimentConfig(
        experiment_name="t",
        seed=42,
        output_dir=Path(tmp),
        model=dummy,
        dataset=dummy,
        metrics=(),
        source_path=Path(tmp) / "c.json",
    )


class EvaluatorMetricsTest(unittest.TestCase):
    def test_duplicate_metric_names_are_rejected(self) -> None:
        metrics = [
            ObjectDetectionMAPDataset({"iou_threshold": 0.15, "classes": ["chair"]}),
            ObjectDetectionMAPDataset({"iou_threshold": 0.15, "classes": ["sofa"]}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                Evaluator(
                    _config(tmp),
                    _FixedEstimator(SceneOutput(layout=None)),
                    _ListDataset([]),
                    metrics,
                )

    def test_evaluator_skips_metrics_without_ground_truth(self) -> None:
        prediction = SceneOutput(
            layout=BoundingBox3D((0, 0, 0), (1, 1, 1)),
            objects=(
                SceneObject("a", "chair", BoundingBox3D((0, 0, 0), (1, 1, 1))),
                SceneObject("b", "chair", BoundingBox3D((0, 0, 0), (1, 1, 1))),
            ),
        )
        sample = SceneSample(
            sample_id="s1", input_path=Path(__file__), ground_truth=None
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = Evaluator(
                _config(tmp),
                _FixedEstimator(prediction),
                _ListDataset([sample]),
                [LayoutIoU3D(), CollisionRate()],
            ).run()
            self.assertEqual(result.summary["layout_iou_3d"]["count"], 0)
            self.assertEqual(result.summary["collision_rate"]["count"], 1)
            self.assertEqual(result.summary["collision_rate"]["mean"], 1.0)
            self.assertNotIn("mean", result.summary["layout_iou_3d"])
            report = (Path(tmp) / "report.md").read_text(encoding="utf-8")
            self.assertIn("| layout_iou_3d | 0 | n/a | n/a | n/a |", report)

    def test_layout_iou_is_not_applicable_without_ground_truth_layout(self) -> None:
        boxed = SceneOutput(layout=BoundingBox3D((0, 0, 0), (1, 1, 1)))
        layout_free = SceneOutput(layout=None)
        self.assertTrue(math.isnan(LayoutIoU3D().compute(boxed, layout_free)))
        with self.assertRaises(ValueError):
            LayoutIoU3D().compute(boxed, None)

    def test_evaluator_excludes_nan_from_aggregation(self) -> None:
        with_object = SceneOutput(
            layout=None,
            objects=(SceneObject("a", "chair", BoundingBox3D((0, 0, 0), (1, 1, 1))),),
        )
        empty = SceneOutput(layout=None, objects=())
        samples = [
            SceneSample("s1", Path(__file__), ground_truth=with_object),
            SceneSample("s2", Path(__file__), ground_truth=empty),
        ]
        estimator = _PerSampleEstimator({"s1": with_object, "s2": empty})
        with tempfile.TemporaryDirectory() as tmp:
            result = Evaluator(
                _config(tmp),
                estimator,
                _SequenceDataset(samples),
                [_NaNWhenEmptyMetric()],
            ).run()
            self.assertEqual(result.summary["nan_when_empty"]["count"], 1)
            self.assertEqual(result.summary["nan_when_empty"]["mean"], 1.0)

    def test_dataset_map_perfect_predictions(self) -> None:
        objects = (
            SceneObject("c", "chair", BoundingBox3D((0, 0, 0), (1, 1, 1)), 0.9),
            SceneObject("t", "table", BoundingBox3D((2, 0, 0), (3, 1, 1)), 0.8),
        )
        scene = SceneOutput(layout=None, objects=objects)
        metric = ObjectDetectionMAPDataset({"iou_threshold": 0.5})
        self.assertEqual(metric.compute_dataset([scene, scene], [scene, scene]), 1.0)

    def test_aggregate_has_bootstrap_ci(self) -> None:
        aggregate = _aggregate([0.5, 0.5, 0.5, 0.5], seed=42)
        self.assertEqual(aggregate["count"], 4)
        self.assertAlmostEqual(aggregate["mean"], 0.5)
        self.assertAlmostEqual(aggregate["ci95_low"], 0.5)
        self.assertAlmostEqual(aggregate["ci95_high"], 0.5)

    def test_aggregate_single_value(self) -> None:
        aggregate = _aggregate([0.7], seed=1)
        self.assertEqual(aggregate["ci95_low"], aggregate["ci95_high"])


if __name__ == "__main__":
    unittest.main()
