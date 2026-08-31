import tempfile
import unittest
from pathlib import Path

from benchmark_toolbox.config import load_experiment_config
from benchmark_toolbox.datasets import BaseDatasetLoader
from benchmark_toolbox.evaluator import Evaluator
from benchmark_toolbox.metrics import BaseMetric
from benchmark_toolbox.models import BaseSceneEstimator


class EvaluatorSmokeTest(unittest.TestCase):
    def test_example_pipeline(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        config = load_experiment_config(
            repository_root / "configs/examples/smoke.json"
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = config.__class__(
                experiment_name=config.experiment_name,
                seed=config.seed,
                output_dir=Path(temporary_dir),
                model=config.model,
                dataset=config.dataset,
                metrics=config.metrics,
                source_path=config.source_path,
            )
            estimator = BaseSceneEstimator.create(
                config.model.type, dict(config.model.parameters)
            )
            dataset = BaseDatasetLoader.create(
                config.dataset.type, dict(config.dataset.parameters)
            )
            metrics = [
                BaseMetric.create(metric.type, dict(metric.parameters))
                for metric in config.metrics
            ]

            result = Evaluator(config, estimator, dataset, metrics).run()

            self.assertEqual(result.scene_count, 1)
            self.assertEqual(result.summary["layout_iou_3d"]["mean"], 1.0)
            self.assertEqual(result.summary["object_map"]["mean"], 1.0)
            self.assertEqual(result.summary["collision_rate"]["mean"], 0.0)
            self.assertTrue((Path(temporary_dir) / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
