import json
import sys
import tempfile
import unittest
from pathlib import Path

from benchmark_toolbox.config import ComponentConfig, ExperimentConfig
from benchmark_toolbox.datasets.base import BaseDatasetLoader
from benchmark_toolbox.domain import SceneSample
from benchmark_toolbox.environments.spec import load_environment_spec
from benchmark_toolbox.evaluator import Evaluator
from benchmark_toolbox.metrics.geometry import CollisionRate
from benchmark_toolbox.models.base import BaseSceneEstimator
from benchmark_toolbox.models.subprocess import SubprocessSceneEstimator

RUNNERS = Path(__file__).resolve().parents[1] / "runners"
sys.path.insert(0, str(RUNNERS))
import _common as rc  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures/fake_model_runner.py"


def _estimator(batch: bool) -> SubprocessSceneEstimator:
    return SubprocessSceneEstimator(
        {
            "command": [
                sys.executable,
                str(FIXTURE),
                "--request={request}",
                "--output={output}",
            ],
            "environment_name": "test-environment",
            "batch": batch,
        }
    )


def _samples(count: int) -> list[SceneSample]:
    return [
        SceneSample(sample_id=f"scene-{index:03d}", input_path=Path(__file__))
        for index in range(count)
    ]


class RequestShapeTest(unittest.TestCase):
    def test_read_requests_accepts_a_single_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.json"
            path.write_text(json.dumps({"sample_id": "a"}), encoding="utf-8")
            self.assertEqual(rc.read_requests(str(path)), [{"sample_id": "a"}])

    def test_read_requests_accepts_a_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.json"
            path.write_text(
                json.dumps({"samples": [{"sample_id": "a"}, {"sample_id": "b"}]}),
                encoding="utf-8",
            )
            self.assertEqual(
                [item["sample_id"] for item in rc.read_requests(str(path))], ["a", "b"]
            )


class BatchEstimatorTest(unittest.TestCase):
    def test_batch_uses_a_single_launch_for_every_sample(self) -> None:
        samples = _samples(4)
        outputs = list(_estimator(batch=True).predict_many(samples))

        self.assertEqual(
            [output.metadata["sample_id"] for output in outputs],
            [sample.sample_id for sample in samples],
        )
        launches = {output.metadata["launch"] for output in outputs}
        self.assertEqual(len(launches), 1, "the batch was split across processes")

    def test_without_batch_each_sample_costs_its_own_launch(self) -> None:
        outputs = list(_estimator(batch=False).predict_many(_samples(4)))
        launches = {output.metadata["launch"] for output in outputs}
        self.assertEqual(len(launches), 4)

    def test_results_are_identical_either_way(self) -> None:
        samples = _samples(3)
        batched = list(_estimator(batch=True).predict_many(samples))
        one_by_one = list(_estimator(batch=False).predict_many(samples))
        for left, right in zip(batched, one_by_one):
            self.assertEqual(left.layout, right.layout)
            self.assertEqual(left.objects, right.objects)
            self.assertEqual(left.metadata["sample_id"], right.metadata["sample_id"])

    def test_answers_are_matched_by_sample_id_not_by_position(self) -> None:
        samples = _samples(3)
        outputs = list(_estimator(batch=True).predict_many(list(reversed(samples))))
        self.assertEqual(
            [output.metadata["sample_id"] for output in outputs],
            [sample.sample_id for sample in reversed(samples)],
        )

    def test_a_dropped_sample_is_an_error_not_a_silent_gap(self) -> None:
        script = """
import argparse, json
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("--request", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
payload = json.loads(Path(args.request).read_text())
# answers only the first sample
first = payload["samples"][0]
Path(args.output).write_text(json.dumps({"outputs": [
    {"layout": None, "objects": [], "relations": [],
     "metadata": {"sample_id": first["sample_id"]}}
]}))
"""
        with tempfile.TemporaryDirectory() as tmp:
            runner = Path(tmp) / "partial_runner.py"
            runner.write_text(script, encoding="utf-8")
            estimator = SubprocessSceneEstimator(
                {
                    "command": [
                        sys.executable,
                        str(runner),
                        "--request={request}",
                        "--output={output}",
                    ],
                    "batch": True,
                }
            )
            with self.assertRaisesRegex(RuntimeError, "no prediction for 2"):
                list(estimator.predict_many(_samples(3)))


class RunnerArgsTest(unittest.TestCase):
    def _command(self, runner_args) -> tuple:
        repo = Path(__file__).resolve().parents[1]
        spec = load_environment_spec(repo / "configs/environments/echo.yaml")
        estimator = SubprocessSceneEstimator(
            {"environment": spec, "runner_args": runner_args}
        )
        return estimator.command

    def test_runner_args_are_appended_after_the_specs_own(self) -> None:
        command = self._command(["--pred-root", "out/pano3d_im3d"])
        self.assertIn("--pred-root", command)
        self.assertEqual(
            command[command.index("--pred-root") + 1], "out/pano3d_im3d"
        )
        self.assertLess(command.index("--pred-root"), command.index("--request"))

    def test_runner_args_are_recorded_in_provenance(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        spec = load_environment_spec(repo / "configs/environments/echo.yaml")
        estimator = SubprocessSceneEstimator(
            {"environment": spec, "runner_args": ["--model", "im3d-pano"]}
        )
        self.assertEqual(
            estimator.provenance()["runner_args"], ["--model", "im3d-pano"]
        )

    def test_a_repeated_option_is_won_by_the_experiment_config(self) -> None:
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--config", default="base.yaml")
        args, _ = parser.parse_known_args(
            ["--config", "configs/pano3d_igibson.yaml"]
            + ["--config", "configs/pano3d_igibson_im3d.yaml"]
        )
        self.assertEqual(args.config, "configs/pano3d_igibson_im3d.yaml")

    def test_shipped_method_configs_select_distinct_prediction_roots(self) -> None:
        from benchmark_toolbox.config import load_experiment_config

        examples = Path(__file__).resolve().parents[1] / "configs/examples"
        roots = {}
        for path in examples.glob("*.yaml"):
            config = load_experiment_config(path)
            arguments = list(config.model.parameters.get("runner_args", ()))
            if "--pred-root" in arguments:
                roots[path.name] = arguments[arguments.index("--pred-root") + 1]
        self.assertTrue(roots, "no method config passes a prediction root")
        self.assertEqual(
            len(set(roots.values())), len(roots), f"prediction roots collide: {roots}"
        )


class BatchSpecTest(unittest.TestCase):
    def test_spec_declares_batch_support(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        self.assertTrue(
            load_environment_spec(repo / "configs/environments/dpc.yaml").runner.batch
        )
        self.assertTrue(
            load_environment_spec(repo / "configs/environments/echo.yaml").runner.batch
        )
        self.assertFalse(
            load_environment_spec(
                repo / "configs/environments/holopano.yaml"
            ).runner.batch
        )


class _ListDataset(BaseDatasetLoader):
    def __init__(self, samples):
        self._samples = samples

    def samples(self):
        return iter(self._samples)


class _CountingEstimator(BaseSceneEstimator):
    """Records how the evaluator asks for predictions."""

    def __init__(self):
        self.batches: list[int] = []

    def predict(self, sample):
        raise AssertionError("the evaluator must go through predict_many")

    def predict_many(self, samples):
        from benchmark_toolbox.domain import SceneOutput

        self.batches.append(len(samples))
        return [SceneOutput(layout=None) for _ in samples]


class _StreamWatchingEstimator(BaseSceneEstimator):
    def __init__(self, predictions_dir: Path):
        self.predictions_dir = predictions_dir
        self.seen: list[int] = []

    def predict(self, sample):
        from benchmark_toolbox.domain import SceneOutput

        existing = (
            len(list(self.predictions_dir.glob("*.json")))
            if self.predictions_dir.exists()
            else 0
        )
        self.seen.append(existing)
        return SceneOutput(layout=None)


class EvaluatorBatchTest(unittest.TestCase):
    def test_predictions_are_written_as_they_are_produced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            estimator = _StreamWatchingEstimator(output_dir / "predictions")
            config = ExperimentConfig(
                experiment_name="t",
                seed=42,
                output_dir=output_dir,
                model=ComponentConfig(type="x", parameters={}),
                dataset=ComponentConfig(type="x", parameters={}),
                metrics=(),
                source_path=output_dir / "c.json",
            )
            Evaluator(
                config, estimator, _ListDataset(_samples(4)), [CollisionRate()]
            ).run()

        self.assertEqual(estimator.seen, [0, 1, 2, 3])

    def test_evaluator_offers_the_whole_set_to_the_estimator(self) -> None:
        estimator = _CountingEstimator()
        samples = _samples(5)
        with tempfile.TemporaryDirectory() as tmp:
            config = ExperimentConfig(
                experiment_name="t",
                seed=42,
                output_dir=Path(tmp),
                model=ComponentConfig(type="x", parameters={}),
                dataset=ComponentConfig(type="x", parameters={}),
                metrics=(),
                source_path=Path(tmp) / "c.json",
            )
            result = Evaluator(
                config, estimator, _ListDataset(samples), [CollisionRate()]
            ).run()

        self.assertEqual(estimator.batches, [5])
        self.assertEqual(result.scene_count, 5)


if __name__ == "__main__":
    unittest.main()
