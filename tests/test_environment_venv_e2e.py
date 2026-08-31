import tempfile
import unittest
from pathlib import Path

from benchmark_toolbox.domain import SceneSample
from benchmark_toolbox.environments import load_environment_spec
from benchmark_toolbox.environments.venv import VenvEnvironmentManager
from benchmark_toolbox.models.subprocess import SubprocessSceneEstimator

REPO = Path(__file__).resolve().parents[1]


class VenvEndToEndTest(unittest.TestCase):
    def test_prepare_and_run_echo_environment(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            spec = load_environment_spec(
                {
                    "name": "echo",
                    "backend": "venv",
                    "variant": "cpu",
                    "workspace": workspace,
                    "runner": {"entry": str(REPO / "runners/echo_runner.py")},
                },
                base_dir=REPO,
            )
            manager = VenvEnvironmentManager()
            self.assertFalse(manager.is_prepared(spec))

            manager.prepare(spec)
            self.assertTrue(manager.is_prepared(spec))

            estimator = SubprocessSceneEstimator({"environment": spec})

            provenance = estimator.provenance()
            self.assertEqual(provenance["environment"], "echo")
            self.assertEqual(provenance["backend"], "venv")
            self.assertIn("fingerprint", provenance)

            output = estimator.predict(
                SceneSample(sample_id="scene-x", input_path=Path(__file__))
            )

            self.assertIsNotNone(output.layout)
            self.assertEqual(len(output.objects), 1)
            self.assertEqual(output.metadata["sample_id"], "scene-x")
            self.assertEqual(output.metadata["environment"], "echo")


if __name__ == "__main__":
    unittest.main()
