import sys
import unittest
from pathlib import Path

from benchmark_toolbox.domain import BoundingBox3D, SceneSample
from benchmark_toolbox.models.subprocess import (
    SubprocessSceneEstimator,
    _usable_wrapper,
)


class UsableWrapperTest(unittest.TestCase):
    def test_keeps_wrapper_when_executable_present(self) -> None:
        self.assertEqual(_usable_wrapper(["env", "-i"]), ("env", "-i"))

    def test_drops_wrapper_when_executable_missing(self) -> None:
        self.assertEqual(
            _usable_wrapper(["benchmark-toolbox-no-such-binary-xyz", "-a"]),
            (),
        )

    def test_empty_wrapper(self) -> None:
        self.assertEqual(_usable_wrapper([]), ())


class SubprocessSceneEstimatorTest(unittest.TestCase):
    def test_json_protocol(self) -> None:
        fixture = Path(__file__).parent / "fixtures/fake_model_runner.py"
        estimator = SubprocessSceneEstimator(
            {
                "command": [
                    sys.executable,
                    str(fixture),
                    "--request={request}",
                    "--output={output}",
                ],
                "environment_name": "test-environment",
            }
        )
        sample = SceneSample(
            sample_id="scene-001",
            input_path=Path(__file__),
        )

        output = estimator.predict(sample)

        self.assertEqual(
            output.layout,
            BoundingBox3D((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        self.assertEqual(output.metadata["sample_id"], "scene-001")
        self.assertEqual(output.metadata["environment"], "test-environment")


if __name__ == "__main__":
    unittest.main()
