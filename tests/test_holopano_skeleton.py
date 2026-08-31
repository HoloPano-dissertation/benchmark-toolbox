import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "runners"))
import holopano_runner  # noqa: E402

from benchmark_toolbox.environments import load_environment_spec  # noqa: E402
from benchmark_toolbox.models import BaseSceneEstimator  # noqa: E402


class HoloPanoSkeletonTest(unittest.TestCase):
    def test_estimator_registered(self) -> None:
        self.assertIn("holopano", BaseSceneEstimator.registry.names)

    def test_env_spec_loads(self) -> None:
        spec = load_environment_spec(REPO / "configs/environments/holopano.yaml")
        self.assertEqual(spec.name, "HoloPano")
        self.assertEqual(spec.backend, "conda")
        self.assertEqual(spec.runner.entry, "runners/holopano_runner.py")

    def test_converter_handles_dpc_like_output(self) -> None:
        native = {
            "objects": [
                {
                    "label": "sofa",
                    "score": 0.8,
                    "bbox3d": {
                        "centroid": [0, 0, 0],
                        "basis": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        "coeffs": [1, 2, 3],
                    },
                }
            ],
            "layout": [[0, 0, 0], [5, 3, 4]],
        }
        scene = holopano_runner.to_scene_output(native)
        self.assertEqual(len(scene["objects"]), 1)
        self.assertEqual(scene["objects"][0]["label"], "sofa")
        bbox = scene["objects"][0]["bbox"]
        self.assertEqual(bbox["center"], [0.0, 0.0, 0.0])
        self.assertEqual(bbox["size"], [2.0, 4.0, 6.0])
        self.assertEqual(scene["layout"]["max_corner"], [5.0, 3.0, 4.0])


if __name__ == "__main__":
    unittest.main()
