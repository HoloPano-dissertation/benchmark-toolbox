import unittest

from benchmark_toolbox.domain import BoundingBox3D, SceneOutput


class SceneOutputRoundTripTest(unittest.TestCase):
    def test_scene_output_round_trip(self) -> None:
        original = SceneOutput(layout=BoundingBox3D((0, 0, 0), (1, 2, 3)))

        restored = SceneOutput.from_dict(original.to_dict())

        self.assertEqual(restored, original)


if __name__ == "__main__":
    unittest.main()
