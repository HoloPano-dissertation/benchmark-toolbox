import math
import sys
import unittest
from pathlib import Path

RUNNERS = Path(__file__).resolve().parents[1] / "runners"
sys.path.insert(0, str(RUNNERS))
import _common as rc  # noqa: E402


class BoxGeometryTest(unittest.TestCase):
    def test_identity_box(self) -> None:
        box = rc.oriented_box_to_axis_aligned(
            [0, 0, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [1, 2, 3]
        )
        self.assertEqual(box["min_corner"], [-1.0, -2.0, -3.0])
        self.assertEqual(box["max_corner"], [1.0, 2.0, 3.0])

    def test_rotation_45_about_z_grows_footprint(self) -> None:
        cos = math.cos(math.pi / 4)
        sin = math.sin(math.pi / 4)
        basis = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
        box = rc.oriented_box_to_axis_aligned([0, 0, 0], basis, [1, 1, 5])
        self.assertAlmostEqual(box["max_corner"][0], math.sqrt(2), places=5)
        self.assertAlmostEqual(box["max_corner"][1], math.sqrt(2), places=5)
        self.assertAlmostEqual(box["max_corner"][2], 5.0, places=5)

    def test_corners_to_axis_aligned(self) -> None:
        box = rc.corners_to_axis_aligned([[0, 0, 0], [1, 2, 3], [-1, 0, 1]])
        self.assertEqual(box["min_corner"], [-1.0, 0.0, 0.0])
        self.assertEqual(box["max_corner"], [1.0, 2.0, 3.0])

    def test_scene_from_oriented_objects(self) -> None:
        objects = [
            {
                "label": "chair",
                "centroid": [0, 0, 0],
                "basis": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "coeffs": [1, 1, 1],
            }
        ]
        scene = rc.scene_from_oriented_objects(
            objects, layout_corners=[[0, 0, 0], [4, 3, 5]]
        )
        self.assertEqual(len(scene["objects"]), 1)
        self.assertEqual(scene["objects"][0]["label"], "chair")
        self.assertEqual(scene["layout"]["max_corner"], [4.0, 3.0, 5.0])


if __name__ == "__main__":
    unittest.main()
