import importlib.util
import math
import unittest
from pathlib import Path

from benchmark_toolbox.domain import OrientedBox3D

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "holopano_to_fixtures.py"
_spec = importlib.util.spec_from_file_location("holopano_to_fixtures", _SCRIPT)
holopano_to_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(holopano_to_fixtures)


def reference_corners(centroid, size, yaw):
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    corners = []
    for sign_x in (-0.5, 0.5):
        for sign_y in (-0.5, 0.5):
            for sign_z in (-0.5, 0.5):
                local = (sign_x * size[0], sign_y * size[1], sign_z * size[2])
                corners.append(
                    (
                        centroid[0] + cos_yaw * local[0] - sin_yaw * local[1],
                        centroid[1] + sin_yaw * local[0] + cos_yaw * local[1],
                        centroid[2] + local[2],
                    )
                )
    return sorted(tuple(round(value, 9) for value in corner) for corner in corners)


class HoloPanoFixtureGeometryTest(unittest.TestCase):
    def test_basis_matches_holopano_corner_convention(self) -> None:
        cases = [
            ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 0.0),
            ((1.5, -2.0, 0.8), (0.6, 1.2, 0.9), math.pi / 2),
            ((-3.0, 0.25, 1.1), (2.0, 0.5, 1.7), 0.7),
            ((0.4, 0.4, 0.4), (1.0, 3.0, 0.2), -2.3),
        ]
        for centroid, size, yaw in cases:
            with self.subTest(yaw=yaw):
                box = OrientedBox3D.from_dict(
                    holopano_to_fixtures._oriented_box(centroid, size, yaw)
                )
                produced = sorted(
                    tuple(round(value, 9) for value in corner)
                    for corner in box.corners()
                )
                for produced_corner, expected_corner in zip(
                    produced, reference_corners(centroid, size, yaw)
                ):
                    for produced_value, expected_value in zip(
                        produced_corner, expected_corner
                    ):
                        self.assertAlmostEqual(produced_value, expected_value, places=7)

    def test_identical_boxes_have_iou_one_and_rotated_ones_do_not(self) -> None:
        first = OrientedBox3D.from_dict(
            holopano_to_fixtures._oriented_box((0.0, 0.0, 0.0), (2.0, 1.0, 1.0), 0.0)
        )
        same = OrientedBox3D.from_dict(
            holopano_to_fixtures._oriented_box((0.0, 0.0, 0.0), (2.0, 1.0, 1.0), 0.0)
        )
        rotated = OrientedBox3D.from_dict(
            holopano_to_fixtures._oriented_box(
                (0.0, 0.0, 0.0), (2.0, 1.0, 1.0), math.pi / 2
            )
        )

        self.assertAlmostEqual(first.iou(same), 1.0)
        self.assertAlmostEqual(first.iou(rotated), 1.0 / 3.0, places=6)


if __name__ == "__main__":
    unittest.main()
