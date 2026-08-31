import math
import unittest

from benchmark_toolbox.domain import (
    BoundingBox3D,
    OrientedBox3D,
    SceneObject,
    SceneOutput,
    parse_box,
)
from benchmark_toolbox.metrics.geometry import ObjectDetectionMAP

IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _yaw_basis(angle: float):
    cos = math.cos(angle)
    sin = math.sin(angle)
    return ((cos, sin, 0.0), (-sin, cos, 0.0), (0.0, 0.0, 1.0))


class OrientedBox3DTest(unittest.TestCase):
    def test_identity_basis_matches_axis_aligned_iou(self) -> None:
        left = BoundingBox3D((0, 0, 0), (2, 2, 2)).as_oriented()
        right = BoundingBox3D((1, 1, 1), (3, 3, 3)).as_oriented()
        self.assertAlmostEqual(left.iou(right), 1 / 15, places=9)

    def test_self_iou_is_one(self) -> None:
        box = OrientedBox3D((1, 2, 3), (2, 4, 6), _yaw_basis(0.37))
        self.assertAlmostEqual(box.iou(box), 1.0, places=9)

    def test_quarter_turn_of_square_footprint_is_identical(self) -> None:
        upright = OrientedBox3D((0, 0, 0), (2, 2, 4), IDENTITY)
        turned = OrientedBox3D((0, 0, 0), (2, 2, 4), _yaw_basis(math.pi / 2))
        self.assertAlmostEqual(upright.iou(turned), 1.0, places=6)

    def test_concentric_45_degree_rotation_exact_value(self) -> None:
        upright = OrientedBox3D((0, 0, 0), (2, 2, 2), IDENTITY)
        turned = OrientedBox3D((0, 0, 0), (2, 2, 2), _yaw_basis(math.pi / 4))
        self.assertAlmostEqual(upright.iou(turned), 1 / math.sqrt(2), places=5)

    def test_no_vertical_overlap_gives_zero(self) -> None:
        low = OrientedBox3D((0, 0, 0), (2, 2, 2), IDENTITY)
        high = OrientedBox3D((0, 0, 10), (2, 2, 2), IDENTITY)
        self.assertEqual(low.iou(high), 0.0)

    def test_disjoint_footprints_give_zero(self) -> None:
        here = OrientedBox3D((0, 0, 0), (2, 2, 2), IDENTITY)
        far = OrientedBox3D((10, 0, 0), (2, 2, 2), IDENTITY)
        self.assertEqual(here.iou(far), 0.0)

    def test_cross_type_iou_is_symmetric(self) -> None:
        aabb = BoundingBox3D((0, 0, 0), (2, 2, 2))
        oriented = OrientedBox3D((1, 1, 1), (2, 2, 2), _yaw_basis(math.pi / 6))
        self.assertAlmostEqual(aabb.iou(oriented), oriented.iou(aabb), places=9)
        self.assertGreater(aabb.iou(oriented), 0.0)

    def test_iou_stays_within_unit_interval(self) -> None:
        a = OrientedBox3D((0, 0, 0), (3, 1, 2), _yaw_basis(0.9))
        b = OrientedBox3D((0.5, 0.2, 0.1), (1, 3, 2), _yaw_basis(-0.4))
        value = a.iou(b)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_dict_round_trip(self) -> None:
        box = OrientedBox3D((1, 2, 3), (2, 4, 6), _yaw_basis(0.5))
        restored = OrientedBox3D.from_dict(box.to_dict())
        self.assertEqual(restored, box)

    def test_parse_box_dispatches_on_keys(self) -> None:
        oriented = parse_box(
            {"center": [0, 0, 0], "size": [1, 1, 1], "basis": IDENTITY}
        )
        axis_aligned = parse_box({"min_corner": [0, 0, 0], "max_corner": [1, 1, 1]})
        self.assertIsInstance(oriented, OrientedBox3D)
        self.assertIsInstance(axis_aligned, BoundingBox3D)

    def test_degenerate_size_becomes_zero_volume(self) -> None:
        box = OrientedBox3D((0, 0, 0), (-1, 1, 1), IDENTITY)
        self.assertEqual(box.size[0], 0.0)
        self.assertEqual(box.volume, 0.0)
        self.assertEqual(box.iou(OrientedBox3D((0, 0, 0), (2, 2, 2), IDENTITY)), 0.0)

    def test_non_finite_values_are_sanitized(self) -> None:
        nan = float("nan")
        box = OrientedBox3D((nan, 0, 0), (nan, 2, 2), IDENTITY)
        self.assertEqual(box.center[0], 0.0)
        self.assertEqual(box.size[0], 0.0)
        self.assertEqual(box.volume, 0.0)


class OrientedMetricTest(unittest.TestCase):
    def _scene(self, box) -> SceneOutput:
        return SceneOutput(
            layout=None,
            objects=(SceneObject("o", "chair", box, score=0.9),),
        )

    def test_object_map_uses_oriented_iou_at_protocol_threshold(self) -> None:
        ground_truth = self._scene(OrientedBox3D((0, 0, 0), (2, 2, 2), IDENTITY))
        prediction = self._scene(
            OrientedBox3D((0, 0, 0), (2, 2, 2), _yaw_basis(math.pi / 4))
        )
        matched = ObjectDetectionMAP({"iou_threshold": 0.15})
        missed = ObjectDetectionMAP({"iou_threshold": 0.8})
        self.assertEqual(matched.compute(prediction, ground_truth), 1.0)
        self.assertEqual(missed.compute(prediction, ground_truth), 0.0)


if __name__ == "__main__":
    unittest.main()
