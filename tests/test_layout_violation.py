import math
import unittest

from benchmark_toolbox.domain import (
    BoundingBox3D,
    OrientedBox3D,
    SceneObject,
    SceneOutput,
)
from benchmark_toolbox.metrics.geometry import LayoutPenetration, LayoutViolationRate

IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
ROOM = BoundingBox3D((0, 0, 0), (10, 10, 3))


def _scene(*objects) -> SceneOutput:
    return SceneOutput(layout=ROOM, objects=tuple(objects))


def _obj(object_id, box) -> SceneObject:
    return SceneObject(object_id, "thing", box)


class LayoutViolationRateTest(unittest.TestCase):
    def test_object_inside_room_is_not_a_violation(self) -> None:
        scene = _scene(_obj("a", BoundingBox3D((1, 1, 0), (2, 2, 1))))
        self.assertEqual(LayoutViolationRate().compute(scene, None), 0.0)

    def test_object_through_wall_is_a_violation(self) -> None:
        scene = _scene(_obj("a", BoundingBox3D((9, 9, 0), (12, 11, 1))))
        self.assertEqual(LayoutViolationRate().compute(scene, None), 1.0)

    def test_object_through_ceiling_is_a_violation(self) -> None:
        scene = _scene(_obj("a", BoundingBox3D((4, 4, 0), (6, 6, 4))))
        self.assertEqual(LayoutViolationRate().compute(scene, None), 1.0)

    def test_mixed_objects_give_fraction(self) -> None:
        scene = _scene(
            _obj("inside", BoundingBox3D((1, 1, 0), (2, 2, 1))),
            _obj("outside", BoundingBox3D((9, 9, 0), (12, 11, 1))),
        )
        self.assertEqual(LayoutViolationRate().compute(scene, None), 0.5)

    def test_tolerance_forgives_small_protrusion(self) -> None:
        scene = _scene(_obj("a", BoundingBox3D((4, 4, 0), (6, 6, 4))))
        self.assertEqual(LayoutViolationRate({"tolerance": 0.3}).compute(scene, None), 0.0)
        self.assertEqual(LayoutViolationRate({"tolerance": 0.0}).compute(scene, None), 1.0)

    def test_no_layout_is_not_applicable(self) -> None:
        scene = SceneOutput(
            layout=None, objects=(_obj("a", BoundingBox3D((0, 0, 0), (1, 1, 1))),)
        )
        self.assertTrue(math.isnan(LayoutViolationRate().compute(scene, None)))

    def test_oriented_object_outside_room_is_a_violation(self) -> None:
        box = OrientedBox3D((9.5, 9.5, 1.5), (2, 2, 2), IDENTITY)
        scene = _scene(_obj("a", box))
        self.assertEqual(LayoutViolationRate().compute(scene, None), 1.0)

    def test_rejects_out_of_range_tolerance(self) -> None:
        with self.assertRaises(ValueError):
            LayoutViolationRate({"tolerance": 1.0})


class LayoutPenetrationTest(unittest.TestCase):
    def test_object_inside_has_zero_penetration(self) -> None:
        scene = _scene(_obj("a", BoundingBox3D((1, 1, 0), (2, 2, 1))))
        self.assertEqual(LayoutPenetration().compute(scene, None), 0.0)

    def test_penetration_is_outside_volume_fraction(self) -> None:
        scene = _scene(_obj("a", BoundingBox3D((4, 4, 0), (6, 6, 4))))
        self.assertAlmostEqual(LayoutPenetration().compute(scene, None), 0.25, places=6)

    def test_penetration_averages_over_objects(self) -> None:
        scene = _scene(
            _obj("inside", BoundingBox3D((1, 1, 0), (2, 2, 1))),
            _obj("quarter", BoundingBox3D((4, 4, 0), (6, 6, 4))),
        )
        self.assertAlmostEqual(
            LayoutPenetration().compute(scene, None), 0.125, places=6
        )

    def test_no_layout_is_not_applicable(self) -> None:
        scene = SceneOutput(
            layout=None, objects=(_obj("a", BoundingBox3D((0, 0, 0), (1, 1, 1))),)
        )
        self.assertTrue(math.isnan(LayoutPenetration().compute(scene, None)))


if __name__ == "__main__":
    unittest.main()
