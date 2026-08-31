import unittest

from benchmark_toolbox.domain import BoundingBox3D, SceneObject, SceneOutput
from benchmark_toolbox.metrics.geometry import (
    CollisionRate,
    LayoutIoU3D,
    ObjectDetectionMAP,
    ObjectDetectionMAPDataset,
)


def scene_object(object_id: str, label: str, minimum, maximum, score=1.0):
    return SceneObject(
        object_id=object_id,
        label=label,
        score=score,
        bbox=BoundingBox3D(minimum, maximum),
    )


class GeometryMetricsTest(unittest.TestCase):
    def test_layout_iou(self) -> None:
        prediction = SceneOutput(layout=BoundingBox3D((0, 0, 0), (2, 2, 2)))
        ground_truth = SceneOutput(layout=BoundingBox3D((1, 1, 1), (3, 3, 3)))

        value = LayoutIoU3D().compute(prediction, ground_truth)

        self.assertAlmostEqual(value, 1 / 15)

    def test_collision_rate_uses_object_pairs(self) -> None:
        prediction = SceneOutput(
            layout=None,
            objects=(
                scene_object("a", "chair", (0, 0, 0), (2, 2, 2)),
                scene_object("b", "chair", (1, 1, 1), (3, 3, 3)),
                scene_object("c", "table", (5, 5, 5), (6, 6, 6)),
            ),
        )

        value = CollisionRate().compute(prediction, None)

        self.assertAlmostEqual(value, 1 / 3)

    def test_object_map_is_one_for_perfect_predictions(self) -> None:
        objects = (
            scene_object("chair", "chair", (0, 0, 0), (1, 1, 1), 0.9),
            scene_object("table", "table", (2, 0, 0), (3, 1, 1), 0.8),
        )
        prediction = SceneOutput(layout=None, objects=objects)
        ground_truth = SceneOutput(layout=None, objects=objects)

        value = ObjectDetectionMAP({"iou_threshold": 0.5}).compute(
            prediction, ground_truth
        )

        self.assertEqual(value, 1.0)

    def test_object_map_partial_with_false_positive(self) -> None:
        ground_truth = SceneOutput(
            layout=None,
            objects=(
                scene_object("a", "chair", (0, 0, 0), (1, 1, 1)),
                scene_object("b", "chair", (10, 0, 0), (11, 1, 1)),
            ),
        )
        prediction = SceneOutput(
            layout=None,
            objects=(
                scene_object("fp", "chair", (100, 100, 100), (101, 101, 101), 0.9),
                scene_object("pa", "chair", (0, 0, 0), (1, 1, 1), 0.8),
                scene_object("pb", "chair", (10, 0, 0), (11, 1, 1), 0.7),
            ),
        )

        value = ObjectDetectionMAP({"iou_threshold": 0.5}).compute(
            prediction, ground_truth
        )

        self.assertAlmostEqual(value, 2 / 3)

    def test_object_map_dataset_pools_detections_across_scenes(self) -> None:
        scene1_gt = SceneOutput(
            layout=None, objects=(scene_object("a", "chair", (0, 0, 0), (1, 1, 1)),)
        )
        scene1_pred = SceneOutput(
            layout=None,
            objects=(scene_object("pa", "chair", (0, 0, 0), (1, 1, 1), 0.9),),
        )
        scene2_gt = SceneOutput(
            layout=None, objects=(scene_object("c", "chair", (0, 0, 0), (1, 1, 1)),)
        )
        scene2_pred = SceneOutput(
            layout=None,
            objects=(
                scene_object("fp", "chair", (100, 100, 100), (101, 101, 101), 0.95),
                scene_object("pc", "chair", (0, 0, 0), (1, 1, 1), 0.8),
            ),
        )

        value = ObjectDetectionMAPDataset({"iou_threshold": 0.5}).compute_dataset(
            [scene1_pred, scene2_pred], [scene1_gt, scene2_gt]
        )

        self.assertAlmostEqual(value, 2 / 3)


if __name__ == "__main__":
    unittest.main()
