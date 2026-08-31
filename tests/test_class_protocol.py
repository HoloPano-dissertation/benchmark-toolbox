import math
import unittest

from benchmark_toolbox.domain import BoundingBox3D, SceneObject, SceneOutput
from benchmark_toolbox.metrics.geometry import (
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


class ClassProtocolTest(unittest.TestCase):
    def test_classes_restrict_the_average_to_the_protocol(self) -> None:
        ground_truth = SceneOutput(
            layout=None,
            objects=(
                scene_object("a", "chair", (0, 0, 0), (1, 1, 1)),
                scene_object("b", "plant", (10, 0, 0), (11, 1, 1)),
            ),
        )
        prediction = SceneOutput(
            layout=None,
            objects=(scene_object("pa", "chair", (0, 0, 0), (1, 1, 1), 0.9),),
        )

        unrestricted = ObjectDetectionMAP({"iou_threshold": 0.15}).compute(
            prediction, ground_truth
        )
        restricted = ObjectDetectionMAP(
            {"iou_threshold": 0.15, "classes": ["chair"]}
        ).compute(prediction, ground_truth)

        self.assertAlmostEqual(unrestricted, 0.5)
        self.assertAlmostEqual(restricted, 1.0)

    def test_out_of_protocol_prediction_is_not_a_false_positive(self) -> None:
        ground_truth = SceneOutput(
            layout=None, objects=(scene_object("a", "chair", (0, 0, 0), (1, 1, 1)),)
        )
        prediction = SceneOutput(
            layout=None,
            objects=(
                scene_object("junk", "plant", (50, 50, 50), (51, 51, 51), 0.99),
                scene_object("pa", "chair", (0, 0, 0), (1, 1, 1), 0.5),
            ),
        )

        value = ObjectDetectionMAP(
            {"iou_threshold": 0.15, "classes": ["chair"]}
        ).compute(prediction, ground_truth)

        self.assertAlmostEqual(value, 1.0)

    def test_class_map_merges_source_labels(self) -> None:
        ground_truth = SceneOutput(
            layout=None,
            objects=(
                scene_object("a", "office_chair", (0, 0, 0), (1, 1, 1)),
                scene_object("b", "stool", (10, 0, 0), (11, 1, 1)),
            ),
        )
        prediction = SceneOutput(
            layout=None,
            objects=(
                scene_object("pa", "office_chair", (0, 0, 0), (1, 1, 1), 0.9),
                scene_object("pb", "stool", (10, 0, 0), (11, 1, 1), 0.8),
            ),
        )

        metric = ObjectDetectionMAP(
            {
                "iou_threshold": 0.15,
                "class_map": {"office_chair": "chair", "stool": "chair"},
            }
        )
        value = metric.compute(prediction, ground_truth)

        self.assertAlmostEqual(value, 1.0)

    def test_class_map_drops_unmapped_labels(self) -> None:
        ground_truth = SceneOutput(
            layout=None,
            objects=(
                scene_object("a", "office_chair", (0, 0, 0), (1, 1, 1)),
                scene_object("b", "plant", (10, 0, 0), (11, 1, 1)),
            ),
        )
        prediction = SceneOutput(
            layout=None,
            objects=(scene_object("pa", "office_chair", (0, 0, 0), (1, 1, 1), 0.9),),
        )

        value = ObjectDetectionMAP(
            {"iou_threshold": 0.15, "class_map": {"office_chair": "chair"}}
        ).compute(prediction, ground_truth)

        self.assertAlmostEqual(value, 1.0)

    def test_scene_without_protocol_classes_is_not_applicable(self) -> None:
        ground_truth = SceneOutput(
            layout=None, objects=(scene_object("b", "plant", (0, 0, 0), (1, 1, 1)),)
        )
        prediction = SceneOutput(layout=None, objects=())

        value = ObjectDetectionMAP(
            {"iou_threshold": 0.15, "classes": ["chair"]}
        ).compute(prediction, ground_truth)

        self.assertTrue(math.isnan(value))

    def test_without_protocol_empty_ground_truth_keeps_legacy_zero(self) -> None:
        ground_truth = SceneOutput(layout=None, objects=())
        prediction = SceneOutput(layout=None, objects=())

        value = ObjectDetectionMAP({"iou_threshold": 0.15}).compute(
            prediction, ground_truth
        )

        self.assertEqual(value, 0.0)

    def test_dataset_metric_restricts_and_renames(self) -> None:
        ground_truth = SceneOutput(
            layout=None,
            objects=(
                scene_object("a", "office_chair", (0, 0, 0), (1, 1, 1)),
                scene_object("b", "plant", (10, 0, 0), (11, 1, 1)),
            ),
        )
        prediction = SceneOutput(
            layout=None,
            objects=(scene_object("pa", "office_chair", (0, 0, 0), (1, 1, 1), 0.9),),
        )

        metric = ObjectDetectionMAPDataset(
            {
                "iou_threshold": 0.15,
                "class_map": {"office_chair": "chair"},
                "classes": ["chair"],
                "name": "ap_chair",
            }
        )
        value = metric.compute_dataset([prediction], [ground_truth])

        self.assertAlmostEqual(value, 1.0)
        self.assertEqual(metric.name, "ap_chair")

    def test_name_defaults_to_the_metric_type(self) -> None:
        self.assertEqual(ObjectDetectionMAP({"iou_threshold": 0.15}).name, "object_map")
        self.assertEqual(
            ObjectDetectionMAPDataset({"iou_threshold": 0.15}).name,
            "object_map_dataset",
        )

    def test_invalid_protocol_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ObjectDetectionMAP({"classes": "chair"})
        with self.assertRaises(ValueError):
            ObjectDetectionMAP({"class_map": ["chair"]})


if __name__ == "__main__":
    unittest.main()
