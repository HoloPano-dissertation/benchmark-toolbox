from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Any, Mapping, Sequence

from benchmark_toolbox.domain import SceneObject, SceneOutput
from benchmark_toolbox.metrics.base import BaseMetric
from benchmark_toolbox.metrics.matching import greedy_match


@BaseMetric.registry.register("layout_iou_3d")
class LayoutIoU3D(BaseMetric):
    name = "layout_iou_3d"

    def __init__(self, parameters: dict[str, Any] | None = None) -> None:
        del parameters

    def compute(
        self, prediction: SceneOutput, ground_truth: SceneOutput | None
    ) -> float:
        if ground_truth is None:
            raise ValueError("Layout 3D IoU requires ground truth")
        if ground_truth.layout is None:
            return float("nan")
        if prediction.layout is None:
            return 0.0
        return prediction.layout.iou(ground_truth.layout)


@BaseMetric.registry.register("collision_rate")
class CollisionRate(BaseMetric):
    name = "collision_rate"
    requires_ground_truth = False

    def __init__(self, parameters: dict[str, Any] | None = None) -> None:
        parameters = parameters or {}
        self.minimum_intersection = float(
            parameters.get("minimum_intersection", 0.0)
        )

    def compute(
        self, prediction: SceneOutput, ground_truth: SceneOutput | None
    ) -> float:
        del ground_truth
        pair_count = 0
        collision_count = 0
        for left_index, left in enumerate(prediction.objects):
            for right in prediction.objects[left_index + 1 :]:
                pair_count += 1
                if (
                    left.bbox.intersection_volume(right.bbox)
                    > self.minimum_intersection
                ):
                    collision_count += 1
        return collision_count / pair_count if pair_count else 0.0


@BaseMetric.registry.register("layout_violation_rate")
class LayoutViolationRate(BaseMetric):
    name = "layout_violation_rate"
    requires_ground_truth = False

    def __init__(self, parameters: dict[str, Any] | None = None) -> None:
        parameters = parameters or {}
        self.tolerance = float(parameters.get("tolerance", 0.0))
        if not 0.0 <= self.tolerance < 1.0:
            raise ValueError("tolerance must be in the interval [0, 1)")

    def compute(
        self, prediction: SceneOutput, ground_truth: SceneOutput | None
    ) -> float:
        del ground_truth
        layout = prediction.layout
        if layout is None:
            return float("nan")
        if not prediction.objects:
            return 0.0
        violations = 0
        considered = 0
        for scene_object in prediction.objects:
            volume = scene_object.bbox.volume
            if volume <= 0.0:
                continue
            considered += 1
            inside = scene_object.bbox.intersection_volume(layout)
            outside_fraction = 1.0 - inside / volume
            if outside_fraction > self.tolerance + 1e-9:
                violations += 1
        return violations / considered if considered else 0.0


@BaseMetric.registry.register("layout_penetration")
class LayoutPenetration(BaseMetric):
    name = "layout_penetration"
    requires_ground_truth = False

    def __init__(self, parameters: dict[str, Any] | None = None) -> None:
        del parameters

    def compute(
        self, prediction: SceneOutput, ground_truth: SceneOutput | None
    ) -> float:
        del ground_truth
        layout = prediction.layout
        if layout is None:
            return float("nan")
        if not prediction.objects:
            return 0.0
        total = 0.0
        considered = 0
        for scene_object in prediction.objects:
            volume = scene_object.bbox.volume
            if volume <= 0.0:
                continue
            considered += 1
            inside = scene_object.bbox.intersection_volume(layout)
            total += max(0.0, 1.0 - inside / volume)
        return total / considered if considered else 0.0


class _ClassProtocol:
    def __init__(
        self,
        classes: Sequence[str] | None,
        class_map: Mapping[str, str] | None,
    ) -> None:
        self.classes = frozenset(classes) if classes is not None else None
        self.class_map = dict(class_map) if class_map is not None else None

    def apply(self, objects: Sequence[SceneObject]) -> list[SceneObject]:
        kept: list[SceneObject] = []
        for scene_object in objects:
            label: str | None = scene_object.label
            if self.class_map is not None:
                label = self.class_map.get(scene_object.label)
                if label is None:
                    continue
            if self.classes is not None and label not in self.classes:
                continue
            kept.append(
                scene_object
                if label == scene_object.label
                else replace(scene_object, label=label)
            )
        return kept


def _parse_class_protocol(parameters: Mapping[str, Any]) -> _ClassProtocol | None:
    classes = parameters.get("classes")
    class_map = parameters.get("class_map")
    if classes is None and class_map is None:
        return None
    if classes is not None and (
        isinstance(classes, str)
        or not all(isinstance(item, str) for item in classes)
    ):
        raise ValueError("classes must be a list of class names")
    if class_map is not None and (
        not isinstance(class_map, Mapping)
        or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in class_map.items()
        )
    ):
        raise ValueError("class_map must be a mapping of source label -> class name")
    return _ClassProtocol(classes, class_map)


def _interpolated_ap(
    precisions: list[float], recalls: list[float]
) -> float:
    average_precision = 0.0
    previous_recall = 0.0
    for recall_level in sorted(set(recalls)):
        precision = max(
            (
                precision
                for precision, recall in zip(precisions, recalls)
                if recall >= recall_level
            ),
            default=0.0,
        )
        average_precision += (recall_level - previous_recall) * precision
        previous_recall = recall_level
    return average_precision


def _ap_from_detections(
    detections: list[tuple[float, bool]], total_gt: int
) -> float:
    if total_gt <= 0:
        return 0.0
    ranked = sorted(detections, key=lambda item: item[0], reverse=True)
    cumulative_tp = 0
    cumulative_fp = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for _score, is_true_positive in ranked:
        cumulative_tp += 1 if is_true_positive else 0
        cumulative_fp += 0 if is_true_positive else 1
        precisions.append(cumulative_tp / (cumulative_tp + cumulative_fp))
        recalls.append(cumulative_tp / total_gt)
    return _interpolated_ap(precisions, recalls)


def _match_scene(
    predictions: list[SceneObject],
    labels: list[SceneObject],
    iou_threshold: float,
) -> list[tuple[float, bool]]:
    return [
        (prediction.score, label is not None)
        for prediction, label in greedy_match(predictions, labels, iou_threshold)
    ]


def _average_precision(
    predictions: list[SceneObject],
    labels: list[SceneObject],
    iou_threshold: float,
) -> float:
    detections = _match_scene(predictions, labels, iou_threshold)
    return _ap_from_detections(detections, len(labels))


@BaseMetric.registry.register("object_map")
class ObjectDetectionMAP(BaseMetric):
    name = "object_map"

    def __init__(self, parameters: dict[str, Any] | None = None) -> None:
        parameters = parameters or {}
        self.iou_threshold = float(parameters.get("iou_threshold", 0.5))
        if not 0.0 < self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in the interval (0, 1]")
        self.protocol = _parse_class_protocol(parameters)
        self.name = str(parameters.get("name", type(self).name))

    def compute(
        self, prediction: SceneOutput, ground_truth: SceneOutput | None
    ) -> float:
        if ground_truth is None:
            raise ValueError("Object mAP requires ground truth")
        prediction_objects = list(prediction.objects)
        label_objects = list(ground_truth.objects)
        if self.protocol is not None:
            prediction_objects = self.protocol.apply(prediction_objects)
            label_objects = self.protocol.apply(label_objects)
        predictions_by_label: dict[str, list[SceneObject]] = defaultdict(list)
        labels_by_label: dict[str, list[SceneObject]] = defaultdict(list)
        for scene_object in prediction_objects:
            predictions_by_label[scene_object.label].append(scene_object)
        for scene_object in label_objects:
            labels_by_label[scene_object.label].append(scene_object)
        if not labels_by_label:
            return float("nan") if self.protocol is not None else 0.0
        average_precisions = [
            _average_precision(
                predictions_by_label[label],
                labels,
                self.iou_threshold,
            )
            for label, labels in labels_by_label.items()
        ]
        return sum(average_precisions) / len(average_precisions)


@BaseMetric.registry.register("object_map_dataset")
class ObjectDetectionMAPDataset(BaseMetric):
    name = "object_map_dataset"
    dataset_level = True

    def __init__(self, parameters: dict[str, Any] | None = None) -> None:
        parameters = parameters or {}
        self.iou_threshold = float(parameters.get("iou_threshold", 0.5))
        if not 0.0 < self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in the interval (0, 1]")
        self.protocol = _parse_class_protocol(parameters)
        self.name = str(parameters.get("name", type(self).name))

    def compute(
        self, prediction: SceneOutput, ground_truth: SceneOutput | None
    ) -> float:
        raise NotImplementedError("object_map_dataset is computed via compute_dataset")

    def compute_dataset(
        self,
        predictions: list[SceneOutput],
        ground_truths: list[SceneOutput],
    ) -> float:
        class_detections: dict[str, list[tuple[float, bool]]] = defaultdict(list)
        class_gt_count: dict[str, int] = defaultdict(int)

        for prediction, ground_truth in zip(predictions, ground_truths):
            prediction_objects = list(prediction.objects)
            label_objects = list(ground_truth.objects)
            if self.protocol is not None:
                prediction_objects = self.protocol.apply(prediction_objects)
                label_objects = self.protocol.apply(label_objects)

            gt_by_label: dict[str, list[SceneObject]] = defaultdict(list)
            for scene_object in label_objects:
                gt_by_label[scene_object.label].append(scene_object)
            for label, labels in gt_by_label.items():
                class_gt_count[label] += len(labels)

            pred_by_label: dict[str, list[SceneObject]] = defaultdict(list)
            for scene_object in prediction_objects:
                pred_by_label[scene_object.label].append(scene_object)
            for label, preds in pred_by_label.items():
                class_detections[label].extend(
                    _match_scene(preds, gt_by_label.get(label, []), self.iou_threshold)
                )

        if not class_gt_count:
            return 0.0

        average_precisions = [
            _ap_from_detections(class_detections.get(label, []), total_gt)
            for label, total_gt in class_gt_count.items()
        ]
        return sum(average_precisions) / len(average_precisions)
