from __future__ import annotations

from typing import Sequence

from benchmark_toolbox.domain import SceneObject

Match = tuple[SceneObject, "SceneObject | None"]


def greedy_match(
    predictions: Sequence[SceneObject],
    labels: Sequence[SceneObject],
    iou_threshold: float,
) -> list[Match]:
    claimed: set[int] = set()
    matches: list[Match] = []
    for prediction in sorted(predictions, key=lambda item: item.score, reverse=True):
        candidates = [
            (prediction.bbox.iou(label.bbox), index)
            for index, label in enumerate(labels)
            if index not in claimed
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_index >= 0 and best_iou >= iou_threshold:
            claimed.add(best_index)
            matches.append((prediction, labels[best_index]))
        else:
            matches.append((prediction, None))
    return matches
