from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


def read_request(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_requests(path: str) -> list:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "samples" in payload:
        return list(payload["samples"])
    return [payload]


def box(min_corner: Iterable[float], max_corner: Iterable[float]) -> dict:
    return {
        "min_corner": [float(value) for value in min_corner],
        "max_corner": [float(value) for value in max_corner],
    }


def oriented_box(centroid: Iterable[float], size: Iterable[float], basis) -> dict:
    return {
        "center": [float(value) for value in centroid],
        "size": [float(value) for value in size],
        "basis": [[float(value) for value in row] for row in basis],
    }


def scene_object(
    object_id: str,
    label: str,
    bbox: Mapping[str, Any],
    score: float = 1.0,
    attributes: "Mapping[str, Any] | None" = None,
) -> dict:
    return {
        "object_id": str(object_id),
        "label": str(label),
        "score": float(score),
        "bbox": dict(bbox),
        "attributes": dict(attributes or {}),
    }


def corners_to_axis_aligned(corners) -> dict:
    xs = [float(point[0]) for point in corners]
    ys = [float(point[1]) for point in corners]
    zs = [float(point[2]) for point in corners]
    return box([min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)])


def oriented_box_to_axis_aligned(centroid, basis, coeffs) -> dict:
    centre = [float(centroid[0]), float(centroid[1]), float(centroid[2])]
    corners = []
    for s0 in (-1.0, 1.0):
        for s1 in (-1.0, 1.0):
            for s2 in (-1.0, 1.0):
                signs = (s0, s1, s2)
                corner = []
                for axis in range(3):
                    value = centre[axis] + sum(
                        signs[k] * float(coeffs[k]) * float(basis[k][axis])
                        for k in range(3)
                    )
                    corner.append(value)
                corners.append(corner)
    return corners_to_axis_aligned(corners)


def scene_from_oriented_objects(objects, layout_corners=None, layout_box=None) -> dict:
    out_objects = []
    for index, item in enumerate(objects):
        size = [2.0 * float(coeff) for coeff in item["coeffs"]]
        bbox = oriented_box(item["centroid"], size, item["basis"])
        attributes = dict(item.get("attributes") or {})
        if item.get("shape") is not None:
            attributes.setdefault("shape", item["shape"])
        out_objects.append(
            scene_object(
                item.get("id", f"obj-{index}"),
                item.get("label", "object"),
                bbox,
                float(item.get("score", 1.0)),
                attributes or None,
            )
        )
    layout = layout_box
    if layout is None and layout_corners is not None:
        layout = corners_to_axis_aligned(layout_corners)
    return {"layout": layout, "objects": out_objects, "relations": []}


def write_output(
    path: str,
    *,
    layout: "Mapping[str, Any] | None" = None,
    objects: "Iterable[Mapping[str, Any]] | None" = None,
    relations: "Iterable[Mapping[str, Any]] | None" = None,
    metadata: "Mapping[str, Any] | None" = None,
) -> dict:
    payload = {
        "layout": dict(layout) if layout is not None else None,
        "objects": [dict(item) for item in (objects or [])],
        "relations": [dict(item) for item in (relations or [])],
        "metadata": dict(metadata or {}),
    }
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def write_outputs(path: str, outputs) -> dict:
    payload = {"outputs": [dict(item) for item in outputs]}
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def seed_everything(seed) -> None:
    if seed is None:
        return
    import random as _random

    _random.seed(seed)
    try:
        import numpy as _numpy

        _numpy.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch as _torch

        _torch.manual_seed(seed)
    except ImportError:
        pass
