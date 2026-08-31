from __future__ import annotations

import array
import bisect
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any, Sequence

from benchmark_toolbox.domain import SceneObject, SceneOutput
from benchmark_toolbox.metrics.base import BaseMetric
from benchmark_toolbox.metrics.matching import greedy_match

Point = tuple[float, float, float]
Triangle = tuple[int, int, int]


class _KDNode:
    __slots__ = ("axis", "point", "left", "right")

    def __init__(self, axis: int, point: Point, left, right) -> None:
        self.axis = axis
        self.point = point
        self.left = left
        self.right = right


def _build_kd(points: list[Point], depth: int = 0):
    if not points:
        return None
    axis = depth % 3
    points.sort(key=lambda p: p[axis])
    mid = len(points) // 2
    return _KDNode(
        axis,
        points[mid],
        _build_kd(points[:mid], depth + 1),
        _build_kd(points[mid + 1 :], depth + 1),
    )


def _nearest_sq(root, query: Point) -> float:
    best = math.inf
    stack = [root]
    qx, qy, qz = query
    while stack:
        node = stack.pop()
        if node is None:
            continue
        px, py, pz = node.point
        dx, dy, dz = qx - px, qy - py, qz - pz
        distance = dx * dx + dy * dy + dz * dz
        if distance < best:
            best = distance
        diff = query[node.axis] - node.point[node.axis]
        near = node.left if diff < 0 else node.right
        far = node.right if diff < 0 else node.left
        if diff * diff < best:
            stack.append(far)
        stack.append(near)
    return best

def chamfer_and_fscore(
    pred_points: Sequence[Point],
    gt_points: Sequence[Point],
    *,
    squared: bool = True,
    fscore_threshold: float = 0.1,
) -> tuple[float, float]:
    if not pred_points or not gt_points:
        return math.nan, math.nan
    gt_tree = _build_kd(list(gt_points))
    pred_tree = _build_kd(list(pred_points))
    pred_to_gt = [_nearest_sq(gt_tree, point) for point in pred_points]
    gt_to_pred = [_nearest_sq(pred_tree, point) for point in gt_points]

    if squared:
        chamfer = (
            sum(pred_to_gt) / len(pred_to_gt) + sum(gt_to_pred) / len(gt_to_pred)
        )
    else:
        chamfer = sum(math.sqrt(d) for d in pred_to_gt) / len(pred_to_gt) + sum(
            math.sqrt(d) for d in gt_to_pred
        ) / len(gt_to_pred)

    tau_sq = fscore_threshold * fscore_threshold
    precision = sum(1 for d in pred_to_gt if d <= tau_sq) / len(pred_to_gt)
    recall = sum(1 for d in gt_to_pred if d <= tau_sq) / len(gt_to_pred)
    fscore = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return chamfer, fscore

def _as_points(values: Any) -> list[Point]:
    points: list[Point] = []
    for item in values:
        if len(item) < 3:
            raise ValueError("each point needs three coordinates")
        points.append((float(item[0]), float(item[1]), float(item[2])))
    return points


def write_f32_points(path: str | Path, points: Sequence[Point]) -> None:
    flat = array.array("f", (coordinate for point in points for coordinate in point))
    if sys.byteorder != "little":
        flat.byteswap()
    Path(path).write_bytes(flat.tobytes())


def _read_f32(path: Path) -> list[Point]:
    flat = array.array("f")
    data = path.read_bytes()
    flat.frombytes(data[: len(data) - len(data) % 4])
    if sys.byteorder != "little":
        flat.byteswap()
    return [
        (flat[i], flat[i + 1], flat[i + 2]) for i in range(0, len(flat) - 2, 3)
    ]


_PLY_TYPE_TO_STRUCT = {
    "char": "b", "int8": "b", "uchar": "B", "uint8": "B",
    "short": "h", "int16": "h", "ushort": "H", "uint16": "H",
    "int": "i", "int32": "i", "uint": "I", "uint32": "I",
    "float": "f", "float32": "f", "double": "d", "float64": "d",
}


def _read_ply(path: Path) -> tuple[list[Point], list[Triangle]]:
    raw = path.read_bytes()
    header_end = raw.find(b"end_header")
    if header_end < 0:
        raise ValueError(f"{path}: PLY without end_header")
    line_end = raw.find(b"\n", header_end)
    header = raw[:line_end].decode("ascii", "replace").splitlines()
    body = raw[line_end + 1 :]

    fmt = "ascii"
    elements: list[tuple[str, int, list[tuple[str, str, str]]]] = []
    for line in header:
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "format":
            fmt = tokens[1]
        elif tokens[0] == "element":
            elements.append((tokens[1], int(tokens[2]), []))
        elif tokens[0] == "property" and elements:
            if tokens[1] == "list":
                elements[-1][2].append(("list", tokens[2], tokens[3]))
            else:
                elements[-1][2].append(("scalar", tokens[1], tokens[2]))

    if fmt == "ascii":
        return _read_ply_ascii(body.decode("ascii", "replace"), elements)
    if fmt == "binary_little_endian":
        return _read_ply_binary(body, elements)
    raise ValueError(f"{path}: unsupported PLY format '{fmt}'")


def _read_ply_ascii(body: str, elements) -> tuple[list[Point], list[Triangle]]:
    tokens = body.split()
    cursor = 0
    vertices: list[Point] = []
    faces: list[Triangle] = []
    for name, count, properties in elements:
        if name == "vertex":
            axis = {prop[2]: index for index, prop in enumerate(properties)}
            width = len(properties)
            xi, yi, zi = axis["x"], axis["y"], axis["z"]
            for _ in range(count):
                row = tokens[cursor : cursor + width]
                cursor += width
                vertices.append((float(row[xi]), float(row[yi]), float(row[zi])))
        elif name == "face":
            for _ in range(count):
                degree = int(tokens[cursor])
                indices = [int(tokens[cursor + 1 + k]) for k in range(degree)]
                cursor += 1 + degree
                _fan(indices, faces)
        else:
            width = len(properties)
            cursor += count * width
    return vertices, faces


def _read_ply_binary(body: bytes, elements) -> tuple[list[Point], list[Triangle]]:
    vertices: list[Point] = []
    faces: list[Triangle] = []
    offset = 0
    for name, count, properties in elements:
        if name == "vertex":
            fmt = "<" + "".join(
                _PLY_TYPE_TO_STRUCT[prop[1]] for prop in properties
            )
            size = struct.calcsize(fmt)
            unpack = struct.Struct(fmt).unpack_from
            axis = {prop[2]: index for index, prop in enumerate(properties)}
            xi, yi, zi = axis["x"], axis["y"], axis["z"]
            for _ in range(count):
                row = unpack(body, offset)
                offset += size
                vertices.append((float(row[xi]), float(row[yi]), float(row[zi])))
        elif name == "face":
            list_prop = next(p for p in properties if p[0] == "list")
            count_char = _PLY_TYPE_TO_STRUCT[list_prop[1]]
            index_char = _PLY_TYPE_TO_STRUCT[list_prop[2]]
            count_size = struct.calcsize(count_char)
            index_size = struct.calcsize(index_char)
            for _ in range(count):
                degree = struct.unpack_from("<" + count_char, body, offset)[0]
                offset += count_size
                indices = [
                    struct.unpack_from("<" + index_char, body, offset + k * index_size)[0]
                    for k in range(degree)
                ]
                offset += degree * index_size
                _fan(indices, faces)
        else:
            fmt = "<" + "".join(
                _PLY_TYPE_TO_STRUCT[prop[1]] for prop in properties if prop[0] == "scalar"
            )
            offset += count * struct.calcsize(fmt)
    return vertices, faces


def _read_obj(path: Path) -> tuple[list[Point], list[Triangle]]:
    vertices: list[Point] = []
    faces: list[Triangle] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("v "):
            _, x, y, z, *_ = line.split()
            vertices.append((float(x), float(y), float(z)))
        elif line.startswith("f "):
            indices = [
                int(token.split("/", 1)[0]) - 1
                for token in line.split()[1:]
            ]
            _fan(indices, faces)
    return vertices, faces


def _fan(indices: list[int], faces: list[Triangle]) -> None:
    for k in range(1, len(indices) - 1):
        faces.append((indices[0], indices[k], indices[k + 1]))


def _sample_surface(
    vertices: list[Point], faces: list[Triangle], count: int, rng
) -> list[Point]:
    if not faces:
        return [rng.choice(vertices) for _ in range(count)] if vertices else []
    cumulative: list[float] = []
    total = 0.0
    for i, j, k in faces:
        total += _triangle_area(vertices[i], vertices[j], vertices[k])
        cumulative.append(total)
    if total <= 0.0:
        return [rng.choice(vertices) for _ in range(count)]
    points: list[Point] = []
    last = len(faces) - 1
    for _ in range(count):
        index = bisect.bisect_left(cumulative, rng.random() * total)
        i, j, k = faces[min(index, last)]
        points.append(_barycentric(vertices[i], vertices[j], vertices[k], rng))
    return points


def _triangle_area(a: Point, b: Point, c: Point) -> float:
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    cx, cy, cz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    return 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)


def _barycentric(a: Point, b: Point, c: Point, rng) -> Point:
    r1 = math.sqrt(rng.random())
    r2 = rng.random()
    wa, wb, wc = 1.0 - r1, r1 * (1.0 - r2), r1 * r2
    return (
        wa * a[0] + wb * b[0] + wc * c[0],
        wa * a[1] + wb * b[1] + wc * c[1],
        wa * a[2] + wb * b[2] + wc * c[2],
    )


def load_shape_points(
    spec: Any, *, num_points: int, rng, mesh_root: Path | None = None
) -> list[Point] | None:
    if spec is None:
        return None
    if isinstance(spec, dict):
        if "points" in spec:
            return _as_points(spec["points"])
        spec = spec.get("file") or spec.get("mesh") or spec.get("path")
    if isinstance(spec, (list, tuple)):
        return _as_points(spec)
    if not isinstance(spec, str):
        raise ValueError(f"unsupported shape reference: {type(spec).__name__}")

    path = Path(spec).expanduser()
    if mesh_root is not None and not path.is_absolute():
        path = Path(mesh_root) / path
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _as_points(json.loads(path.read_text(encoding="utf-8")))
    if suffix in (".f32", ".bin"):
        return _read_f32(path)
    if suffix == ".ply":
        return _sample_surface(*_read_ply(path), num_points, rng)
    if suffix == ".obj":
        return _sample_surface(*_read_obj(path), num_points, rng)
    raise ValueError(f"unsupported shape file '{path.name}'")

def _canonicalize(points: list[Point]) -> list[Point]:
    lo = [min(p[a] for p in points) for a in range(3)]
    hi = [max(p[a] for p in points) for a in range(3)]
    centre = [(lo[a] + hi[a]) / 2.0 for a in range(3)]
    diag = math.sqrt(sum((hi[a] - lo[a]) ** 2 for a in range(3))) or 1.0
    return [
        ((p[0] - centre[0]) / diag, (p[1] - centre[1]) / diag, (p[2] - centre[2]) / diag)
        for p in points
    ]


def _normalize_pair(
    pred: list[Point], gt: list[Point], mode: str
) -> tuple[list[Point], list[Point]]:
    if mode == "none":
        return pred, gt
    if mode == "canonical":
        return _canonicalize(pred), _canonicalize(gt)
    if mode != "unit":
        raise ValueError(f"unknown normalize mode '{mode}'")
    cx = sum(p[0] for p in gt) / len(gt)
    cy = sum(p[1] for p in gt) / len(gt)
    cz = sum(p[2] for p in gt) / len(gt)
    lo = [min(p[a] for p in gt) for a in range(3)]
    hi = [max(p[a] for p in gt) for a in range(3)]
    diag = math.sqrt(sum((hi[a] - lo[a]) ** 2 for a in range(3))) or 1.0

    def apply(points: list[Point]) -> list[Point]:
        return [
            ((p[0] - cx) / diag, (p[1] - cy) / diag, (p[2] - cz) / diag)
            for p in points
        ]

    return apply(pred), apply(gt)


def _match_pairs(
    predictions: Sequence[SceneObject],
    labels: Sequence[SceneObject],
    iou_threshold: float,
) -> list[tuple[SceneObject, SceneObject]]:
    return [
        (prediction, label)
        for prediction, label in greedy_match(predictions, labels, iou_threshold)
        if label is not None
    ]

class _ShapeMetric(BaseMetric):
    def __init__(self, parameters: dict[str, Any] | None = None) -> None:
        parameters = parameters or {}
        self.iou_threshold = float(parameters.get("iou_threshold", 0.15))
        if not 0.0 < self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in the interval (0, 1]")
        self.num_points = int(parameters.get("num_points", 2048))
        if self.num_points <= 0:
            raise ValueError("num_points must be positive")
        self.seed = int(parameters.get("seed", 0))
        self.squared = bool(parameters.get("squared", True))
        self.fscore_threshold = float(parameters.get("fscore_threshold", 0.1))
        self.normalize = str(parameters.get("normalize", "none"))
        mesh_root = parameters.get("mesh_root")
        self.mesh_root = Path(mesh_root) if mesh_root else None

    def _pair_points(self, obj: SceneObject):
        import random

        return load_shape_points(
            obj.attributes.get("shape"),
            num_points=self.num_points,
            rng=random.Random(self.seed),
            mesh_root=self.mesh_root,
        )

    def _reduce_pair(self, pred_points, gt_points) -> float:
        raise NotImplementedError

    def compute(
        self, prediction: SceneOutput, ground_truth: SceneOutput | None
    ) -> float:
        if ground_truth is None:
            raise ValueError(f"{self.name} requires ground truth")
        values: list[float] = []
        for pred_obj, gt_obj in _match_pairs(
            prediction.objects, ground_truth.objects, self.iou_threshold
        ):
            pred_points = self._pair_points(pred_obj)
            gt_points = self._pair_points(gt_obj)
            if not pred_points or not gt_points:
                continue
            pred_points, gt_points = _normalize_pair(
                pred_points, gt_points, self.normalize
            )
            value = self._reduce_pair(pred_points, gt_points)
            if not math.isnan(value):
                values.append(value)
        if not values:
            return math.nan
        return sum(values) / len(values)


@BaseMetric.registry.register("mesh_chamfer")
class MeshChamfer(_ShapeMetric):
    name = "mesh_chamfer"

    def _reduce_pair(self, pred_points, gt_points) -> float:
        chamfer, _ = chamfer_and_fscore(
            pred_points,
            gt_points,
            squared=self.squared,
            fscore_threshold=self.fscore_threshold,
        )
        return chamfer


@BaseMetric.registry.register("mesh_fscore")
class MeshFScore(_ShapeMetric):
    name = "mesh_fscore"

    def _reduce_pair(self, pred_points, gt_points) -> float:
        _, fscore = chamfer_and_fscore(
            pred_points,
            gt_points,
            squared=self.squared,
            fscore_threshold=self.fscore_threshold,
        )
        return fscore
