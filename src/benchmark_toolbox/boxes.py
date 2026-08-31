from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from benchmark_toolbox.polygon import Point2D, convex_intersection_area


def _vector3(value: Sequence[float], field_name: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"{field_name} must contain exactly three coordinates")
    return tuple(float(coordinate) for coordinate in value)


def _finite(value: float) -> float:
    number = float(value)
    return number if math.isfinite(number) else 0.0


_CORNER_SIGNS = tuple(
    (sx, sy, sz)
    for sx in (-1.0, 1.0)
    for sy in (-1.0, 1.0)
    for sz in (-1.0, 1.0)
)

@dataclass(frozen=True)
class BoundingBox3D:
    min_corner: tuple[float, float, float]
    max_corner: tuple[float, float, float]

    def __post_init__(self) -> None:
        minimum = _vector3(self.min_corner, "min_corner")
        maximum = _vector3(self.max_corner, "max_corner")
        if any(low > high for low, high in zip(minimum, maximum)):
            raise ValueError("min_corner coordinates must not exceed max_corner")
        object.__setattr__(self, "min_corner", minimum)
        object.__setattr__(self, "max_corner", maximum)

    @property
    def volume(self) -> float:
        dimensions = (
            maximum - minimum
            for minimum, maximum in zip(self.min_corner, self.max_corner)
        )
        result = 1.0
        for dimension in dimensions:
            result *= max(0.0, dimension)
        return result

    def intersection_volume(self, other: "BoundingBox3D | OrientedBox3D") -> float:
        if isinstance(other, OrientedBox3D):
            return self.as_oriented().intersection_volume(other)
        result = 1.0
        for own_min, own_max, other_min, other_max in zip(
            self.min_corner,
            self.max_corner,
            other.min_corner,
            other.max_corner,
        ):
            result *= max(0.0, min(own_max, other_max) - max(own_min, other_min))
        return result

    def iou(self, other: "BoundingBox3D | OrientedBox3D") -> float:
        if isinstance(other, OrientedBox3D):
            return self.as_oriented().iou(other)
        intersection = self.intersection_volume(other)
        union = self.volume + other.volume - intersection
        return intersection / union if union > 0.0 else 0.0

    def as_oriented(self) -> "OrientedBox3D":
        centre = tuple(
            (low + high) / 2.0
            for low, high in zip(self.min_corner, self.max_corner)
        )
        size = tuple(
            high - low for low, high in zip(self.min_corner, self.max_corner)
        )
        return OrientedBox3D(
            center=centre,
            size=size,
            basis=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "min_corner": list(self.min_corner),
            "max_corner": list(self.max_corner),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BoundingBox3D:
        return cls(
            min_corner=_vector3(data["min_corner"], "min_corner"),
            max_corner=_vector3(data["max_corner"], "max_corner"),
        )


@dataclass(frozen=True)
class OrientedBox3D:
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    basis: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]

    def __post_init__(self) -> None:
        center = tuple(_finite(value) for value in _vector3(self.center, "center"))
        size = tuple(max(0.0, _finite(value)) for value in _vector3(self.size, "size"))
        rows = tuple(
            tuple(_finite(value) for value in _vector3(row, "basis row"))
            for row in self.basis
        )
        if len(rows) != 3:
            raise ValueError("basis must contain exactly three rows")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "basis", rows)

    def corners(self) -> list[tuple[float, float, float]]:
        cx, cy, cz = self.center
        hx, hy, hz = (self.size[0] / 2.0, self.size[1] / 2.0, self.size[2] / 2.0)
        basis = self.basis
        result = []
        for sign_x, sign_y, sign_z in _CORNER_SIGNS:
            ox, oy, oz = sign_x * hx, sign_y * hy, sign_z * hz
            result.append(
                (
                    cx + ox * basis[0][0] + oy * basis[1][0] + oz * basis[2][0],
                    cy + ox * basis[0][1] + oy * basis[1][1] + oz * basis[2][1],
                    cz + ox * basis[0][2] + oy * basis[1][2] + oz * basis[2][2],
                )
            )
        return result

    @property
    def volume(self) -> float:
        return abs(self.size[0] * self.size[1] * self.size[2])

    def _footprint(self) -> list[Point2D]:
        return [(corner[0], corner[1]) for corner in self.corners()]

    def _z_span(self) -> tuple[float, float]:
        heights = [corner[2] for corner in self.corners()]
        return min(heights), max(heights)

    def intersection_volume(self, other: "BoundingBox3D | OrientedBox3D") -> float:
        box = other.as_oriented() if isinstance(other, BoundingBox3D) else other
        area = convex_intersection_area(self._footprint(), box._footprint())
        if area <= 0.0:
            return 0.0
        own_low, own_high = self._z_span()
        other_low, other_high = box._z_span()
        vertical = max(0.0, min(own_high, other_high) - max(own_low, other_low))
        intersection = area * vertical
        return min(intersection, self.volume, box.volume)

    def iou(self, other: "BoundingBox3D | OrientedBox3D") -> float:
        box = other.as_oriented() if isinstance(other, BoundingBox3D) else other
        intersection = self.intersection_volume(box)
        union = self.volume + box.volume - intersection
        return intersection / union if union > 0.0 else 0.0

    def as_oriented(self) -> "OrientedBox3D":
        return self

    def to_dict(self) -> dict[str, list[Any]]:
        return {
            "center": list(self.center),
            "size": list(self.size),
            "basis": [list(row) for row in self.basis],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OrientedBox3D:
        return cls(
            center=_vector3(data["center"], "center"),
            size=_vector3(data["size"], "size"),
            basis=tuple(_vector3(row, "basis row") for row in data["basis"]),
        )


def parse_box(data: Mapping[str, Any]) -> "BoundingBox3D | OrientedBox3D":
    if "basis" in data or "center" in data:
        return OrientedBox3D.from_dict(data)
    return BoundingBox3D.from_dict(data)
