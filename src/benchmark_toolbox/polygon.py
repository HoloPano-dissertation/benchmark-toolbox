from __future__ import annotations

from typing import Sequence

Point2D = tuple[float, float]


def convex_hull(points: Sequence[Point2D]) -> list[Point2D]:
    unique = sorted({(round(float(x), 12), round(float(y), 12)) for x, y in points})
    if len(unique) <= 2:
        return [(x, y) for x, y in unique]

    def cross(o: Point2D, a: Point2D, b: Point2D) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[Point2D] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[Point2D] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def polygon_area(polygon: Sequence[Point2D]) -> float:
    count = len(polygon)
    if count < 3:
        return 0.0
    doubled = 0.0
    for index in range(count):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % count]
        doubled += x1 * y2 - x2 * y1
    return abs(doubled) * 0.5


def line_intersection(
    p1: Point2D, p2: Point2D, a: Point2D, b: Point2D
) -> Point2D:
    """Intersection of segment p1->p2 with the line through a,b."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = a
    x4, y4 = b
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-15:
        return p2
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def convex_intersection_area(
    corners_a: Sequence[Point2D], corners_b: Sequence[Point2D]
) -> float:
    subject = convex_hull(corners_a)
    clip = convex_hull(corners_b)
    if len(subject) < 3 or len(clip) < 3:
        return 0.0
    output = list(subject)
    edges = len(clip)
    for edge in range(edges):
        if not output:
            return 0.0
        a = clip[edge]
        b = clip[(edge + 1) % edges]
        edge_x, edge_y = b[0] - a[0], b[1] - a[1]

        def inside(point: Point2D) -> bool:
            return edge_x * (point[1] - a[1]) - edge_y * (point[0] - a[0]) >= -1e-12

        clipped: list[Point2D] = []
        for index in range(len(output)):
            current = output[index]
            previous = output[index - 1]
            current_in = inside(current)
            previous_in = inside(previous)
            if current_in:
                if not previous_in:
                    clipped.append(line_intersection(previous, current, a, b))
                clipped.append(current)
            elif previous_in:
                clipped.append(line_intersection(previous, current, a, b))
        output = clipped
    return polygon_area(output)
