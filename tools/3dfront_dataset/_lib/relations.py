
import numpy as np
from shapely.geometry import Polygon, shape

DEFAULT_EXPAND_DISTANCE = 0.1
WALL_THICKNESS = 0.05
FLOOR_ID = "layout::floor"
CEILING_ID = "layout::ceiling"
WALL_ID = "layout::wall_%d"

CORNER_SIGNS = [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]


def box_corners(center, size, basis):
    center = np.asarray(center, dtype=float)
    size = np.asarray(size, dtype=float)
    basis = np.asarray(basis, dtype=float)
    offsets = np.array(CORNER_SIGNS, dtype=float) * size / 2.0
    return center + offsets @ basis


def grown(size, distance):
    size = np.asarray(size, dtype=float) + distance
    return np.maximum(size, 0.01)


def footprint(corners):
    return Polygon(corners[:, :2]).convex_hull


def z_span(corners):
    return float(corners[:, 2].min()), float(corners[:, 2].max())


def boxes_touch(first, second, expand=DEFAULT_EXPAND_DISTANCE):
    a = box_corners(first["center"], grown(first["size"], expand), first["basis"])
    b = box_corners(second["center"], grown(second["size"], expand), second["basis"])
    a_low, a_high = z_span(a)
    b_low, b_high = z_span(b)
    if min(a_high, b_high) <= max(a_low, b_low):
        return False
    return footprint(a).intersects(footprint(b)) and \
        footprint(a).intersection(footprint(b)).area > 0.0


def wall_boxes(polygon, floor_z, ceiling_z, thickness=WALL_THICKNESS):
    ring = np.asarray(shape(polygon).exterior.coords, dtype=float)[:-1]
    height = ceiling_z - floor_z
    walls = []
    for index in range(len(ring)):
        start, end = ring[index], ring[(index + 1) % len(ring)]
        along = end - start
        length = float(np.hypot(*along))
        if length <= 1e-9:
            continue
        along = along / length
        across = np.array([-along[1], along[0]])
        middle = (start + end) / 2.0
        walls.append({
            "center": [float(middle[0]), float(middle[1]), float(floor_z + height / 2)],
            "size": [length, thickness, height],
            "basis": [[float(along[0]), float(along[1]), 0.0],
                      [float(across[0]), float(across[1]), 0.0],
                      [0.0, 0.0, 1.0]],
        })
    return walls


def slab_box(polygon, z, thickness=WALL_THICKNESS):
    bounds = shape(polygon).bounds
    return {
        "center": [(bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0, float(z)],
        "size": [bounds[2] - bounds[0], bounds[3] - bounds[1], thickness],
        "basis": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    }


def camera_frame_layout(layout, camera):
    camera = np.asarray(camera, dtype=float)
    ring = np.asarray(shape(layout["polygon"]).exterior.coords, dtype=float)[:, :2]
    moved = ring - camera[:2]
    return {
        "polygon": {"type": "Polygon", "coordinates": [moved.tolist()]},
        "floor_z": float(layout["floor_z"]) - float(camera[2]),
        "ceiling_z": float(layout["ceiling_z"]) - float(camera[2]),
    }


def scene_relations(objects, layout, expand=DEFAULT_EXPAND_DISTANCE):
    boxes = [dict(item["bbox"]) for item in objects]
    identifiers = [item["object_id"] for item in objects]
    relations = []

    for first in range(len(boxes)):
        for second in range(first + 1, len(boxes)):
            if boxes_touch(boxes[first], boxes[second], expand):
                relations.append({"source_id": identifiers[first],
                                  "target_id": identifiers[second],
                                  "relation_type": "touching", "score": 1.0})

    floor = slab_box(layout["polygon"], layout["floor_z"])
    ceiling = slab_box(layout["polygon"], layout["ceiling_z"])
    walls = wall_boxes(layout["polygon"], layout["floor_z"], layout["ceiling_z"])
    for index, box in enumerate(boxes):
        if boxes_touch(box, floor, expand):
            relations.append({"source_id": identifiers[index], "target_id": FLOOR_ID,
                              "relation_type": "touching_floor", "score": 1.0})
        if boxes_touch(box, ceiling, expand):
            relations.append({"source_id": identifiers[index], "target_id": CEILING_ID,
                              "relation_type": "touching_ceiling", "score": 1.0})
        for wall_index, wall in enumerate(walls):
            if boxes_touch(box, wall, expand):
                relations.append({"source_id": identifiers[index],
                                  "target_id": WALL_ID % wall_index,
                                  "relation_type": "touching_wall", "score": 1.0})
    return relations, walls
