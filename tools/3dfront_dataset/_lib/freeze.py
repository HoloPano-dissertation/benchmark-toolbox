from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Point, shape

from .metric_scale import ScaleError, load_scale_table, resolve_scale

SPLITS = ("train", "val", "test")
DEFAULT_VAL = 0.15
DEFAULT_TEST = 0.15
POLICY_ID = "excluded-rooms"


def source_rooms(scene_root):
    if not scene_root.is_dir():
        raise NotADirectoryError("Source of rooms not found: %s" % scene_root)
    rooms = []
    for house in sorted(p for p in scene_root.iterdir() if p.is_dir()):
        for room in sorted(p for p in house.iterdir() if p.is_dir()):
            if any(room.glob("*.glb")):
                rooms.append("%s/%s" % (house.name, room.name))
    return rooms


def recover(scene_root, room_id):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "3dfront_panorama_renderer"))
    from room_layout import recover_layout
    return recover_layout(scene_root / room_id)


def renderer_module(name):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "3dfront_panorama_renderer"))
    return __import__(name)


def surface_tree(scene_root, room_id, spacing):
    from scipy.spatial import cKDTree

    from .sdf_samples import triangle_samples
    geometry = renderer_module("glb_geometry")
    points = []
    for path in sorted((scene_root / room_id).glob("*.glb")):
        try:
            points.append(triangle_samples(geometry.glb_triangles(path), spacing))
        except Exception:
            continue
    if not points:
        return None
    return cKDTree(np.concatenate(points, axis=0))


def camera_failure(scene_root, room_id, layout, views=4):
    policy = renderer_module("camera_policy")
    geometry = renderer_module("glb_geometry")
    polygon = shape(layout["polygon"])
    region = shape(layout["camera_region"])
    floor_z, ceiling_z = layout["floor_z"], layout["ceiling_z"]
    height = ceiling_z - floor_z
    eye_z = policy.eye_height(floor_z, height, 1.6, 0.6)
    boxes = []
    for path in sorted((scene_root / room_id).glob("*.glb")):
        if policy.is_structural_file(path.name):
            continue
        try:
            boxes.append(geometry.glb_bounds(path))
        except Exception:
            continue
    tree = surface_tree(scene_root, room_id, max(height * 0.01, 1e-3))
    if tree is None:
        return "No geometry to measure the clearance against"
    free, standing = [], []
    for candidate in policy.candidate_grid(region.bounds, eye_z):
        point = Point(candidate[:2])
        if not region.contains(point) or polygon.boundary.distance(point) < height * 0.02:
            continue
        clearance = min(float(tree.query(np.asarray(candidate))[0]),
                        float(polygon.boundary.distance(point)))
        target = standing if policy.stands_on_furniture(
            candidate, boxes, floor_z, height) else free
        target.append((clearance, candidate))
    try:
        selected, _ = policy.choose_from_candidates(
            free, standing, views, policy.min_clearance_for(height))
    except ValueError as error:
        return str(error).split(" Candidate diagnostics")[0]
    locations = [np.asarray(point) for _, point in selected]
    if not policy.poses_separated(locations, height):
        return "Camera poses are distinct but insufficiently separated"
    return None


def review_rooms(scene_root, rooms, table=None, reference_height=None, views=4):
    kept, excluded = [], []
    for room_id in rooms:
        try:
            layout = recover(scene_root, room_id)
        except Exception as error:
            excluded.append({"room_id": room_id,
                             "reason": "Room geometry cannot be recovered: %s" % error,
                             "evidence": "Layout recovery of the renderer"})
            continue
        options = {"table": table} if table else {}
        if reference_height:
            options["reference_height"] = reference_height
        try:
            resolve_scale(room_id, layout, **options)
        except ScaleError as error:
            excluded.append({"room_id": room_id, "reason": str(error),
                             "evidence": "Metric checks of the scale report"})
            continue
        if table and room_id not in table:
            excluded.append({
                "room_id": room_id,
                "reason": "The furniture of the room does not agree on a single metric "
                          "scale, so the room cannot be measured",
                "evidence": "Scale report built from the original 3D-FRONT"})
            continue
        failure = camera_failure(scene_root, room_id, layout, views)
        if failure:
            excluded.append({"room_id": room_id,
                             "reason": "No usable camera poses: %s" % failure,
                             "evidence": "Camera selection of the renderer policy, "
                                         "run without Blender"})
            continue
        kept.append(room_id)
    return kept, excluded


def house_disjoint_split(rooms, val=DEFAULT_VAL, test=DEFAULT_TEST, seed=0):
    if not 0 < val + test < 1:
        raise ValueError("The validation and test shares must leave room for training")
    houses = {}
    for room_id in rooms:
        houses.setdefault(room_id.split("/")[0], []).append(room_id)
    order = sorted(houses)
    random.Random(seed).shuffle(order)
    total = len(rooms)
    target = {"test": total * test, "val": total * val,
              "train": total * (1.0 - val - test)}
    assigned = {name: [] for name in SPLITS}
    for house in order:
        name = max(SPLITS, key=lambda s: target[s] - len(assigned[s]))
        assigned[name].extend(sorted(houses[house]))
    return {name: sorted(values) for name, values in assigned.items()}


def write_split(splits_dir, assigned, excluded, scene_root, force=False):
    splits_dir.mkdir(parents=True, exist_ok=True)
    existing = [splits_dir / (name + ".txt") for name in SPLITS]
    if any(path.is_file() for path in existing) and not force:
        raise FileExistsError(
            "A frozen split is already present in %s. Replacing it makes earlier results "
            "belong to a different set, so pass --force to do it deliberately." % splits_dir)
    retained = sum(len(values) for values in assigned.values())
    for name in SPLITS:
        (splits_dir / (name + ".txt")).write_text(
            "".join(room_id + "\n" for room_id in assigned[name]), encoding="utf-8")
    by_split = {room_id: name for name in SPLITS for room_id in assigned[name]}
    rooms = []
    for item in excluded:
        rooms.append({"room_id": item["room_id"],
                      "house_id": item["room_id"].split("/")[0],
                      "split": by_split.get(item["room_id"], "excluded"),
                      "reason": item["reason"],
                      "evidence": item["evidence"]})
    policy = {
        "policy_id": POLICY_ID,
        "user_decision": "Exclude these rooms from train, validation, test and published "
                         "benchmark; do not reconstruct geometry or delete source assets.",
        "expected_original_rooms": retained + len(rooms),
        "expected_retained_rooms": retained,
        "views_per_room": 4,
        "source_root": str(scene_root),
        "evidence": "Layout recovery and the metric checks of the scale report, applied "
                    "to every room of the source.",
        "rooms": rooms,
    }
    (splits_dir / "excluded_rooms.json").write_text(
        json.dumps(policy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "rooms": retained,
        "room_counts": {name: len(assigned[name]) for name in SPLITS},
        "panoramas": retained * policy["views_per_room"],
        "excluded": len(rooms),
        "houses": len({room_id.split("/")[0] for room_id in by_split}),
        "house_disjoint": True,
    }


def freeze(scene_root, splits_dir, metadata=None, val=DEFAULT_VAL, test=DEFAULT_TEST,
           seed=0, reference_height=None, force=False, views=4):
    scene_root = Path(scene_root).resolve()
    table = load_scale_table(Path(metadata)) if metadata else None
    rooms = source_rooms(scene_root)
    if not rooms:
        raise ValueError("No rooms of the form <house>/<room>/*.glb under %s" % scene_root)
    kept, excluded = review_rooms(scene_root, rooms, table, reference_height, views)
    assigned = house_disjoint_split(kept, val, test, seed)
    report = write_split(Path(splits_dir), assigned, excluded, scene_root, force)
    report["source_rooms"] = len(rooms)
    report["scale_source"] = "original 3D-FRONT" if table else "ceiling anchor"
    return report
