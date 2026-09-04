#!/usr/bin/env python3

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "_lib"))
from export_ground_truth import room_geometry

UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
MINIMUM_OBJECTS = 2
MAXIMUM_SPREAD = 0.15
MINIMUM_CLASS_INSTANCES = 100
NORMALISED_SHELL = 1.9
TAIL_CLASS = "other"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("scene_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--minimum-objects", type=int, default=MINIMUM_OBJECTS)
    parser.add_argument("--maximum-spread", type=float, default=MAXIMUM_SPREAD)
    parser.add_argument("--minimum-class-instances", type=int,
                        default=MINIMUM_CLASS_INSTANCES)
    return parser.parse_args()


def flat_dimensions(value):
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or len(value) != 3:
        return None
    if any(isinstance(entry, list) for entry in value):
        return None
    return [abs(float(entry)) for entry in value]


def scene_furniture(archive, house):
    scene = json.loads(archive.read("3D-FRONT/%s.json" % house))
    sizes, categories, titles, by_uid = {}, {}, {}, {}
    for item in scene.get("furniture", []):
        jid = item.get("jid")
        if not jid:
            continue
        dimensions = flat_dimensions(item.get("size") or item.get("bbox"))
        if dimensions:
            sizes[jid] = dimensions
            by_uid[item.get("uid")] = jid
        if item.get("sourceCategoryId"):
            categories[jid] = item["sourceCategoryId"]
        if item.get("title"):
            titles[jid] = item["title"]

    stretched = collections.defaultdict(lambda: collections.defaultdict(list))
    for room in scene.get("scene", {}).get("room", []):
        for child in room.get("children", []):
            jid = by_uid.get(child.get("ref"))
            if jid is None:
                continue
            factor = np.abs(np.asarray(child.get("scale", [1, 1, 1]), dtype=float))
            if factor.shape != (3,) or not np.all(np.isfinite(factor)):
                continue
            stretched[room.get("instanceid")][jid].append(
                (np.asarray(sizes[jid], dtype=float) * factor).tolist())
    return sizes, categories, titles, stretched


STRUCTURAL_MESH = ("Floor", "Ceiling", "Wall", "Baseboard", "Pocket")


def quaternion_matrix(rotation):
    x, y, z, w = [float(v) for v in rotation]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def scene_rooms(archive, house):
    scene = json.loads(archive.read("3D-FRONT/%s.json" % house))
    meshes = {}
    for mesh in scene.get("mesh", []):
        if not str(mesh.get("type", "")).startswith(STRUCTURAL_MESH):
            continue
        points = np.asarray(mesh.get("xyz", []), dtype=float)
        if points.size >= 3:
            meshes[mesh["uid"]] = points.reshape(-1, 3)
    rooms = {}
    for room in scene.get("scene", {}).get("room", []):
        gathered = []
        for child in room.get("children", []):
            points = meshes.get(child.get("ref"))
            if points is None:
                continue
            scaled = points * np.asarray(child.get("scale", [1, 1, 1]), dtype=float)
            rotated = scaled @ quaternion_matrix(child.get("rot", [0, 0, 0, 1])).T
            gathered.append(rotated + np.asarray(child.get("pos", [0, 0, 0]), dtype=float))
        if gathered:
            joined = np.concatenate(gathered, axis=0)
            rooms[room.get("instanceid")] = float((joined.max(0) - joined.min(0)).max())
    return rooms


def room_scale(objects, sizes, instances=None):
    pooled = []
    matched = 0
    instances = instances or {}
    for item in objects:
        found = UUID.search(item["object_id"])
        if not found or found.group(0) not in sizes:
            continue
        ours = np.sort(np.asarray(item["bbox_world"]["size"], dtype=float))
        usable = ours > 1e-6
        if usable.sum() < 3:
            continue
        candidates = instances.get(found.group(0)) or [sizes[found.group(0)]]
        best = None
        for candidate in candidates:
            real = np.sort(np.asarray(candidate, dtype=float))
            ratios = real[usable] / ours[usable]
            deviation = float(np.max(ratios) / max(np.min(ratios), 1e-9))
            if best is None or deviation < best[0]:
                best = (deviation, ratios.tolist())
        matched += 1
        pooled.extend(best[1])
    if not pooled:
        return None, 0, None
    pooled = np.asarray(pooled, dtype=float)
    scale = float(np.median(pooled))
    deviation = float(np.mean(np.abs(pooled - scale)))
    spread = deviation / max(scale, 1e-9) if len(pooled) > 1 else 0.0
    return scale, matched, spread


def class_names(categories, titles):
    by_category = collections.defaultdict(collections.Counter)
    for jid, category in categories.items():
        if jid in titles:
            by_category[category][titles[jid]] += 1
    names = {}
    for category, counted in by_category.items():
        names[category] = counted.most_common(1)[0][0]
    return names


def training_classes(fine, instances, minimum):
    counted = collections.Counter()
    for jid, name in fine.items():
        counted[name.split("/")[0].strip()] += instances.get(jid, 0)
    kept = {name for name, count in counted.items() if count >= minimum}
    assignment = {}
    for jid, name in fine.items():
        head = name.split("/")[0].strip()
        assignment[jid] = head if head in kept else TAIL_CLASS
    return assignment, sorted(kept | {TAIL_CLASS}), counted


def main() -> None:
    args = parse_args()
    scene_root = args.scene_root.resolve()
    archive = zipfile.ZipFile(args.archive)
    available = {name.split("/")[-1][:-5] for name in archive.namelist()
                 if name.endswith(".json")}

    scales, diagnostics = {}, {}
    categories, titles = {}, {}
    instances = collections.Counter()
    geometry_scales = {}
    rejected = collections.Counter()
    for house_dir in sorted(p for p in scene_root.iterdir() if p.is_dir()):
        house = house_dir.name
        if house not in available:
            rejected["house_not_in_archive"] += 1
            continue
        sizes, house_categories, house_titles, stretched = scene_furniture(archive, house)
        try:
            extents = scene_rooms(archive, house)
        except Exception:
            extents = {}
        categories.update(house_categories)
        titles.update(house_titles)
        for room_dir in sorted(p for p in house_dir.iterdir() if p.is_dir()):
            room_id = "%s/%s" % (house, room_dir.name)
            try:
                _, _, objects = room_geometry(room_dir, scene_root)
            except Exception:
                rejected["room_geometry_failed"] += 1
                continue
            for item in objects:
                found = UUID.search(item["object_id"])
                if found:
                    instances[found.group(0)] += 1
            scale, count, spread = room_scale(objects, sizes, stretched.get(room_dir.name))
            extent = extents.get(room_dir.name)
            geometry = extent / NORMALISED_SHELL if extent else None
            if geometry:
                geometry_scales[room_id] = geometry
            diagnostics[room_id] = {"objects": count, "spread": spread, "scale": scale,
                                    "geometry_scale": geometry}
            if scale is None or count < args.minimum_objects:
                rejected["too_few_objects"] += 1
                continue
            if spread is not None and spread > args.maximum_spread:
                rejected["inconsistent_objects"] += 1
                continue
            scales[room_id] = scale

    names = class_names(categories, titles)
    classes = {jid: names[category] for jid, category in categories.items()
               if category in names}
    training, training_names, per_head = training_classes(
        classes, instances, args.minimum_class_instances)
    payload = {
        "source_archive": str(args.archive),
        "scales": scales,
        "geometry_scales": geometry_scales,
        "scale_diagnostics": diagnostics,
        "fine_classes": classes,
        "fine_class_names": sorted(set(classes.values())),
        "training_classes": training,
        "class_names": training_names,
        "instances_per_head": dict(per_head.most_common()),
        "minimum_class_instances": args.minimum_class_instances,
        "rooms_with_exact_scale": len(scales),
        "rooms_with_geometry_scale": len(geometry_scales),
        "rooms_seen": len(diagnostics),
        "rejected": dict(rejected),
        "minimum_objects": args.minimum_objects,
        "maximum_spread": args.maximum_spread,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    summary = {key: payload[key] for key in
               ("rooms_with_exact_scale", "rooms_with_geometry_scale", "rooms_seen",
                "rejected", "class_names")}
    summary["training_classes"] = payload["class_names"]
    summary["fine_class_count"] = len(payload["fine_class_names"])
    summary["models_with_a_class"] = len(classes)
    del summary["class_names"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
