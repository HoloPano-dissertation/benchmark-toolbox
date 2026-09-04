"""Recover a room envelope from semantic structure, never the global asset bbox.

The input filenames do not reliably separate floors, walls and ceilings. Select
room-facing planes near furniture anchors; fail rather than invent an enclosure.
"""

from pathlib import Path

import numpy as np
from shapely.geometry import Point, box, mapping
from shapely import set_precision

from glb_geometry import glb_triangles, glb_bounds, floor_footprint


SHELL_MARGIN = 0.02


def room_shell_footprint(room_dir, margin=SHELL_MARGIN):
    room = Path(room_dir)
    shell = room.parent / (room.name + ".glb")
    if not shell.is_file():
        return None
    lower, upper = glb_bounds(shell)
    span = [float(upper[axis] - lower[axis]) for axis in range(2)]
    grow = [value * margin for value in span]
    return box(float(lower[0]) - grow[0], float(lower[1]) - grow[1],
               float(upper[0]) + grow[0], float(upper[1]) + grow[1])


def clip_patches_to_shell(patches, shell):
    kept = []
    for patch in patches:
        clipped = patch["polygon"].intersection(shell)
        if clipped.is_empty or clipped.area <= 1e-8:
            continue
        for piece in (list(clipped.geoms) if clipped.geom_type == "MultiPolygon"
                      else [clipped]):
            if piece.is_empty or piece.area <= 1e-8 or piece.geom_type != "Polygon":
                continue
            kept.append({**patch, "polygon": piece, "area": float(piece.area),
                         "fragment_area": float(clipped.area - piece.area)})
    return kept


def structural_planes(room_dir):
    """Large horizontal patches; a file called ceil.glb may contain an entire room."""
    room = Path(room_dir)
    pieces, source_ids, names = [], [], []
    for stem in ("floor", "ceil", "wall", "others"):
        path = room / f"{stem}.glb"
        if path.is_file():
            triangles = glb_triangles(path)
            pieces.append(triangles)
            source_ids.extend([len(names)] * len(triangles))
            names.append(path.name)
    if not pieces:
        raise ValueError("No structural GLB geometry")
    triangles = np.concatenate(pieces)
    sources = np.asarray(source_ids)
    cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    area = np.abs(cross[:, 2]) / 2
    span_z = np.ptp(triangles[:, :, 2])
    tolerance = max(span_z*1e-5, 1e-6)
    horizontal = (np.ptp(triangles[:, :, 2], axis=1) < tolerance) & (area > 1e-9)
    t, a, s = triangles[horizontal], area[horizontal], sources[horizontal]
    up = cross[horizontal, 2] > 0
    z = t[:, :, 2].mean(axis=1)
    order = np.argsort(z)
    clusters = []
    for index in order:
        if not clusters or z[index]-z[clusters[-1][0]] > tolerance:
            clusters.append([])
        clusters[-1].append(index)
    patches = []
    for group in clusters:
        for facing_up in (True, False):
            indices = [i for i in group if up[i] == facing_up]
            if not indices or a[indices].sum() < a.sum()*0.0001:
                continue
            footprint, _ = floor_footprint(t[indices])
            footprint = set_precision(footprint, max(tolerance*0.1, 1e-7))
            components = list(footprint.geoms) if footprint.geom_type == "MultiPolygon" else [footprint]
            for polygon in components:
                if polygon.is_empty or polygon.area <= 1e-8:
                    continue
                patches.append({"z": float(np.average(z[indices], weights=a[indices])),
                                "polygon": polygon, "area": polygon.area,
                                "facing": "up" if facing_up else "down",
                                "fragment_area": footprint.area-polygon.area,
                                "sources": sorted({names[i] for i in s[indices]})})
    if not patches:
        raise ValueError("No horizontal structural planes")
    return patches, triangles


def recover_layout(room_dir):
    patches, triangles = structural_planes(room_dir)
    shell = room_shell_footprint(room_dir)
    if shell is not None:
        clipped = clip_patches_to_shell(patches, shell)
        if clipped:
            patches = clipped
    object_paths = [p for p in Path(room_dir).glob("*.glb")
                    if p.stem not in {"floor", "ceil", "wall", "others"}]
    boxes = [glb_bounds(p) for p in object_paths]
    if not boxes:
        raise ValueError("No furniture anchors to identify the intended room")
    centers = np.array([(lo+hi)/2 for lo, hi in boxes])
    anchor = np.median(centers[:, :2], axis=0)
    scale = max(float(np.median([max(hi[:2]-lo[:2]) for lo, hi in boxes])),
                float(max(np.ptp(centers[:, :2], axis=0)))*0.5, 0.2)
    anchor_point = Point(anchor)
    near = [p for p in patches if p["polygon"].distance(anchor_point) < scale*0.6]
    floor_options = [p for p in near if p["facing"] == "up"]
    if not floor_options:
        raise ValueError("No floor patch near the room furniture")
    max_floor_area = max(p["area"] for p in floor_options)
    floor_options = [p for p in floor_options if p["area"] >= max_floor_area*0.4]
    floor = min(floor_options, key=lambda p: p["z"])
    furniture_tops = [hi[2] for path, (lo, hi) in zip(object_paths, boxes)
                      if not path.name.startswith("Lighting_")]
    # Wide shelves/steps in others.glb can resemble a low suspended ceiling.
    # A ceiling below the typical furniture top cannot enclose that furniture.
    minimum_ceiling_z = max(floor["z"]+0.2*scale,
                           float(np.median(furniture_tops)) if furniture_tops else floor["z"])
    ceiling_options = [p for p in near if p["facing"] == "down" and p["z"] > minimum_ceiling_z]
    # Require spatial agreement, not just large area: unrelated neighboring
    # ceilings can be present in the same structural GLB.
    ceiling_options = [p for p in ceiling_options
                       if p["polygon"].intersection(floor["polygon"]).area >= floor["area"]*0.4]
    if not ceiling_options:
        raise ValueError("No matching ceiling patch; explicit reconstruction is required")
    # A higher, larger slab can coexist with a real suspended ceiling below it.
    # Prefer the lowest substantial room-facing patch, not maximum slab area.
    # The 40% overlap condition above rejects narrow wall caps and small trim.
    ceiling = min(ceiling_options, key=lambda p: p["z"])
    polygon, roof = floor["polygon"], ceiling["polygon"]
    simplification = (ceiling["z"]-floor["z"])*0.0005
    simplified = polygon.simplify(simplification, preserve_topology=True)
    if polygon.intersection(simplified).area/polygon.union(simplified).area >= 0.999:
        polygon = simplified
    else:
        simplification = 0.0
    overlap = polygon.intersection(roof)
    if overlap.is_empty or overlap.area < max(polygon.area, roof.area)*0.4:
        raise ValueError("Dominant floor/ceiling planes disagree spatially")
    warnings = []
    if polygon.interiors:
        warnings.append("floor_plan_has_holes")
    if floor["fragment_area"] > polygon.area*0.01:
        warnings.append("small_disconnected_floor_fragments")
    iou = overlap.area / polygon.union(roof).area
    if iou < 0.95:
        warnings.append("floor_ceiling_footprints_differ")
    return {
        "layout_version": "oriented-structural-envelope",
        "polygon": mapping(polygon), "camera_region": mapping(overlap),
        "floor_z": floor["z"], "ceiling_z": ceiling["z"],
        "sources": {"footprint": "horizontal floor patch in " + ", ".join(floor["sources"]),
                    "floor_height": floor["sources"], "ceiling_height": ceiling["sources"]},
        "clipped_to_room_shell": shell is not None,
        "reconstruct_floor": False, "reconstruct_ceiling": False,
        "floor_ceiling_iou": iou, "warnings": warnings,
        "area": float(polygon.area),
        "furniture_anchor": anchor.tolist(),
        "footprint_simplification_tolerance": simplification,
        "bounds_min": [polygon.bounds[0], polygon.bounds[1], floor["z"]],
        "bounds_max": [polygon.bounds[2], polygon.bounds[3], ceiling["z"]],
    }
