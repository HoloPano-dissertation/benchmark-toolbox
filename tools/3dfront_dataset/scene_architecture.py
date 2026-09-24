#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3dfront_panorama_renderer"))
from glb_geometry import glb_triangles  # noqa: E402

INTERIOR = ("WallInner", "Floor", "Ceiling", "CustomizedCeiling", "CustomizedFeatureWall",
            "Baseboard", "CustomizedPlatform", "CustomizedBackgroundModel", "Cabinet",
            "CustomizedFurniture", "Front", "Back", "Hole", "Pocket")
DEFAULT_COLOUR = (0.78, 0.78, 0.78)
CEILING_TYPES = frozenset({"Ceiling", "CustomizedCeiling"})


def face_the_room(points, faces, centre):
    if not len(faces):
        return faces
    corners = points[faces]
    normals = np.cross(corners[:, 1]-corners[:, 0], corners[:, 2]-corners[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    lengths[lengths == 0] = 1.0
    normals = normals / lengths[:, None]
    middles = corners.mean(axis=1)
    horizontal = np.abs(normals[:, 2]) > 0.8
    target = middles - centre
    target[:, 2] = 0.0
    lateral = np.linalg.norm(target, axis=1)
    lateral[lateral == 0] = 1.0
    target = -target / lateral[:, None]                      # walls look inwards
    target[horizontal] = 0.0
    target[horizontal, 2] = np.where(middles[horizontal, 2] > centre[2], -1.0, 1.0)
    flip = (normals*target).sum(axis=1) < 0
    faces = faces.copy()
    faces[flip] = faces[flip][:, ::-1]
    return faces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", type=Path, help="3D-FRONT.zip of the original release")
    parser.add_argument("textures", type=Path, help="3D-FRONT-texture.zip")
    parser.add_argument("scene_root", type=Path, help="Root of the processed GLB rooms")
    parser.add_argument("room_id", help="<house>/<room>, or - with --rooms")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--rooms", type=Path,
                        help="Do every room of this list instead of one: a JSON array "
                             "of <house>/<room>, or one id per line. Each room is "
                             "written to <output dir>/<house>/<room>/")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--metadata", type=Path,
                        help="source_metadata.json, for the exact metric scale")
    parser.add_argument("--types", nargs="*", default=list(INTERIOR))
    return parser.parse_args()


def room_of(scene, room_name):
    for room in scene.get("scene", {}).get("room", []):
        if room.get("instanceid") == room_name:
            return room
    raise KeyError("Room %s is not in this scene" % room_name)


def placement(scene, room, scene_root, room_id, scale):
    house, room_name = room_id.split("/")
    furniture = {item["uid"]: item for item in scene.get("furniture", [])}
    room_dir = Path(scene_root) / house / room_name
    dx, dz, floors = [], [], []
    for child in room.get("children", []):
        item = furniture.get(child.get("ref"))
        if not item or not item.get("jid"):
            continue
        matches = [p for p in room_dir.glob("*.glb") if item["jid"] in p.name]
        if not matches:
            continue
        points = glb_triangles(matches[0]).reshape(-1, 3)
        lo, hi = points.min(axis=0), points.max(axis=0)
        centre = (lo + hi) / 2.0
        position = np.asarray(child["pos"], dtype=float)
        dx.append(position[0] - centre[0]*scale)
        dz.append(position[2] + centre[1]*scale)
        if abs(position[1]) < 1e-6:
            floors.append(lo[2])
    if not dx:
        raise ValueError("No furniture is common to both sources for " + room_id)
    floor_z = float(np.median(floors)) if floors else None
    return {"scale": scale, "cx": float(np.median(dx)), "cz": float(np.median(dz)),
            "cy": 0.0 - (floor_z*scale if floor_z is not None else 0.0),
            "matched": len(dx), "floor_from": len(floors)}


def to_room(points, place):
    s = place["scale"]
    x = (points[:, 0] - place["cx"]) / s
    y = -(points[:, 2] - place["cz"]) / s
    z = (points[:, 1] - place["cy"]) / s
    return np.stack((x, y, z), axis=1)


def texture_index(archive):
    index = {}
    for name in archive.namelist():
        parts = name.split("/")
        if len(parts) >= 3 and parts[-1].startswith("texture."):
            index[parts[1]] = name
    return index


def flat_level(corners):
    if np.ptp(corners[:, 2]) > 1e-4:
        return None
    return round(float(corners[:, 2].mean()), 4)


def facing_the_room_only(kind, faces, normals, vertex_count):
    if kind not in CEILING_TYPES:
        return faces
    if not normals or len(normals) != vertex_count*3:
        return faces
    vertex_normals = np.asarray(normals, dtype=float).reshape(-1, 3)
    facing = vertex_normals[faces][:, :, 1].mean(axis=1)      # ось Y сцены — вверх
    kept = faces[facing < -0.1]
    return kept if len(kept) else faces


def thin_coincident_ceilings(surfaces):
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    shapes = {}
    for index, (kind, points, faces, _) in enumerate(surfaces):
        if kind not in CEILING_TYPES:
            continue
        for number, face in enumerate(faces):
            if flat_level(points[face]) is None:
                continue
            polygon = Polygon(points[face][:, :2])
            if not polygon.is_valid:
                polygon = polygon.buffer(0)               # самопересечения в исходнике
            if polygon.is_valid and polygon.area > 1e-9:
                shapes[(index, number)] = polygon
    if len(shapes) < 2:
        return surfaces
    levels = {}
    for index, (kind, points, faces, _) in enumerate(surfaces):
        if kind not in CEILING_TYPES:
            continue
        for number, face in enumerate(faces):
            level = flat_level(points[face])
            if level is not None:
                levels.setdefault(level, set()).add(index)
    shared = {level for level, owners in levels.items() if len(owners) > 1}
    dropped = set()
    for (index, number), polygon in shapes.items():
        level = flat_level(surfaces[index][1][surfaces[index][2][number]])
        if level not in shared:
            continue
        try:
            others = unary_union([shape for key, shape in shapes.items()
                                  if key != (index, number) and key not in dropped])
            uncovered = polygon.difference(others).area
        except Exception:
            # вырожденная геометрия исходника: решить, закрыто ли место, нечем —
            # грань остаётся, потому что дыра в потолке хуже лишней поверхности
            continue
        if uncovered <= polygon.area*0.02:
            dropped.add((index, number))
    if not dropped:
        return surfaces
    thinned = []
    for index, (kind, points, faces, mesh) in enumerate(surfaces):
        keep = [face for number, face in enumerate(faces) if (index, number) not in dropped]
        kept = np.asarray(keep, dtype=int).reshape(-1, 3) if keep else faces
        thinned.append((kind, points, kept, mesh))
    return thinned


def room_centre(scene, room, place, types):
    meshes = {item["uid"]: item for item in scene.get("mesh", [])}
    lows, highs = [], []
    for child in room.get("children", []):
        mesh = meshes.get(child.get("ref"))
        if not mesh or mesh.get("type") not in types or not mesh.get("xyz"):
            continue
        points = to_room(np.asarray(mesh["xyz"], dtype=float).reshape(-1, 3), place)
        lows.append(points.min(axis=0))
        highs.append(points.max(axis=0))
    if not lows:
        return np.zeros(3)
    return (np.min(lows, axis=0) + np.max(highs, axis=0)) / 2.0


def write_room(scene, room, place, materials, textures, index, types, output_dir,
               texture_dir=None, texture_prefix=""):
    output_dir.mkdir(parents=True, exist_ok=True)
    meshes = {item["uid"]: item for item in scene.get("mesh", [])}
    obj_lines, mtl_lines, written = [], [], {}
    offset = uv_offset = 1
    kept = {}
    centre = room_centre(scene, room, place, types)
    surfaces = []
    for child in room.get("children", []):
        mesh = meshes.get(child.get("ref"))
        if not mesh or mesh.get("type") not in types:
            continue
        xyz = np.asarray(mesh["xyz"], dtype=float).reshape(-1, 3)
        faces = np.asarray(mesh["faces"], dtype=int).reshape(-1, 3)
        if not len(xyz) or not len(faces):
            continue
        faces = facing_the_room_only(mesh["type"], faces, mesh.get("normal"), len(xyz))
        points = to_room(xyz, place)
        surfaces.append((mesh["type"], points, face_the_room(points, faces, centre), mesh))
    for _, points, faces, mesh in thin_coincident_ceilings(surfaces):
        if not len(faces):
            continue
        xyz = np.asarray(mesh["xyz"], dtype=float).reshape(-1, 3)
        uv = np.asarray(mesh.get("uv") or [], dtype=float).reshape(-1, 2)
        material = materials.get(mesh.get("material"))
        name = material_name(material, index, textures, output_dir, written, mtl_lines,
                             texture_dir, texture_prefix)
        kind = mesh["type"]
        kept[kind] = kept.get(kind, 0) + 1
        obj_lines.append("o %s_%d" % (kind, kept[kind]))
        obj_lines.append("usemtl %s" % name)
        for point in points:
            obj_lines.append("v %.6f %.6f %.6f" % tuple(point))
        for pair in uv:
            obj_lines.append("vt %.6f %.6f" % tuple(pair))
        textured = len(uv) == len(xyz)
        for face in faces:
            if textured:
                obj_lines.append("f %d/%d %d/%d %d/%d" % (
                    face[0]+offset, face[0]+uv_offset, face[1]+offset, face[1]+uv_offset,
                    face[2]+offset, face[2]+uv_offset))
            else:
                obj_lines.append("f %d %d %d" % (face[0]+offset, face[1]+offset, face[2]+offset))
        offset += len(xyz)
        uv_offset += len(uv)
    if not kept:
        raise ValueError("The scene holds no interior surfaces for this room")
    (output_dir / "architecture.mtl").write_text("\n".join(mtl_lines) + "\n", encoding="utf-8")
    (output_dir / "architecture.obj").write_text(
        "mtllib architecture.mtl\n" + "\n".join(obj_lines) + "\n", encoding="utf-8")
    return kept


def material_name(material, index, textures, output_dir, written, mtl_lines,
                  texture_dir=None, texture_prefix=""):
    key = (material or {}).get("uid", "plain")
    if key in written:
        return written[key]
    name = "m%d" % len(written)
    written[key] = name
    jid = (material or {}).get("jid", "")
    colour = (material or {}).get("color") or []
    rgb = tuple(c/255.0 for c in colour[:3]) if len(colour) >= 3 else DEFAULT_COLOUR
    mtl_lines.append("newmtl %s" % name)
    mtl_lines.append("Kd %.4f %.4f %.4f" % rgb)
    mtl_lines.append("Ks 0 0 0")
    if jid and jid in index:
        image = texture_dir if texture_dir is not None else output_dir / "textures"
        image.mkdir(parents=True, exist_ok=True)
        path = image / (jid + Path(index[jid]).suffix)
        if not path.is_file():
            path.write_bytes(textures.read(index[jid]))
        mtl_lines.append("map_Kd %s%s" % (texture_prefix or "textures/", path.name))
    mtl_lines.append("")
    return name


def wanted_rooms(args):
    if not args.rooms:
        return [args.room_id]
    text = args.rooms.read_text(encoding="utf-8")
    listed = json.loads(text) if args.rooms.suffix == ".json" else text.split()
    rooms = sorted({str(value).strip().strip("/") for value in listed})
    return rooms[args.shard_index::args.shard_count]


def build_one(room_id, scene, args, table, textures, index, output_dir,
              texture_dir=None, texture_prefix=""):
    house, room_name = room_id.split("/")
    room = room_of(scene, room_name)
    scale = table.get("scales", {}).get(room_id) \
        or table.get("geometry_scales", {}).get(room_id) or 1.0
    place = placement(scene, room, args.scene_root, room_id, scale)
    materials = {item["uid"]: item for item in scene.get("material", [])}
    kept = write_room(scene, room, place, materials, textures, index,
                      set(args.types), output_dir, texture_dir, texture_prefix)
    report = {"room_id": room_id, "placement": place, "surfaces": kept,
              "output": str(output_dir)}
    (output_dir / "architecture.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    rooms = wanted_rooms(args)
    scenes = zipfile.ZipFile(args.scenes)
    textures = zipfile.ZipFile(args.textures)
    index = texture_index(textures)
    table = json.loads(Path(args.metadata).read_text()) if args.metadata else {}
    done, failed = 0, []
    scene, loaded = None, None
    for room_id in rooms:
        house = room_id.split("/")[0]
        output_dir = args.output_dir / room_id if args.rooms else args.output_dir
        try:
            if house != loaded:
                scene = json.loads(scenes.read("3D-FRONT/%s.json" % house))
                loaded = house
            shared = args.output_dir / "textures" if args.rooms else None
            report = build_one(room_id, scene, args, table, textures, index, output_dir,
                               shared, "../../textures/" if args.rooms else "")
            done += 1
            if not args.rooms:
                print(json.dumps(report, indent=2, ensure_ascii=False))
        except Exception as error:
            failed.append({"room_id": room_id, "error": "%s: %s"
                           % (type(error).__name__, str(error)[:200])})
            print("FAILED %s: %s" % (room_id, error), flush=True)
    if args.rooms:
        summary = {"rooms": len(rooms), "built": done, "failed": failed}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / ("summary-%03d.json" % args.shard_index)).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"rooms": len(rooms), "built": done, "failed": len(failed)},
                         ensure_ascii=False))
    if failed and args.rooms is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
