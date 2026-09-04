#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
import struct
from collections import Counter
from pathlib import Path
from typing import Iterator

import numpy as np

from metric_scale import (
    DEFAULT_REFERENCE_HEIGHT,
    NORMALISED_SHELL_EXTENT,
    load_scale_table,
    resolve_scale,
    scale_layout_geometry,
)


JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942
COMPONENT_DTYPES = {
    5120: np.dtype("i1"),
    5121: np.dtype("u1"),
    5122: np.dtype("<i2"),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
    5126: np.dtype("<f4"),
}
TYPE_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
TAIL_CLASS = "other"
UUID_ONLY = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)
UUID_SUFFIX = re.compile(
    r"_(?=[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_[0-9]+$)",
    re.IGNORECASE,
)
STRUCTURAL_STEMS = {"ceil", "floor", "wall", "others"}
CLASS_MAP = {
    "Bed": "bed",
    "Cabinet_Shelf_Desk": "cabinet_shelf_desk",
    "Chair": "chair",
    "Lighting": "lighting",
    "Others": "other",
    "Pier_Stool": "stool",
    "Sofa": "sofa",
    "Table": "table",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--scene-root", type=Path)
    parser.add_argument(
        "--scale-table", type=Path,
        help="Exact per-room scales derived from the original 3D-FRONT",
    )
    parser.add_argument(
        "--reference-height", type=float, default=DEFAULT_REFERENCE_HEIGHT,
        help="Ceiling height in metres assumed by the anchor",
    )
    parser.add_argument(
        "--class-table", type=Path,
        help="Metadata of the original 3D-FRONT holding the object classes")
    parser.add_argument(
        "--allow-unscalable", action="store_true",
        help="Record rooms that fail a scale check instead of stopping",
    )
    return parser.parse_args()


def read_glb(path: Path) -> tuple[dict, bytes]:
    with path.open("rb") as source:
        magic, version, total_length = struct.unpack("<4sII", source.read(12))
        if magic != b"glTF" or version != 2:
            raise ValueError(f"Unsupported GLB header: {path}")
        document = None
        binary = None
        consumed = 12
        while consumed < total_length:
            chunk_length, chunk_type = struct.unpack("<II", source.read(8))
            chunk = source.read(chunk_length)
            consumed += 8 + chunk_length
            if chunk_type == JSON_CHUNK:
                document = json.loads(chunk.decode("utf-8").rstrip("\x00 \t\r\n"))
            elif chunk_type == BIN_CHUNK:
                binary = chunk
    if document is None or binary is None:
        raise ValueError(f"GLB must contain JSON and BIN chunks: {path}")
    return document, binary


def accessor_array(document: dict, binary: bytes, accessor_index: int) -> np.ndarray:
    accessor = document["accessors"][accessor_index]
    if "sparse" in accessor:
        raise ValueError("Sparse glTF accessors are not supported")
    view = document["bufferViews"][accessor["bufferView"]]
    dtype = COMPONENT_DTYPES[accessor["componentType"]]
    components = TYPE_COMPONENTS[accessor["type"]]
    offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    stride = int(view.get("byteStride", components * dtype.itemsize))
    return np.ndarray(
        shape=(int(accessor["count"]), components),
        dtype=dtype,
        buffer=binary,
        offset=offset,
        strides=(stride, dtype.itemsize),
    ).copy()


def quaternion_matrix(rotation: list[float]) -> np.ndarray:
    x, y, z, w = (float(value) for value in rotation)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        return np.eye(4)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )


def node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T
    translation = np.eye(4)
    translation[:3, 3] = np.asarray(node.get("translation", [0, 0, 0]), dtype=float)
    scale = np.eye(4)
    scale[np.arange(3), np.arange(3)] = np.asarray(
        node.get("scale", [1, 1, 1]), dtype=float
    )
    return translation @ quaternion_matrix(node.get("rotation", [0, 0, 0, 1])) @ scale


def mesh_nodes(document: dict) -> Iterator[tuple[int, np.ndarray]]:
    nodes = document.get("nodes", [])

    def visit(index: int, parent: np.ndarray) -> Iterator[tuple[int, np.ndarray]]:
        node = nodes[index]
        world = parent @ node_matrix(node)
        if "mesh" in node:
            yield int(node["mesh"]), world
        for child in node.get("children", []):
            yield from visit(int(child), world)

    scene_index = int(document.get("scene", 0))
    for root in document.get("scenes", [{}])[scene_index].get("nodes", []):
        yield from visit(int(root), np.eye(4))


def glb_vertices(path: Path) -> np.ndarray:
    document, binary = read_glb(path)
    parts = []
    for mesh_index, transform in mesh_nodes(document):
        mesh = document["meshes"][mesh_index]
        for primitive in mesh.get("primitives", []):
            position_index = primitive.get("attributes", {}).get("POSITION")
            if position_index is None:
                continue
            points = accessor_array(document, binary, int(position_index)).astype(np.float64)
            homogeneous = np.column_stack((points, np.ones(len(points))))
            parts.append((homogeneous @ transform.T)[:, :3])
    if not parts:
        raise ValueError(f"No POSITION data in {path}")
    gltf_points = np.concatenate(parts, axis=0)
    return np.column_stack(
        (gltf_points[:, 0], -gltf_points[:, 2], gltf_points[:, 1])
    )


def label_for(path: Path) -> str | None:
    if path.stem in STRUCTURAL_STEMS:
        return None
    match = UUID_SUFFIX.search(path.stem)
    raw = path.stem[: match.start()] if match else path.stem
    if raw not in CLASS_MAP:
        raise ValueError(f"Unknown MIDI label '{raw}' in {path.name}")
    return CLASS_MAP[raw]


def oriented_box(points: np.ndarray, camera: np.ndarray) -> dict:
    xy = points[:, :2]
    centered = xy - xy.mean(axis=0)
    covariance = centered.T @ centered / max(len(centered), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis_x = eigenvectors[:, int(np.argmax(eigenvalues))]
    if axis_x[0] < 0 or (abs(axis_x[0]) < 1e-12 and axis_x[1] < 0):
        axis_x = -axis_x
    axis_y = np.array([-axis_x[1], axis_x[0]])
    projection_x = xy @ axis_x
    projection_y = xy @ axis_y
    low_x, high_x = float(projection_x.min()), float(projection_x.max())
    low_y, high_y = float(projection_y.min()), float(projection_y.max())
    low_z, high_z = float(points[:, 2].min()), float(points[:, 2].max())
    center_world = np.array(
        [
            *((low_x + high_x) / 2 * axis_x + (low_y + high_y) / 2 * axis_y),
            (low_z + high_z) / 2,
        ]
    )
    return {
        "center": (center_world - camera).tolist(),
        "size": [high_x - low_x, high_y - low_y, high_z - low_z],
        "basis": [
            [float(axis_x[0]), float(axis_x[1]), 0.0],
            [float(axis_y[0]), float(axis_y[1]), 0.0],
            [0.0, 0.0, 1.0],
        ],
    }


def room_geometry(room_dir: Path, scene_root: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    structural_points = []
    all_points = []
    objects = []
    for glb_path in sorted(room_dir.glob("*.glb")):
        points = glb_vertices(glb_path)
        all_points.append(points)
        label = label_for(glb_path)
        if label is None:
            structural_points.append(points)
            continue
        objects.append(
            {
                "object_id": glb_path.stem,
                "label": label,
                "bbox_world": oriented_box(points, np.zeros(3)),
                "source_glb": glb_path.relative_to(scene_root).as_posix(),
            }
        )
    layout_points = structural_points or all_points
    if not layout_points:
        raise ValueError(f"No GLB geometry in {room_dir}")
    joined = np.concatenate(layout_points, axis=0)
    return joined.min(axis=0), joined.max(axis=0), objects


def main() -> None:
    args = parse_args()
    experiment_root = args.experiment_root.resolve()
    source_link = experiment_root / "source" / "3D-FRONT-TEST-SCENE"
    scene_root = (args.scene_root or source_link).resolve()
    source_manifests = experiment_root / "manifests"
    output_manifests = experiment_root / "manifests_gt"
    ground_truth_root = experiment_root / "ground_truth"
    output_manifests.mkdir(parents=True, exist_ok=True)
    category_counts: Counter[str] = Counter()
    room_cache: dict[str, tuple[np.ndarray, np.ndarray, list[dict]]] = {}
    split_counts: Counter[str] = Counter()
    object_count = 0
    scale_table = load_scale_table(args.scale_table) if args.scale_table else None
    class_table, fine_table, class_list = {}, {}, sorted(set(CLASS_MAP.values()))
    if args.class_table:
        metadata = json.loads(args.class_table.read_text(encoding="utf-8"))
        class_table = metadata.get("training_classes", {})
        fine_table = metadata.get("fine_classes", {})
        class_list = metadata.get("class_names") or class_list
    coarse_labels = Counter()
    scale_cache: dict[str, tuple[float, str]] = {}
    scale_reports: list[dict] = []
    unscalable: list[dict] = []
    needs_review: list[dict] = []

    for split in ("train", "val", "test"):
        output_records = []
        source_records = [
            json.loads(line)
            for line in (source_manifests / f"{split}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        for record in source_records:
            metadata = dict(record["metadata"])
            room_id = str(metadata["room_id"])
            view_index = int(metadata["view_index"])
            if room_id not in room_cache:
                room_cache[room_id] = room_geometry(scene_root / room_id, scene_root)
            layout_min, layout_max, room_objects = room_cache[room_id]
            render_dir = experiment_root / "outputs" / split / room_id
            render_path = render_dir / "render.json"
            if not render_path.is_file():
                render_path = render_dir / "pilot.json"
            render_metadata = json.loads(render_path.read_text(encoding="utf-8"))
            recovered_layout = render_metadata.get("layout")
            if recovered_layout is not None:
                layout_min = np.asarray(recovered_layout["bounds_min"])
                layout_max = np.asarray(recovered_layout["bounds_max"])
            if room_id not in scale_cache:
                scale, report = resolve_scale(
                    room_id, recovered_layout, table=scale_table,
                    source_shell_extent=NORMALISED_SHELL_EXTENT,
                    reference_height=args.reference_height, strict=False,
                )
                scale_reports.append(report)
                if report["rejected"]:
                    unscalable.append(report)
                if report["review"]:
                    needs_review.append(report)
                scale_cache[room_id] = (scale, report["source"])
            scale, scale_source = scale_cache[room_id]
            camera = np.asarray(render_metadata["camera_locations"][view_index], dtype=float)
            objects = []
            for item in room_objects:
                model = UUID_ONLY.search(item["object_id"])
                key = model.group(0) if model else None
                label = class_table.get(key)
                if label is None:
                    label = item["label"] if not class_table else TAIL_CLASS
                fine = fine_table.get(key)
                bbox = dict(item["bbox_world"])
                if min(bbox["size"]) <= 1e-7:
                    continue
                bbox["center"] = (
                    (np.asarray(bbox["center"], dtype=float) - camera) * scale
                ).tolist()
                bbox["size"] = (np.asarray(bbox["size"], dtype=float) * scale).tolist()
                attributes = {"source_glb": item["source_glb"]}
                if fine:
                    attributes["fine_class"] = fine
                objects.append(
                    {
                        "object_id": item["object_id"],
                        "label": label,
                        "score": 1.0,
                        "bbox": bbox,
                        "attributes": attributes,
                    }
                )
                category_counts[label] += 1
                coarse_labels[item["label"]] += 1
            object_count += len(objects)
            gt_path = ground_truth_root / split / room_id / f"{view_index}.json"
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            gt_path.write_text(
                json.dumps(
                    {
                        "layout": {
                            "min_corner": ((layout_min - camera) * scale).tolist(),
                            "max_corner": ((layout_max - camera) * scale).tolist(),
                        },
                        "objects": objects,
                        "relations": [],
                        "metadata": {
                            "dataset": "MIDI-3D-FRONT-1K-custom",
                            "coordinate_frame": "camera_centered_xyz_z_up",
                            "camera_location_world": (camera * scale).tolist(),
                            "units": "metres recovered by a per-room scale",
                            "metric_scale": scale,
                            "metric_scale_source": scale_source,
                            "layout_geometry_world": scale_layout_geometry(
                                recovered_layout, scale),
                            "layout_box_semantics": "enclosing AABB for legacy toolbox metrics; not exact polygon volume",
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            record["ground_truth"] = str(gt_path)
            output_records.append(record)
            split_counts[split] += 1
        (output_manifests / f"{split}.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in output_records),
            encoding="utf-8",
        )

    status = {
        "ready": True,
        "room_count": len(room_cache),
        "sample_counts": dict(split_counts),
        "object_instances_across_views": object_count,
        "category_instances_across_views": dict(sorted(category_counts.items())),
        "categories": class_list,
        "categories_seen": sorted(category_counts),
        "class_source": "original 3D-FRONT" if class_table else "MIDI filename prefix",
        "coarse_label_counts": dict(sorted(coarse_labels.items())),
        "coordinate_frame": "camera_centered_xyz_z_up",
        "box_type": "upright PCA-oriented",
        "layout_type": "recovered envelope AABB for legacy metrics; full polygon/plane sources in metadata",
        "units": "metres recovered by a per-room scale",
        "metric_scale": {
            "reference_height": args.reference_height,
            "table": str(args.scale_table) if args.scale_table else None,
            "rooms_from_table": sum(1 for r in scale_reports if r["source"] == "table"),
            "rooms_from_ceiling_anchor": sum(
                1 for r in scale_reports if r["source"] == "ceiling_height"),
            "unscalable_room_count": len(unscalable),
            "unscalable_rooms": [r["room_id"] for r in unscalable],
            "contour_review_room_count": len(needs_review),
        },
        "metric_scale_certified": bool(scale_table) and not unscalable and not any(
            report["source"] != "table" for report in scale_reports),
    }
    state_dir = experiment_root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "ground_truth.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    (state_dir / "classes.json").write_text(
        json.dumps({"classes": class_list,
                    "source": "original 3D-FRONT" if class_table else "MIDI filename prefix"},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (state_dir / "metric_scale.json").write_text(
        json.dumps({"reference_height": args.reference_height,
                    "rooms": sorted(scale_reports, key=lambda r: r["room_id"])},
                   indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(status, indent=2))
    if unscalable and not args.allow_unscalable:
        raise SystemExit(
            "Scale checks failed for %d room(s); exclude them deliberately or pass "
            "--allow-unscalable. Reasons are in state/metric_scale.json:\n%s"
            % (len(unscalable),
               "\n".join("  %s: %s" % (r["room_id"], "; ".join(r["rejected"]))
                          for r in unscalable[:20]))
        )


if __name__ == "__main__":
    main()
