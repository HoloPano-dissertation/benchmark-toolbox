#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "3dfront_panorama_renderer"))
from glb_geometry import glb_triangles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--scene-root", type=Path)
    return parser.parse_args()


def write_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    body = vertices.astype("<f4").tobytes()
    face_bytes = bytearray()
    for face in faces:
        face_bytes += struct.pack("<B3i", 3, int(face[0]), int(face[1]), int(face[2]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + body + bytes(face_bytes))


def object_id(source_glb: str) -> str:
    return "__".join(Path(source_glb).with_suffix("").parts)


def unit_mesh(triangles: np.ndarray, bbox: dict, camera: np.ndarray, scale: float):
    center = np.asarray(bbox["center"], dtype=float)
    size = np.asarray(bbox["size"], dtype=float)
    basis = np.asarray(bbox["basis"], dtype=float)
    points = triangles.reshape(-1, 3) * scale - camera
    local = (points - center) @ basis.T
    vertices = local / np.maximum(size, 1e-9)
    faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    return vertices, faces


def main() -> None:
    args = parse_args()
    root = args.experiment_root.resolve()
    scene_root = (args.scene_root or root / "source" / "3D-FRONT-TEST-SCENE").resolve()
    shape_root = root / "objects"
    written: dict[str, Path] = {}
    failures: list[dict] = []
    per_class: Counter[str] = Counter()
    linked = 0

    for split in ("train", "val", "test"):
        records = [
            json.loads(line)
            for line in (root / "manifests_gt" / f"{split}.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        for record in records:
            path = Path(record["ground_truth"])
            scene = json.loads(path.read_text(encoding="utf-8"))
            metadata = scene["metadata"]
            camera = np.asarray(metadata["camera_location_world"], dtype=float)
            scale = float(metadata["metric_scale"])
            changed = False
            for item in scene["objects"]:
                source = item["attributes"].get("source_glb")
                if source is None:
                    continue
                target = (shape_root / item["label"] / object_id(source)
                          / "mesh.ply")
                if source not in written:
                    try:
                        triangles = glb_triangles(scene_root / source)
                        vertices, faces = unit_mesh(triangles, item["bbox"], camera, scale)
                        write_ply(target, vertices, faces)
                        written[source] = target
                        per_class[item["label"]] += 1
                    except Exception as error:
                        failures.append({"source_glb": source, "error": str(error)[:200]})
                        continue
                item["attributes"]["shape"] = str(written[source])
                linked += 1
                changed = True
            if changed:
                path.write_text(json.dumps(scene, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")

    status = {
        "ready": not failures,
        "meshes": len(written),
        "meshes_per_class": dict(sorted(per_class.items())),
        "objects_linked_across_views": linked,
        "frame": "unit box of the instance; multiply by size, rotate by basis, add center",
        "layout": "objects/<class>/<object_id>, the layout the DPC object dataset expects",
        "watertight": False,
        "failures": failures[:20],
        "failure_count": len(failures),
    }
    (root / "state" / "shape.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))
    if failures:
        raise SystemExit("Shape export failed for %d source meshes" % len(failures))


if __name__ == "__main__":
    main()
