#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from sdf_samples import (
    HALF_EXTENT,
    SAMPLE_COUNT,
    coarse_grid,
    near_surface_samples,
    occupancy,
    signed_field,
    uniform_samples,
    watertight_mesh,
    world_to_grid,
    write_grd,
    write_matrix,
    write_samples,
)
from export_shape import write_ply

LDIF_MESH_SCALE = 0.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--samples", type=int, default=SAMPLE_COUNT)
    return parser.parse_args()


def read_ply_triangles(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    end = raw.find(b"end_header")
    header = raw[:end].decode("ascii").splitlines()
    count = next(int(line.split()[2]) for line in header
                 if line.startswith("element vertex"))
    body = raw[raw.find(b"\n", end) + 1:]
    vertices = np.frombuffer(body[:count * 12], dtype="<f4").reshape(-1, 3)
    return vertices.reshape(-1, 3, 3).astype(float)


def object_transform(bbox: dict, camera: np.ndarray, scale: float) -> np.ndarray:
    size = np.asarray(bbox["size"], dtype=float)
    basis = np.asarray(bbox["basis"], dtype=float)
    center = np.asarray(bbox["center"], dtype=float)
    linear = np.diag(LDIF_MESH_SCALE / np.maximum(size, 1e-9)) @ basis
    matrix = np.eye(4)
    matrix[:3, :3] = linear * scale
    matrix[:3, 3] = -linear @ (center + camera)
    return matrix


def main() -> None:
    args = parse_args()
    root = args.experiment_root.resolve()
    tasks: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        for line in (root / "manifests_gt" / f"{split}.jsonl").read_text(
                encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            scene = json.loads(Path(record["ground_truth"]).read_text(encoding="utf-8"))
            camera = np.asarray(scene["metadata"]["camera_location_world"], dtype=float)
            scale = float(scene["metadata"]["metric_scale"])
            for item in scene["objects"]:
                shape = item["attributes"].get("shape")
                if shape and shape not in tasks:
                    tasks[shape] = {"bbox": item["bbox"], "camera": camera,
                                    "scale": scale, "label": item["label"]}

    ordered = sorted(tasks)
    selected = ordered[args.shard_index::args.shard_count]
    done: Counter[str] = Counter()
    failures = []
    for shape in selected:
        folder = Path(shape).parent
        try:
            triangles = read_ply_triangles(Path(shape)) * LDIF_MESH_SCALE
            solid, shell = occupancy(triangles)
            field = signed_field(solid, shell)
            points, values = near_surface_samples(triangles, field, count=args.samples)
            write_samples(folder / "nss_points.sdf", points, values)
            points, values = uniform_samples(field, count=args.samples)
            write_samples(folder / "uniform_points.sdf", points, values)
            write_grd(folder / "coarse_grid.grd", world_to_grid(), coarse_grid(field))
            vertices, faces = watertight_mesh(field)
            write_ply(folder / "mesh_watertight.ply", vertices, faces)
            write_matrix(folder / "orig_to_gaps.txt",
                         object_transform(tasks[shape]["bbox"], tasks[shape]["camera"],
                                          tasks[shape]["scale"]))
            done[tasks[shape]["label"]] += 1
        except Exception as error:
            failures.append({"shape": shape, "error": str(error)[:200]})

    status = {
        "ready": not failures,
        "shard": [args.shard_index, args.shard_count],
        "objects_total": len(ordered),
        "objects_done": sum(done.values()),
        "per_class": dict(sorted(done.items())),
        "samples_per_file": args.samples,
        "half_extent": HALF_EXTENT,
        "mesh_scale": LDIF_MESH_SCALE,
        "failures": failures[:10],
        "failure_count": len(failures),
    }
    state = root / "state" / "sdf"
    state.mkdir(parents=True, exist_ok=True)
    (state / ("shard-%03d.json" % args.shard_index)).write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))
    if failures:
        raise SystemExit("SDF export failed for %d meshes" % len(failures))


if __name__ == "__main__":
    main()
