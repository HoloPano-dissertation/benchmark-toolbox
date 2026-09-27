#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

SAMPLE_COUNT = 10000
NEIGHBOURS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--samples", type=int, default=SAMPLE_COUNT)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser.parse_args()


def read_ply_mesh(path):
    raw = path.read_bytes()
    end = raw.find(b"end_header")
    if end < 0:
        raise ValueError("Not a binary PLY: %s" % path)
    header = raw[:end].decode("ascii").splitlines()
    vertex_count = next(int(line.split()[2]) for line in header
                        if line.startswith("element vertex"))
    face_count = next((int(line.split()[2]) for line in header
                       if line.startswith("element face")), 0)
    body = raw[raw.find(b"\n", end) + 1:]
    vertices = np.frombuffer(body[:vertex_count * 12], dtype="<f4").reshape(-1, 3)
    faces = np.empty((face_count, 3), dtype=np.int64)
    position = vertex_count * 12
    for index in range(face_count):
        sides, first, second, third = struct.unpack_from("<B3i", body, position)
        if sides != 3:
            raise ValueError("Only triangles are supported, face %d has %d sides"
                             % (index, sides))
        faces[index] = (first, second, third)
        position += 13
    return vertices.astype(float), faces


def sample_surface(vertices, faces, count, seed):
    corners = vertices[faces]
    spans = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(*spans), axis=1)
    total = areas.sum()
    if total <= 0:
        raise ValueError("Mesh has no surface to sample")
    generator = np.random.default_rng(seed)
    chosen = generator.choice(len(faces), size=count, p=areas / total)
    first, second = generator.random((2, count, 1))
    beyond = (first + second > 1.0).ravel()
    first[beyond] = 1.0 - first[beyond]
    second[beyond] = 1.0 - second[beyond]
    origin = corners[chosen, 0]
    return (origin
            + first * (corners[chosen, 1] - origin)
            + second * (corners[chosen, 2] - origin)).astype(np.float64)


def local_densities(points, neighbours=NEIGHBOURS):
    from scipy.spatial import cKDTree

    neighbours = min(neighbours, len(points))
    distances, indices = cKDTree(points).query(points, k=neighbours)
    return np.asarray([max(distances[group, 1]) ** 2 for group in indices],
                      dtype=np.float64)


def seed_of(folder):
    return int.from_bytes(folder.name.encode("utf-8")[-8:], "little") % (2 ** 31)


def write_for(folder, count):
    mesh_path = folder / "mesh_watertight.ply"
    if not mesh_path.is_file():
        return "no watertight mesh"
    vertices, faces = read_ply_mesh(mesh_path)
    if len(faces) == 0:
        return "mesh has no faces"
    points = sample_surface(vertices, faces, count, seed_of(folder))
    points.tofile(folder / "gt_3dpoints.mgn")
    local_densities(points).tofile(folder / "densities.mgn")
    return None


def main() -> None:
    args = parse_args()
    root = args.experiment_root.resolve()
    folders = sorted(p.parent for p in (root / "objects").glob("*/*/mesh_watertight.ply"))
    if args.shard_count > 1:
        folders = folders[args.shard_index::args.shard_count]

    written = 0
    failures = []
    for folder in folders:
        if (folder / "gt_3dpoints.mgn").is_file() and (folder / "densities.mgn").is_file():
            written += 1
            continue
        try:
            reason = write_for(folder, args.samples)
        except Exception as error:
            reason = "%s: %s" % (type(error).__name__, str(error)[:120])
        if reason is None:
            written += 1
        else:
            failures.append({"object": str(folder.relative_to(root)), "reason": reason})

    report = {"shard": [args.shard_index, args.shard_count], "objects": len(folders),
              "written": written, "failure_count": len(failures), "failures": failures[:20],
              "samples": args.samples}
    state = root / "state" / "mgn_points"
    state.mkdir(parents=True, exist_ok=True)
    (state / ("shard-%d.json" % args.shard_index)).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit("MGN point export failed for %d objects" % len(failures))


if __name__ == "__main__":
    main()
