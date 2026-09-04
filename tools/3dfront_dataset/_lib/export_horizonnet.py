#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np

from export_ground_truth import room_geometry
from layout_targets import polygon_targets, native_polygon_corners


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-legacy-cuboid", action="store_true")
    return parser.parse_args()


def corner_labels(
    layout_min: np.ndarray,
    layout_max: np.ndarray,
    camera: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    if not np.logical_and(camera > layout_min, camera < layout_max).all():
        raise ValueError("Camera must be strictly inside the structural layout cuboid")
    xy = np.array(
        [
            [layout_min[0], layout_min[1]],
            [layout_max[0], layout_min[1]],
            [layout_max[0], layout_max[1]],
            [layout_min[0], layout_max[1]],
        ],
        dtype=np.float64,
    )
    projected = []
    for x, y in xy:
        dx, dy = x - camera[0], y - camera[1]
        radius = math.hypot(dx, dy)
        if radius <= 1e-8:
            raise ValueError("Camera lies on a layout corner")
        longitude = math.atan2(dx, dy)
        u = (0.5 + longitude / (2.0 * math.pi)) % 1.0
        ceiling_latitude = math.atan2(layout_max[2] - camera[2], radius)
        floor_latitude = math.atan2(layout_min[2] - camera[2], radius)
        ceiling_v = 0.5 - ceiling_latitude / math.pi
        floor_v = 0.5 - floor_latitude / math.pi
        projected.append((u * width - 0.5, ceiling_v * height - 0.5,
                          floor_v * height - 0.5))

    projected.sort(key=lambda item: item[0])
    corners = []
    for x, ceiling_y, floor_y in projected:
        corners.extend(((x, ceiling_y), (x, floor_y)))
    result = np.asarray(corners, dtype=np.float64)
    if not np.all(result[0::2, 1] < result[1::2, 1]):
        raise ValueError("Ceiling must project above floor")
    return result


def ensure_symlink(source: Path, destination: Path) -> None:
    if destination.is_symlink() and destination.resolve() == source.resolve():
        return
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    os.symlink(source, destination)


def prune_stale(root, split, keep):
    for folder in ("img", "label_cor", "label_dense"):
        directory = root / "horizonnet" / split / folder
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if path.stem not in keep:
                path.unlink()


def main() -> None:
    args = parse_args()
    experiment_root = args.experiment_root.resolve()
    output_root = (args.output or experiment_root / "horizonnet").resolve()
    scene_root = (experiment_root / "source" / "3D-FRONT-TEST-SCENE").resolve()
    room_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    counts: Counter[str] = Counter()
    rooms_seen = set()

    for split in ("train", "val", "test"):
        image_dir = output_root / split / "img"
        label_dir = output_root / split / "label_cor"
        dense_dir = output_root / split / "label_dense"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        dense_dir.mkdir(parents=True, exist_ok=True)
        manifest = experiment_root / "manifests" / f"{split}.jsonl"
        records = [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for record in records:
            metadata = record["metadata"]
            room_id = str(metadata["room_id"])
            rooms_seen.add(room_id)
            view_index = int(metadata["view_index"])
            render_dir = experiment_root / "outputs" / split / room_id
            render_path = render_dir / "render.json"
            if not render_path.is_file():
                render_path = render_dir / "pilot.json"
            render_metadata = json.loads(render_path.read_text())
            layout = render_metadata.get("layout")
            if layout is None and not args.allow_legacy_cuboid:
                raise ValueError("Renderer metadata has no recovered polygon; legacy cuboid labels are disabled")
            if layout is None and room_id not in room_cache:
                layout_min, layout_max, _ = room_geometry(
                    scene_root / room_id, scene_root
                )
                room_cache[room_id] = (layout_min, layout_max)
            width, height = (int(v) for v in render_metadata["resolution"])
            camera = np.asarray(
                render_metadata["camera_locations"][view_index], dtype=np.float64
            )
            if layout is not None:
                targets = polygon_targets(layout, camera, width, height)
                try:
                    corners = native_polygon_corners(layout, camera, width, height)
                except ValueError:
                    corners = None
            else:
                layout_min, layout_max = room_cache[room_id]
                corners = corner_labels(layout_min, layout_max, camera, width, height)
                targets = None
            sample_name = str(record["sample_id"]).replace("/", "__")
            image_source = Path(record["input"]).resolve()
            image_target = image_dir / f"{sample_name}{image_source.suffix.lower()}"
            label_target = label_dir / f"{sample_name}.txt"
            ensure_symlink(image_source, image_target)
            if targets is not None:
                np.savez_compressed(dense_dir / f"{sample_name}.npz", **targets)
            if corners is not None:
                label_target.write_text(
                "".join(f"{x:.6f} {y:.6f}\n" for x, y in corners),
                encoding="utf-8",
                )
            counts[split] += 1
        prune_stale(experiment_root, split,
                    {str(r["sample_id"]).replace("/", "__") for r in records})

    status = {
        "ready": True,
        "room_count": len(rooms_seen),
        "sample_counts": dict(counts),
        "layout_model": "recovered floor-contour extrusion with inferred structural floor/ceiling heights",
        "label_format": "dense boundary/corner targets; native text additionally for hole-free contours",
        "legacy_cuboid_allowed": args.allow_legacy_cuboid,
        "projection": "equirectangular 1024x512",
        "corner_order": "native labels preserve polygon traversal, not azimuth sorting",
    }
    state_dir = experiment_root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "horizonnet.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
