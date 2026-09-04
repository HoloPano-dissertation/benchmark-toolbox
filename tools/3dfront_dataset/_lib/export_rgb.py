#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--views", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_manifest = args.experiment_root / "splits" / "rooms.jsonl"
    room_records = [
        json.loads(line) for line in split_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifests: dict[str, list[dict[str, object]]] = defaultdict(list)
    complete_rooms = 0
    incomplete_rooms: list[str] = []
    duplicate_camera_rooms: list[str] = []
    out_of_bounds_camera_rooms: list[str] = []

    for room in room_records:
        split = str(room["split"])
        room_id = str(room["room_id"])
        house_id = str(room["house_id"])
        render_dir = args.experiment_root / "outputs" / split / room_id
        frame_paths = [render_dir / f"{view}.hdf5" for view in range(args.views)]
        if not (render_dir / ".complete").is_file() or not all(
            path.is_file() and path.stat().st_size > 0 for path in frame_paths
        ):
            incomplete_rooms.append(f"{split}/{room_id}")
            continue
        metadata_path = render_dir / "render.json"
        if not metadata_path.is_file():
            metadata_path = render_dir / "pilot.json"
        render_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        camera_locations = render_metadata.get("camera_locations", [])
        unique_locations = {
            tuple(round(float(value), 7) for value in location)
            for location in camera_locations
        }
        if len(camera_locations) != args.views or len(unique_locations) != args.views:
            duplicate_camera_rooms.append(f"{split}/{room_id}")
            continue
        locations = np.asarray(camera_locations, dtype=np.float64)
        bounds_min = np.asarray(render_metadata["bounds_min"], dtype=np.float64)
        bounds_max = np.asarray(render_metadata["bounds_max"], dtype=np.float64)
        if not np.logical_and(
            locations >= bounds_min - 1e-7, locations <= bounds_max + 1e-7
        ).all():
            out_of_bounds_camera_rooms.append(f"{split}/{room_id}")
            continue
        complete_rooms += 1
        rgb_dir = args.experiment_root / "rgb" / split / room_id
        rgb_dir.mkdir(parents=True, exist_ok=True)
        for view, hdf5_path in enumerate(frame_paths):
            rgb_path = rgb_dir / f"{view}.png"
            if not rgb_path.is_file():
                with h5py.File(hdf5_path, "r") as source:
                    Image.fromarray(source["colors"][...]).save(rgb_path)
            manifests[split].append(
                {
                    "sample_id": f"{room_id}/{view}",
                    "input": str(rgb_path),
                    "metadata": {
                        "dataset": "MIDI-3D-FRONT-1K-custom",
                        "projection": "equirectangular",
                        "split": split,
                        "house_id": house_id,
                        "room_id": room_id,
                        "view_index": view,
                        "hdf5": str(hdf5_path),
                    },
                }
            )

    manifest_dir = args.experiment_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    sample_counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        records = sorted(manifests[split], key=lambda record: str(record["sample_id"]))
        sample_counts[split] = len(records)
        (manifest_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )

    status = {
        "expected_rooms": len(room_records),
        "complete_rooms": complete_rooms,
        "incomplete_room_count": len(incomplete_rooms),
        "incomplete_rooms": incomplete_rooms,
        "duplicate_camera_room_count": len(duplicate_camera_rooms),
        "duplicate_camera_rooms": duplicate_camera_rooms,
        "out_of_bounds_camera_room_count": len(out_of_bounds_camera_rooms),
        "out_of_bounds_camera_rooms": out_of_bounds_camera_rooms,
        "views_per_room": args.views,
        "sample_counts": sample_counts,
        "ready": (
            complete_rooms == len(room_records)
            and not duplicate_camera_rooms
            and not out_of_bounds_camera_rooms
        ),
    }
    (args.experiment_root / "state" / "finalize.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, indent=2))
    if not status["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
