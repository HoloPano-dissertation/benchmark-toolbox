#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

MARKER = ".complete"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hdf5_root", type=Path)
    parser.add_argument("image_root", type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--format", choices=("png", "jpeg"), default="png")
    parser.add_argument("--quality", type=int, default=92, help="JPEG quality")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def finished_rooms(hdf5_root):
    return sorted(marker.parent for marker in hdf5_root.glob("*/*/" + MARKER))


def colour_of(path):
    with h5py.File(path, "r") as source:
        if "colors" not in source:
            raise KeyError("no colours in %s" % path)
        return np.asarray(source["colors"][:], dtype=np.uint8)


def export_room(room, target, suffix, quality):
    views = sorted(room.glob("*.hdf5"), key=lambda p: p.stem)
    if not views:
        raise FileNotFoundError("marked as finished but holds no views")
    written = 0
    for view in views:
        image = target / (view.stem + "." + suffix)
        if image.is_file() and image.stat().st_size:
            continue
        target.mkdir(parents=True, exist_ok=True)
        array = colour_of(view)
        staging = target / (view.stem + "." + suffix + ".part")
        if suffix == "jpeg":
            Image.fromarray(array).save(staging, "JPEG", quality=quality)
        else:
            Image.fromarray(array).save(staging, "PNG")
        staging.replace(image)
        written += 1
    return written


def main() -> None:
    args = parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("Invalid shard index/count")
    rooms = finished_rooms(args.hdf5_root)
    if not rooms:
        raise SystemExit("No finished rooms under %s" % args.hdf5_root)
    rooms = rooms[args.shard_index::args.shard_count]
    if args.limit:
        rooms = rooms[:args.limit]
    suffix = args.format
    written = failed = 0
    for room in rooms:
        target = args.image_root / room.parent.name / room.name
        try:
            written += export_room(room, target, suffix, args.quality)
        except Exception as error:
            failed += 1
            print("FAILED %s/%s: %s" % (room.parent.name, room.name, error), flush=True)
    print("rooms %d, images written %d, rooms failed %d" % (len(rooms), written, failed),
          flush=True)
    if failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
