#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from panorama_seam import annotation_masks, to_frame, tight_box

MARGIN = 0.1
MINIMUM_SIDE = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--margin", type=float, default=MARGIN)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--merge", action="store_true",
                        help="Join the shard reports into the object catalogue")
    return parser.parse_args()


def decode(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value))


def padded_box(box, width, height, margin):
    x, y, box_width, box_height = box
    pad_x = int(round(box_width * margin))
    pad_y = int(round(box_height * margin))
    left = max(0, x - pad_x)
    top = max(0, y - pad_y)
    right = min(width, x + box_width + pad_x)
    bottom = min(height, y + box_height + pad_y)
    return left, top, right, bottom


def read_records(root, split):
    return [json.loads(line)
            for line in (root / "manifests_gt" / f"{split}.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]


def crops_of(records, margin):
    stems = []
    skipped = Counter()
    for record in records:
        scene = json.loads(Path(record["ground_truth"]).read_text(encoding="utf-8"))
        with h5py.File(record["metadata"]["hdf5"], "r") as source:
            instances = np.asarray(source["instance_segmaps"][()])
            attributes = decode(source["instance_attribute_maps"][()])
        with Image.open(record["input"]) as handle:
            rendered = np.asarray(handle.convert("RGB"))
        height, width = instances.shape
        by_source = {Path(str(item.get("source_file") or "")).name: int(item["idx"])
                     for item in attributes if item.get("source_file")}
        frames = {0: rendered, 1: to_frame(rendered, 1)}
        sample = record["sample_id"].replace("/", "__")
        for item in scene["objects"]:
            shape = item["attributes"].get("shape")
            if shape is None:
                continue
            instance = by_source.get(Path(item["attributes"]["source_glb"]).name)
            if instance is None:
                skipped["missing_instance"] += 1
                continue
            mask = instances == instance
            if not mask.any():
                skipped["not_visible"] += 1
                continue
            frame, piece = max(annotation_masks(mask),
                               key=lambda option: int(option[1].sum()))
            box = tight_box(piece)
            if box is None or min(box[2], box[3]) < MINIMUM_SIDE:
                skipped["too_small"] += 1
                continue
            left, top, right, bottom = padded_box(box, width, height, margin)
            folder = Path(shape).parent
            name = "crop-%s" % sample
            Image.fromarray(frames[frame][top:bottom, left:right]).save(
                folder / (name + ".png"))
            stems.append("%s/%s/%s" % (folder.parent.name, folder.name, name))
    return stems, skipped


def shard_path(root, split, index):
    return root / "objects" / "_shards" / ("%s.%d.json" % (split, index))


def gather(root, shard_count):
    splits = {}
    skipped = Counter()
    for split in ("train", "val", "test"):
        stems = []
        for index in range(shard_count):
            path = shard_path(root, split, index)
            if not path.is_file():
                raise FileNotFoundError(
                    "Shard %d of %s left no report; its crops are missing from the "
                    "catalogue: %s" % (index, split, path))
            report = json.loads(path.read_text(encoding="utf-8"))
            stems.extend(report["stems"])
            skipped.update(report["skipped"])
        splits[split] = stems
    return splits, skipped


def write_catalogue(root, splits, skipped, margin):
    catalogue = root / "objects"
    for split, stems in splits.items():
        (catalogue / f"{split}.json").write_text(
            json.dumps(sorted(stems), indent=2) + "\n", encoding="utf-8")
    status = {
        "ready": True,
        "crops": {split: len(stems) for split, stems in splits.items() if stems},
        "skipped": dict(skipped),
        "margin": margin,
        "split_files": sorted(f"{name}.json" for name in splits),
        "note": "point data.split of a training config at one of these files; the DPC "
                "loader maps its own val mode onto test.json otherwise",
    }
    (root / "state" / "crops.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))


def main() -> None:
    args = parse_args()
    root = args.experiment_root.resolve()

    if args.merge:
        splits, skipped = gather(root, args.shard_count)
        write_catalogue(root, splits, skipped, args.margin)
        return

    splits = {}
    missed_by_split = {}
    for split in ("train", "val", "test"):
        records = read_records(root, split)
        if args.shard_count > 1:
            records = records[args.shard_index::args.shard_count]
        splits[split], missed_by_split[split] = crops_of(records, args.margin)

    if args.shard_count > 1:
        for split, stems in splits.items():
            path = shard_path(root, split, args.shard_index)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"stems": stems, "skipped": dict(missed_by_split[split])}) + "\n",
                encoding="utf-8")
        print(json.dumps({"shard": args.shard_index, "of": args.shard_count,
                          "crops": {s: len(v) for s, v in splits.items()}}))
        return

    skipped = Counter()
    for missed in missed_by_split.values():
        skipped.update(missed)
    write_catalogue(root, splits, skipped, args.margin)


if __name__ == "__main__":
    main()
