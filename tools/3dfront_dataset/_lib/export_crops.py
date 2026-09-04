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


def main() -> None:
    args = parse_args()
    root = args.experiment_root.resolve()
    splits: dict[str, list[str]] = {}
    written = Counter()
    skipped = Counter()

    for split in ("train", "val", "test"):
        stems: list[str] = []
        records = [
            json.loads(line)
            for line in (root / "manifests_gt" / f"{split}.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()
        ]
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
                left, top, right, bottom = padded_box(box, width, height, args.margin)
                folder = Path(shape).parent
                name = "crop-%s" % sample
                Image.fromarray(frames[frame][top:bottom, left:right]).save(
                    folder / (name + ".png"))
                stems.append("%s/%s/%s" % (folder.parent.name, folder.name, name))
                written[split] += 1
        splits[split] = stems

    catalogue = root / "objects"
    for split, stems in splits.items():
        (catalogue / f"{split}.json").write_text(
            json.dumps(sorted(stems), indent=2) + "\n", encoding="utf-8")

    status = {
        "ready": True,
        "crops": dict(written),
        "skipped": dict(skipped),
        "margin": args.margin,
        "split_files": sorted(f"{name}.json" for name in splits),
        "note": "point data.split of a training config at one of these files; the DPC "
                "loader maps its own val mode onto test.json otherwise",
    }
    (root / "state" / "crops.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
