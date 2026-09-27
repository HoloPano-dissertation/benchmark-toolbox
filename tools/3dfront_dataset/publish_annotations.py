#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

SPLITS = ("train", "val", "test")
PANORAMA_PREFIX = "panoramas/png"
DROPPED_ATTRIBUTES = ("shape",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--merge", action="store_true",
                        help="Join the per-view work into one COCO file and the README")
    return parser.parse_args()


def read_records(root, split):
    return [json.loads(line)
            for line in (root / "manifests_gt" / f"{split}.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]


def view_of(record):
    house, room, view = str(record["sample_id"]).split("/")
    return house, room, view


def published_scene(scene):
    published = {key: value for key, value in scene.items() if key != "objects"}
    published["objects"] = []
    for item in scene["objects"]:
        attributes = {key: value for key, value in item["attributes"].items()
                      if key not in DROPPED_ATTRIBUTES}
        published["objects"].append(dict(item, attributes=attributes))
    return published


def copy_views(root, target, records):
    scenes = target / "scenes"
    layout = target / "layout"
    written = 0
    for record in records:
        house, room, view = view_of(record)
        scene = json.loads(Path(record["ground_truth"]).read_text(encoding="utf-8"))
        folder = scenes / house / room
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{view}.json").write_text(
            json.dumps(published_scene(scene), ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8")

        split = str(record["metadata"]["split"])
        stem = f"{house}__{room}__{view}"
        folder = layout / house / room
        folder.mkdir(parents=True, exist_ok=True)
        for source, suffix in ((root / "horizonnet" / split / "label_dense" / (stem + ".npz"), ".npz"),
                               (root / "horizonnet" / split / "label_cor" / (stem + ".txt"), ".txt")):
            if source.is_file():
                shutil.copyfile(source, folder / (view + suffix))
        written += 1
    return written


def panorama_name(file_name):
    parts = Path(file_name).parts
    if parts[:1] != ("rgb",) or len(parts) < 5:
        raise ValueError("Unexpected image path in the annotation: %s" % file_name)
    return "/".join((PANORAMA_PREFIX,) + parts[2:])


def joined_coco(root):
    images = []
    annotations = []
    categories = None
    for split in SPLITS:
        payload = json.loads((root / "coco" / f"{split}.json").read_text(encoding="utf-8"))
        categories = categories or payload["categories"]
        renumbered = {}
        for image in payload["images"]:
            renumbered[image["id"]] = len(images) + 1
            images.append(dict(image, id=renumbered[image["id"]],
                               file_name=panorama_name(image["file_name"])))
        for annotation in payload["annotations"]:
            annotations.append(dict(annotation, id=len(annotations) + 1,
                                    image_id=renumbered[annotation["image_id"]]))
    return {
        "info": {
            "description": "3D-FRONT panoramas: two dimensional annotation",
            "projection": "equirectangular",
            "file_name": "relative to the dataset root, the directory holding panoramas/",
            "panorama_frame": "0 is the panorama itself; 1 is the same panorama turned "
                              "by half its width, where an object crossing the seam is "
                              "one box instead of two",
        },
        "licenses": [],
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }


def main() -> None:
    args = parse_args()
    root = args.experiment_root.resolve()
    target = args.target.resolve()
    target.mkdir(parents=True, exist_ok=True)

    if args.merge:
        payload = joined_coco(root)
        (target / "instances.json").write_text(
            json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
        print(json.dumps({"images": len(payload["images"]),
                          "annotations": len(payload["annotations"]),
                          "categories": len(payload["categories"])}, indent=2))
        return

    written = 0
    for split in SPLITS:
        records = read_records(root, split)
        if args.shard_count > 1:
            records = records[args.shard_index::args.shard_count]
        written += copy_views(root, target, records)
    print(json.dumps({"shard": [args.shard_index, args.shard_count], "views": written}))


if __name__ == "__main__":
    main()
