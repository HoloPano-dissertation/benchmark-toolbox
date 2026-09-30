#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_root", type=Path)
    parser.add_argument("--classes", required=True,
                        help="Comma separated class list, in the order the model uses")
    parser.add_argument("--split", default="train")
    return parser.parse_args()


def average_sizes(sizes_by_class, classes):
    if not sizes_by_class:
        raise ValueError("No object sizes were found, so no average can be taken")
    averages = {name: np.mean(np.stack(sizes), axis=0)
                for name, sizes in sizes_by_class.items()}
    default = np.mean(np.stack(list(averages.values())), axis=0)
    return np.stack([averages.get(name, default.copy()) for name in classes])


def read_scene(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def collect(scene_root, split, transform_of):
    folders = json.loads((scene_root / (split + ".json")).read_text(encoding="utf-8"))
    sizes_by_class = defaultdict(list)
    distance_max = 0.0
    for folder in folders:
        scene = read_scene(scene_root / folder / "data.pkl")
        transform = transform_of(scene)
        for item in scene["objs"]:
            box = item["bdb3d"]
            sizes_by_class[item["classname"]].append(np.asarray(box["size"], dtype=float))
            distance = float(transform.world2campix(box)["dis"])
            distance_max = max(distance_max, distance)
    return sizes_by_class, distance_max


def main() -> None:
    args = parse_args()
    import sys
    sys.path.insert(0, str(Path.cwd()))
    from utils.igibson_utils import IGScene

    classes = [name for name in args.classes.split(",") if name]
    scene_root = args.scene_root.resolve()
    sizes_by_class, distance_max = collect(
        scene_root, args.split, lambda scene: IGScene(scene).transform)

    path = scene_root / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    unseen = sorted(set(classes) - set(sizes_by_class))
    metadata["size_avg"] = average_sizes(sizes_by_class, classes).tolist()
    metadata["dis_max"] = distance_max
    path.write_text(json.dumps(metadata, indent=1) + "\n", encoding="utf-8")

    print(json.dumps({"classes": len(classes), "scenes": len(json.loads(
        (scene_root / (args.split + ".json")).read_text(encoding="utf-8"))),
        "dis_max": distance_max, "classes_without_objects": unseen,
        "kept_keys": sorted(k for k in metadata if k.endswith("_bins"))}, indent=2,
        ensure_ascii=False))


if __name__ == "__main__":
    main()
