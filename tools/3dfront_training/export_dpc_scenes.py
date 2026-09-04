#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from utils.igibson_utils import IGScene
from utils.relation_utils import RelationOptimization
from utils.render_utils import seg2obj, is_obj_valid

WIDTH, HEIGHT = 1024, 512
EXPAND_DISTANCE = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--expand-distance", type=float, default=EXPAND_DISTANCE)
    return parser.parse_args()


def decode(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value))


def manhattan_world(layout):
    ring = np.asarray(layout["polygon"]["coordinates"][0], dtype=float)[:-1, :2]
    floor = np.column_stack((ring, np.full(len(ring), float(layout["floor_z"]))))
    ceiling = np.column_stack((ring, np.full(len(ring), float(layout["ceiling_z"]))))
    return np.concatenate((floor, ceiling), axis=0)


def camera_dict(position):
    position = np.asarray(position, dtype=float)
    return {"pos": position, "target": position + np.array([0.0, 1.0, 0.0]),
            "up": np.array([0.0, 0.0, 1.0]), "width": WIDTH, "height": HEIGHT}


def experiment_classes(root):
    path = root / "state" / "classes.json"
    return list(json.loads(path.read_text(encoding="utf-8"))["classes"])


def object_entry(scene_object, instance_id, segmentation, camera, classes):
    geometry = seg2obj(segmentation, instance_id, camera)
    if geometry is None:
        return None
    box = scene_object["bbox"]
    entry = {
        "classname": scene_object["label"],
        "label": classes.index(scene_object["label"]),
        "model_path": scene_object["attributes"]["shape"],
        "is_fixed": False,
        "bdb3d": {
            "centroid": np.asarray(box["center"], dtype=float)
            + np.asarray(camera["pos"], dtype=float),
            "basis": np.asarray(box["basis"], dtype=float),
            "size": np.asarray(box["size"], dtype=float),
        },
    }
    entry.update(geometry)
    entry["bdb2d_clip"] = dict(entry["bdb2d"])
    entry["contour_clip"] = dict(entry["contour"])
    return entry


def build_scene(record, ground_truth, segmentation, attributes, horizon, classes):
    metadata = ground_truth["metadata"]
    house, room, view = record["sample_id"].split("/")
    camera = camera_dict(metadata["camera_location_world"])
    by_source = {}
    for item in attributes:
        source = str(item.get("source_file") or "")
        if source:
            by_source[Path(source).name] = int(item["idx"])

    objects = []
    for scene_object in ground_truth["objects"]:
        name = Path(scene_object["attributes"]["source_glb"]).name
        instance_id = by_source.get(name)
        if instance_id is None:
            continue
        entry = object_entry(scene_object, instance_id, segmentation, camera, classes)
        if entry is None or not is_obj_valid(entry):
            continue
        entry["id"] = len(objects) + 1
        objects.append(entry)

    return {
        "name": view,
        "scene": house,
        "room": room,
        "camera": camera,
        "layout": {
            "manhattan_world": manhattan_world(metadata["layout_geometry_world"]),
            "horizon": {"bon": horizon["boundary"].astype(np.float32),
                        "cor": horizon["corner"].astype(np.float32)},
        },
        "objs": objects,
    }


def main() -> None:
    args = parse_args()
    root = args.experiment_root.resolve()
    output = args.output_root.resolve()
    optimiser = RelationOptimization(expand_dis=args.expand_distance)
    classes = experiment_classes(root)
    written = Counter()
    dropped = Counter()
    failures = []

    for split in ("train", "val", "test"):
        records = [
            json.loads(line)
            for line in (root / "manifests_gt" / f"{split}.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        records = records[args.shard_index::args.shard_count]
        for record in records:
            try:
                ground_truth = json.loads(
                    Path(record["ground_truth"]).read_text(encoding="utf-8"))
                with h5py.File(record["metadata"]["hdf5"], "r") as source:
                    segmentation = np.asarray(source["instance_segmaps"][()])
                    attributes = decode(source["instance_attribute_maps"][()])
                name = record["sample_id"].replace("/", "__")
                horizon = np.load(root / "horizonnet" / split / "label_dense"
                                  / (name + ".npz"))
                data = build_scene(record, ground_truth, segmentation,
                                   attributes, horizon, classes)
                dropped[split] += len(ground_truth["objects"]) - len(data["objs"])
                scene = IGScene(data)
                scene.data["layout"]["manhattan_pix"] = scene.transform.world2campix(
                    data["layout"]["manhattan_world"])
                optimiser.generate_relation(scene)
                target = output / split / data["scene"] / data["room"] / data["name"]
                target.mkdir(parents=True, exist_ok=True)
                with (target / "data.pkl").open("wb") as handle:
                    pickle.dump(scene.data, handle)
                written[split] += 1
            except Exception as error:
                failures.append({"sample_id": record["sample_id"],
                                 "error": "%s: %s" % (type(error).__name__, error)})

    status = {
        "ready": not failures,
        "shard": [args.shard_index, args.shard_count],
        "scenes": dict(written),
        "objects_dropped_as_invisible": dict(dropped),
        "classes": classes,
        "expand_distance": args.expand_distance,
        "failures": failures[:10],
        "failure_count": len(failures),
    }
    state = output / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / ("shard-%03d.json" % args.shard_index)).write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))
    if failures:
        raise SystemExit("DPC scene export failed for %d samples" % len(failures))


if __name__ == "__main__":
    main()
