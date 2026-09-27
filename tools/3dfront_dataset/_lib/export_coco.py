#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from classes import experiment_classes
from panorama_seam import annotation_masks, to_frame, tight_box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--merge", action="store_true",
                        help="Join the shard reports into one file per split")
    return parser.parse_args()


def rolled_png(input_path: Path, frame: int) -> Path:
    target = input_path.with_suffix(f".frame{frame}.png")
    if not target.exists():
        with Image.open(input_path) as image:
            rolled = to_frame(np.asarray(image), frame)
        Image.fromarray(rolled).save(target)
    return target


def decode_json_dataset(value) -> object:
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value))


def uncompressed_rle(mask: np.ndarray) -> dict:
    pixels = np.asarray(mask, dtype=np.uint8).ravel(order="F")
    starts = np.flatnonzero(
        np.concatenate(([True], pixels[1:] != pixels[:-1]))
    )
    lengths = np.diff(np.concatenate((starts, [len(pixels)]))).tolist()
    if pixels[0]:
        lengths.insert(0, 0)
    return {"size": list(mask.shape), "counts": lengths}


def inside_dataset(path, prefixes):
    text = str(path)
    for prefix in prefixes:
        if text.startswith(prefix):
            return text[len(prefix):].lstrip("/")
    raise ValueError(
        "%s lies outside the dataset root, so the annotation could not name it by a "
        "path another machine can follow" % text)


def dataset_prefixes(root, given):
    return tuple(sorted({str(Path(p)) + "/" for p in (root, Path(given).resolve(), given)},
                        key=len, reverse=True))


def shard_path(root, split, index):
    return root / "coco" / "_shards" / ("%s.%d.json" % (split, index))


def view_annotations(records, category_id, prefixes):
    images = []
    annotations = []
    visible_counts: Counter[str] = Counter()
    moved_to_rolled = 0
    split_instances = 0
    for record in records:
        input_path = Path(record["input"])
        hdf5_path = Path(record["metadata"]["hdf5"])
        with h5py.File(hdf5_path, "r") as source:
            instances = np.asarray(source["instance_segmaps"][()])
            attributes = decode_json_dataset(source["instance_attribute_maps"][()])
        height, width = (int(value) for value in instances.shape)
        by_instance = {int(item["idx"]): item for item in attributes}
        scene = json.loads(Path(record["ground_truth"]).read_text(encoding="utf-8"))
        label_by_source = {
            Path(entry["attributes"]["source_glb"]).name: entry["label"]
            for entry in scene["objects"]}

        per_frame: dict[int, list[tuple[str, np.ndarray, dict]]] = {}
        for instance_id in np.unique(instances):
            instance_id = int(instance_id)
            item = by_instance.get(instance_id)
            if instance_id == 0 or item is None:
                continue
            source = str(item.get("source_file") or "")
            class_name = label_by_source.get(Path(source).name) if source else None
            if class_name is None or class_name not in category_id:
                continue
            mask = instances == instance_id
            if not mask.any():
                continue
            pieces = annotation_masks(mask)
            if len(pieces) == 1 and pieces[0][0] == 1:
                moved_to_rolled += 1
            elif len(pieces) > 1:
                split_instances += 1
            for frame, piece in pieces:
                per_frame.setdefault(frame, []).append((class_name, piece, item))

        frame_image_id = {}
        per_frame.setdefault(0, [])
        for frame in sorted(per_frame):
            frame_image_id[frame] = len(images) + 1
            file_name = input_path if frame == 0 else rolled_png(input_path, frame)
            images.append({
                "id": frame_image_id[frame],
                "file_name": inside_dataset(file_name, prefixes),
                "width": width,
                "height": height,
                "sample_id": record["sample_id"],
                "panorama_frame": frame,
            })
        for frame, entries in per_frame.items():
            for class_name, piece, item in entries:
                box = tight_box(piece)
                if box is None:
                    continue
                annotations.append({
                    "id": len(annotations) + 1,
                    "image_id": frame_image_id[frame],
                    "category_id": category_id[class_name],
                    "bbox": box,
                    "area": int(piece.sum()),
                    "segmentation": uncompressed_rle(piece),
                    "iscrowd": 0,
                    "source_file": item.get("source_file"),
                })
                visible_counts[class_name] += 1
    return {"images": images, "annotations": annotations,
            "visible_categories": dict(sorted(visible_counts.items())),
            "instances_moved_to_rolled_frame": moved_to_rolled,
            "instances_split_across_both_seams": split_instances}


def joined(parts):
    images = []
    annotations = []
    visible_counts: Counter[str] = Counter()
    moved_to_rolled = 0
    split_instances = 0
    for part in parts:
        renumbered = {}
        for image in part["images"]:
            renumbered[image["id"]] = len(images) + 1
            images.append(dict(image, id=renumbered[image["id"]]))
        for annotation in part["annotations"]:
            annotations.append(dict(annotation, id=len(annotations) + 1,
                                    image_id=renumbered[annotation["image_id"]]))
        visible_counts.update(part["visible_categories"])
        moved_to_rolled += part["instances_moved_to_rolled_frame"]
        split_instances += part["instances_split_across_both_seams"]
    return {"images": images, "annotations": annotations,
            "visible_categories": dict(sorted(visible_counts.items())),
            "instances_moved_to_rolled_frame": moved_to_rolled,
            "instances_split_across_both_seams": split_instances}


def write_split(output_root, split, classes, category_id, payload, panoramas):
    document = {
        "info": {
            "description": "MIDI-3D/3D-FRONT synthetic panoramas",
            "projection": "equirectangular",
            "split": split,
            "file_name": "relative to the dataset root, the directory holding rgb/",
            "panorama_frame": "0 is the panorama itself; 1 is the same panorama turned "
                              "by half its width, where an object crossing the seam is "
                              "one box instead of two",
        },
        "licenses": [],
        "categories": [
            {"id": category_id[name], "name": name, "supercategory": "furniture"}
            for name in classes
        ],
        "images": payload["images"],
        "annotations": payload["annotations"],
    }
    (output_root / f"{split}.json").write_text(
        json.dumps(document, separators=(",", ":")) + "\n", encoding="utf-8")
    images = payload["images"]
    return {
        "panoramas": panoramas,
        "images": len(images),
        "visible_annotations": len(payload["annotations"]),
        "visible_categories": payload["visible_categories"],
        "instances_moved_to_rolled_frame": payload["instances_moved_to_rolled_frame"],
        "instances_split_across_both_seams": payload["instances_split_across_both_seams"],
        "frame_wide_annotations": sum(
            1 for a in payload["annotations"]
            if a["bbox"][2] > 0.9 * images[0]["width"]) if images else 0,
    }


def read_records(manifest_root, split):
    return [json.loads(line)
            for line in (manifest_root / f"{split}.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    root = args.experiment_root.resolve()
    manifest_root = root / "manifests_gt"
    output_root = root / "coco"
    output_root.mkdir(parents=True, exist_ok=True)
    classes = experiment_classes(root)
    prefixes = dataset_prefixes(root, args.experiment_root)
    category_id = {name: index + 1 for index, name in enumerate(classes)}
    status = {"ready": True, "splits": {}, "classes": classes}

    for split in ("train", "val", "test"):
        records = read_records(manifest_root, split)
        if args.merge:
            parts = []
            for index in range(args.shard_count):
                path = shard_path(root, split, index)
                if not path.is_file():
                    raise FileNotFoundError(
                        "Shard %d of %s left no report; its annotations are missing "
                        "from the split: %s" % (index, split, path))
                parts.append(json.loads(path.read_text(encoding="utf-8")))
            status["splits"][split] = write_split(
                output_root, split, classes, category_id, joined(parts), len(records))
            continue
        if args.shard_count > 1:
            shard = view_annotations(records[args.shard_index::args.shard_count],
                                     category_id, prefixes)
            path = shard_path(root, split, args.shard_index)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(shard, separators=(",", ":")) + "\n",
                            encoding="utf-8")
            continue
        status["splits"][split] = write_split(
            output_root, split, classes, category_id,
            view_annotations(records, category_id, prefixes), len(records))

    if args.shard_count > 1 and not args.merge:
        print(json.dumps({"shard": args.shard_index, "of": args.shard_count}))
        return
    (root / "state" / "coco.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
