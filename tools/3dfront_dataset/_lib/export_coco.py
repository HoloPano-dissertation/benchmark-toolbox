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


def main() -> None:
    args = parse_args()
    root = args.experiment_root.resolve()
    manifest_root = root / "manifests_gt"
    output_root = root / "coco"
    output_root.mkdir(parents=True, exist_ok=True)
    classes = experiment_classes(root)
    category_id = {name: index + 1 for index, name in enumerate(classes)}
    status = {"ready": True, "splits": {}, "classes": classes}

    for split in ("train", "val", "test"):
        source_records = [
            json.loads(line)
            for line in (manifest_root / f"{split}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        images = []
        annotations = []
        visible_counts: Counter[str] = Counter()
        annotation_index = 1
        image_index = 0
        moved_to_rolled = 0
        split_instances = 0
        for record in source_records:
            input_path = Path(record["input"])
            metadata = record["metadata"]
            hdf5_path = Path(metadata["hdf5"])
            with h5py.File(hdf5_path, "r") as source:
                instances = np.asarray(source["instance_segmaps"][()])
                attributes = decode_json_dataset(source["instance_attribute_maps"][()])
            height, width = (int(value) for value in instances.shape)
            by_instance = {int(item["idx"]): item for item in attributes}
            scene = json.loads(
                Path(record["ground_truth"]).read_text(encoding="utf-8"))
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
                image_index += 1
                frame_image_id[frame] = image_index
                file_name = input_path if frame == 0 else rolled_png(input_path, frame)
                images.append(
                    {
                        "id": image_index,
                        "file_name": str(file_name),
                        "width": width,
                        "height": height,
                        "sample_id": record["sample_id"],
                        "panorama_frame": frame,
                    }
                )
            for frame, entries in per_frame.items():
                for class_name, piece, item in entries:
                    box = tight_box(piece)
                    if box is None:
                        continue
                    annotations.append(
                        {
                            "id": annotation_index,
                            "image_id": frame_image_id[frame],
                            "category_id": category_id[class_name],
                            "bbox": box,
                            "area": int(piece.sum()),
                            "segmentation": uncompressed_rle(piece),
                            "iscrowd": 0,
                            "source_file": item.get("source_file"),
                        }
                    )
                    annotation_index += 1
                    visible_counts[class_name] += 1
        payload = {
            "info": {
                "description": "MIDI-3D/3D-FRONT synthetic panoramas",
                "projection": "equirectangular",
                "split": split,
            },
            "licenses": [],
            "categories": [
                {"id": category_id[name], "name": name, "supercategory": "furniture"}
                for name in classes
            ],
            "images": images,
            "annotations": annotations,
        }
        (output_root / f"{split}.json").write_text(
            json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8"
        )
        status["splits"][split] = {
            "panoramas": len(source_records),
            "images": len(images),
            "visible_annotations": len(annotations),
            "visible_categories": dict(sorted(visible_counts.items())),
            "instances_moved_to_rolled_frame": moved_to_rolled,
            "instances_split_across_both_seams": split_instances,
            "frame_wide_annotations": sum(
                1 for a in annotations
                if a["bbox"][2] > 0.9 * images[0]["width"]) if images else 0,
        }
    (root / "state" / "coco.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
