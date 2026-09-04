import json
from pathlib import Path

import numpy as np
from PIL import Image

from .classes import experiment_classes
from .selection import index_rows


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def by_sample(rows, expected, name):
    indexed = {row["sample_id"]: row for row in rows}
    require(len(indexed) == len(rows) and set(indexed) == expected,
            name + ": duplicate, missing or unexpected samples")
    return indexed


def validate_annotation(annotation, image):
    width, height = image["width"], image["height"]
    x, y, w, h = annotation["bbox"]
    require(np.isfinite([x, y, w, h]).all() and x >= 0 and y >= 0 and w > 0 and h > 0
            and x+w <= width and y+h <= height, "Invalid COCO bbox")
    rle = annotation["segmentation"]
    counts = rle["counts"]
    require(rle["size"] == [height, width] and isinstance(counts, list) and len(counts) > 0,
            "Invalid COCO RLE shape")
    require(all(type(n) is int and n >= 0 for n in counts)
            and sum(counts) == width*height, "Invalid COCO RLE runs")
    require(sum(counts[1::2]) == annotation["area"] and 0 < annotation["area"] <= w*h,
            "COCO RLE area mismatch")


def validate_dataset(root):
    root = Path(root)
    report_path = root / "state/derived_validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text('{"ready":false,"training_approved":false}\n')
    rooms = read_rows(root / "splits/rooms.jsonl")
    index_rows(rooms)
    excluded = read_rows(root / "splits/excluded_rooms.jsonl")
    require(not {r["room_id"] for r in rooms} & {r["room_id"] for r in excluded},
            "Excluded rooms leaked into the dataset")
    mapping = read_rows(root / "dpc_dataset/sample_map.jsonl")
    require(len({r["dpc_name"] for r in mapping}) == len(mapping), "Duplicate DPC names")
    require(all(r["split"] in ("train", "val", "test") for r in mapping), "Invalid DPC split")
    counts, annotation_counts = {}, {}
    classes = experiment_classes(root)
    for split in ("train", "val", "test"):
        room_ids = {r["room_id"] for r in rooms if r["split"] == split}
        split_lines = (root / "splits" / (split+".txt")).read_text().splitlines()
        require(set(split_lines) == room_ids and len(split_lines) == len(room_ids), "Split text mismatch")
        expected = {room+"/"+str(view) for room in room_ids for view in range(4)}
        rgb = by_sample(read_rows(root / "manifests" / (split+".jsonl")), expected, "RGB")
        gt = by_sample(read_rows(root / "manifests_gt" / (split+".jsonl")), expected, "GT")
        dimensions = {}
        for sample_id, record in rgb.items():
            require(record["metadata"]["split"] == split, "RGB split mismatch")
            require(Path(record["input"]).resolve() == Path(gt[sample_id]["input"]).resolve(), "RGB/GT path mismatch")
            with Image.open(record["input"]) as image:
                dimensions[sample_id] = image.size
                require(image.size == (1024, 512), "Expected 1024x512 panorama")
                image.verify()
            scene = json.loads(Path(gt[sample_id]["ground_truth"]).read_text())
            lo, hi = (np.asarray(scene["layout"][key]) for key in ("min_corner", "max_corner"))
            require(lo.shape == hi.shape == (3,) and np.isfinite([lo, hi]).all()
                    and (hi > lo).all() and (lo <= 0).all() and (hi >= 0).all(), "Invalid layout bounds")
            for obj in scene["objects"]:
                box = obj["bbox"]
                center, size, basis = (np.asarray(box[key]) for key in ("center", "size", "basis"))
                require(obj["label"] in classes and center.shape == size.shape == (3,)
                        and np.isfinite([center, size]).all() and (size > 0).all(), "Invalid 3D box")
                require(basis.shape == (3, 3) and np.isfinite(basis).all()
                        and np.allclose(basis @ basis.T, np.eye(3), atol=1e-5), "Invalid 3D box basis")
        coco = json.loads((root / "coco" / (split+".json")).read_text())
        require(coco["categories"] == [{"id": i+1, "name": name, "supercategory": "furniture"}
                                        for i, name in enumerate(classes)], "COCO class order mismatch")
        images = {r["id"]: r for r in coco["images"]}
        require(len(images) == len(coco["images"]), "Duplicate COCO image IDs")
        by_frame = {(r["sample_id"], r.get("panorama_frame", 0)) for r in coco["images"]}
        require(len(by_frame) == len(coco["images"]), "Duplicate COCO sample/frame pairs")
        require({sample for sample, frame in by_frame if frame == 0} == expected,
                "COCO: duplicate, missing or unexpected samples")
        require(all(sample in expected for sample, _ in by_frame),
                "COCO: unexpected samples")
        for image in images.values():
            sample_id = image["sample_id"]
            rendered = Path(rgb[sample_id]["input"])
            frame = image.get("panorama_frame", 0)
            expected_file = rendered if frame == 0 else rendered.with_suffix(
                ".frame%d.png" % frame)
            require((image["width"], image["height"]) == dimensions[sample_id]
                    and Path(image["file_name"]).resolve() == expected_file.resolve()
                    and expected_file.is_file(),
                    "COCO image mismatch")
        annotations = coco["annotations"]
        require(len({a["id"] for a in annotations}) == len(annotations), "Duplicate COCO annotation IDs")
        for annotation in annotations:
            require(annotation["image_id"] in images and annotation["category_id"] in range(1, len(classes)+1),
                    "Unknown COCO image/category")
            validate_annotation(annotation, images[annotation["image_id"]])
        mapped = by_sample([r for r in mapping if r["split"] == split], expected, "DPC")
        paths = json.loads((root / "dpc_dataset" / (split+".json")).read_text())
        expected_paths = {"images/"+split+"/"+r["dpc_name"]+".png" for r in mapped.values()}
        require(len(paths) == len(expected_paths) and set(paths) == expected_paths, "DPC split mismatch")
        for sample_id, record in mapped.items():
            link = root / "dpc_dataset/images" / split / (record["dpc_name"]+".png")
            require(link.is_file() and link.resolve() == Path(rgb[sample_id]["input"]).resolve()
                    and Path(record["input"]).resolve() == link.resolve()
                    and Path(record["ground_truth"]).resolve() == Path(gt[sample_id]["ground_truth"]).resolve(),
                    "DPC source mismatch")
        counts[split], annotation_counts[split] = len(expected), len(annotations)
    report = {"ready": True, "training_approved": False, "sample_counts": counts,
              "coco_annotations": annotation_counts,
              "scope": "Structural consistency only; not a visual or metric-scale certification"}
    report_path.write_text(json.dumps(report, indent=2)+"\n")
    return report
