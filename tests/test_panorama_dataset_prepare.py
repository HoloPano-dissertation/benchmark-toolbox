import json
from pathlib import Path
import subprocess
import sys

import pytest

DATASET = Path(__file__).resolve().parents[1] / "tools/3dfront_dataset"


@pytest.fixture(autouse=True)
def dataset_imports(monkeypatch):
    monkeypatch.syspath_prepend(str(DATASET))


def test_frozen_split_is_house_disjoint_and_excludes_approved_rooms():
    from prepare import frozen_rooms
    rows = frozen_rooms()
    assert {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")} == {
        "train": 663, "val": 148, "test": 141}
    houses = {s: {r["house_id"] for r in rows if r["split"] == s} for s in ("train", "val", "test")}
    assert not houses["train"] & houses["test"]
    assert not houses["train"] & houses["val"]
    assert not houses["test"] & houses["val"]
    excluded = json.loads((DATASET / "splits/excluded_rooms.json").read_text())["rooms"]
    assert len(excluded) == 45
    assert not {r["room_id"] for r in rows} & {r["room_id"] for r in excluded}


def test_initialization_does_not_create_partial_experiment_with_missing_source(tmp_path):
    from prepare import initialize
    root = tmp_path / "new-experiment"
    with pytest.raises((ValueError, FileNotFoundError, NotADirectoryError)):
        initialize(tmp_path / "missing-source", root)
    assert not root.exists()


def test_initialization_refuses_existing_directory(tmp_path):
    from prepare import initialize
    with pytest.raises(FileExistsError):
        initialize(tmp_path / "source", tmp_path)


@pytest.fixture
def exported_dataset(tmp_path):
    from PIL import Image
    from _lib.classes import DEFAULT_CLASSES as CLASSES

    def write(path, data, jsonl=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r)+"\n" for r in data) if jsonl else json.dumps(data))

    rooms, mapping = [], []
    for split in ("train", "val", "test"):
        room_id = split+"-house/room"
        rooms.append({"room_id": room_id, "house_id": split+"-house", "split": split})
        records, images, annotations, dpc_paths = [], [], [], []
        for view in range(4):
            sample_id = room_id+"/"+str(view)
            rgb = tmp_path / "rgb" / split / (str(view)+".png")
            rgb.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (1024, 512), "gray").save(rgb)
            gt = tmp_path / "gt" / split / (str(view)+".json")
            write(gt, {"layout": {"min_corner": [-1, -1, -1], "max_corner": [1, 1, 1]}, "objects": []})
            records.append({"sample_id": sample_id, "input": str(rgb), "ground_truth": str(gt),
                            "metadata": {"split": split}})
            images.append({"sample_id": sample_id, "id": view+1, "file_name": str(rgb), "width": 1024, "height": 512})
            annotations.append({"id": view+1, "image_id": view+1, "category_id": 1, "bbox": [0, 0, 1, 1],
                                "area": 1, "segmentation": {"size": [512, 1024], "counts": [0, 1, 512*1024-1]}})
            name = sample_id.replace("/", "__")
            link = tmp_path / "dpc_dataset/images" / split / (name+".png")
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(rgb)
            dpc_paths.append(link.relative_to(tmp_path / "dpc_dataset").as_posix())
            mapping.append({"dpc_name": name, "sample_id": sample_id, "split": split, "input": str(rgb), "ground_truth": str(gt)})
        for folder in ("manifests", "manifests_gt"):
            write(tmp_path / folder / (split+".jsonl"), records, True)
        write(tmp_path / "coco" / (split+".json"), {"images": images, "annotations": annotations,
              "categories": [{"id": i+1, "name": name, "supercategory": "furniture"} for i, name in enumerate(CLASSES)]})
        write(tmp_path / "dpc_dataset" / (split+".json"), dpc_paths)
        (tmp_path / "splits").mkdir(exist_ok=True)
        (tmp_path / "splits" / (split+".txt")).write_text(room_id+"\n")
    write(tmp_path / "splits/rooms.jsonl", rooms, True)
    write(tmp_path / "splits/excluded_rooms.jsonl", [], True)
    write(tmp_path / "dpc_dataset/sample_map.jsonl", mapping, True)
    return tmp_path


def test_all_exports_are_checked_without_approving_training(exported_dataset):
    from _lib.validate_dataset import validate_dataset
    report = validate_dataset(exported_dataset)
    assert report["ready"] and not report["training_approved"]
    assert report["sample_counts"] == {"train": 4, "val": 4, "test": 4}
    assert report["coco_annotations"] == report["sample_counts"]


def test_excluded_room_leak_invalidates_previous_success(exported_dataset):
    from _lib.validate_dataset import validate_dataset
    validate_dataset(exported_dataset)
    (exported_dataset / "splits/excluded_rooms.jsonl").write_text('{"room_id":"train-house/room"}\n')
    with pytest.raises(ValueError, match="Excluded"):
        validate_dataset(exported_dataset)
    assert not json.loads((exported_dataset / "state/derived_validation.json").read_text())["ready"]


@pytest.mark.parametrize("mutation", ["rle", "category", "duplicate", "bbox"])
def test_coco_corruption_is_rejected(exported_dataset, mutation):
    from _lib.validate_dataset import validate_dataset
    path = exported_dataset / "coco/train.json"
    coco = json.loads(path.read_text())
    if mutation == "rle":
        coco["annotations"][0]["segmentation"]["counts"][-1] -= 1
    elif mutation == "category":
        coco["annotations"][0]["category_id"] = 99
    elif mutation == "duplicate":
        coco["images"].append(coco["images"][0])
    else:
        coco["annotations"][0]["bbox"][0] = -1
    path.write_text(json.dumps(coco))
    with pytest.raises(ValueError):
        validate_dataset(exported_dataset)


def test_wrong_dpc_link_is_rejected(exported_dataset):
    from _lib.validate_dataset import validate_dataset
    link = next((exported_dataset / "dpc_dataset/images/train").glob("*.png"))
    link.unlink()
    link.symlink_to(exported_dataset / "rgb/test/0.png")
    with pytest.raises(ValueError, match="DPC source mismatch"):
        validate_dataset(exported_dataset)


def test_all_flagged_review_includes_each_failed_geometry_room(tmp_path):
    from PIL import Image
    report, output = tmp_path / "audit", tmp_path / "review"
    (report / "previews").mkdir(parents=True)
    rooms = []
    for index in range(20):
        preview = "previews/room-%02d.jpg" % index
        Image.new("RGB", (16, 16), "gray").save(report / preview)
        rooms.append({"room_id": "house/room-%02d" % index, "split": "train", "preview": preview,
                      "geometry_pass": False, "flags": ["inside_furniture_bbox"]})
    (report / "rooms.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rooms))
    subprocess.run([sys.executable, str(DATASET / "review.py"), str(report), str(output), "--all-flagged"], check=True)
    selected = json.loads((output / "selection.json").read_text())
    assert {r["room_id"] for r in selected} == {r["room_id"] for r in rooms}
