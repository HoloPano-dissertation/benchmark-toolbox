import json
from pathlib import Path
import subprocess
import sys

import pytest

np = pytest.importorskip("numpy")
h5py = pytest.importorskip("h5py")
Image = pytest.importorskip("PIL.Image")

DATASET = Path(__file__).resolve().parents[1] / "tools" / "3dfront_dataset"
WIDTH, HEIGHT = 128, 64
VIEWS = 3


def build(root):
    for split in ("train", "val", "test"):
        lines = []
        for view in range(VIEWS):
            rgb = root / "rgb" / split / f"{view}.png"
            rgb.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)).save(rgb)
            instances = np.zeros((HEIGHT, WIDTH), dtype=np.int64)
            instances[10 + view:30 + view, 40:60] = 1
            hdf5 = root / "outputs" / split / f"{view}.hdf5"
            hdf5.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(hdf5, "w") as handle:
                handle.create_dataset("instance_segmaps", data=instances)
                handle.create_dataset("instance_attribute_maps", data=json.dumps(
                    [{"idx": 1, "source_file": "Bed_a_0.glb"}]))
            gt = root / "gt" / split / f"{view}.json"
            gt.parent.mkdir(parents=True, exist_ok=True)
            gt.write_text(json.dumps({"objects": [
                {"label": "bed", "attributes": {"source_glb": "house/room/Bed_a_0.glb"}}]}))
            lines.append(json.dumps({
                "sample_id": f"house/room/{view}", "input": str(rgb),
                "ground_truth": str(gt), "metadata": {"hdf5": str(hdf5)}}))
        manifest = root / "manifests_gt" / f"{split}.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    state = root / "state"
    state.mkdir(exist_ok=True)
    (state / "classes.json").write_text(json.dumps({"classes": ["bed"]}))
    return root


def run(root, *extra):
    result = subprocess.run(
        [sys.executable, str(DATASET / "_lib" / "export_coco.py"), str(root), *extra],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result


@pytest.fixture
def experiment(tmp_path):
    return build(tmp_path)


def test_one_pass_numbers_images_and_annotations_from_one(experiment):
    run(experiment)
    payload = json.loads((experiment / "coco" / "train.json").read_text())
    assert [image["id"] for image in payload["images"]] == [1, 2, 3]
    assert [item["id"] for item in payload["annotations"]] == [1, 2, 3]
    assert {item["image_id"] for item in payload["annotations"]} == {1, 2, 3}


def test_shards_rebuild_exactly_what_one_pass_writes(experiment):
    run(experiment)
    whole = {split: json.loads((experiment / "coco" / f"{split}.json").read_text())
             for split in ("train", "val", "test")}
    status = json.loads((experiment / "state" / "coco.json").read_text())
    for split in whole:
        (experiment / "coco" / f"{split}.json").unlink()
    for index in range(2):
        run(experiment, "--shard-count", "2", "--shard-index", str(index))
    run(experiment, "--shard-count", "2", "--merge")
    for split, expected in whole.items():
        rebuilt = json.loads((experiment / "coco" / f"{split}.json").read_text())
        assert len(rebuilt["images"]) == len(expected["images"])
        assert [image["id"] for image in rebuilt["images"]] == \
            list(range(1, len(rebuilt["images"]) + 1))
        assert [item["id"] for item in rebuilt["annotations"]] == \
            list(range(1, len(rebuilt["annotations"]) + 1))
        by_sample = {(image["sample_id"], image["panorama_frame"]): image["id"]
                     for image in rebuilt["images"]}
        for item in rebuilt["annotations"]:
            assert item["image_id"] in by_sample.values()
        assert sorted(i["sample_id"] for i in rebuilt["images"]) == \
            sorted(i["sample_id"] for i in expected["images"])
    assert json.loads((experiment / "state" / "coco.json").read_text()) == status


def test_a_lost_shard_stops_the_split(experiment):
    run(experiment, "--shard-count", "2", "--shard-index", "0")
    result = subprocess.run(
        [sys.executable, str(DATASET / "_lib" / "export_coco.py"), str(experiment),
         "--shard-count", "2", "--merge"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "Shard 1 of train left no report" in result.stderr


def test_paths_are_written_relative_to_the_dataset_root(experiment):
    run(experiment)
    payload = json.loads((experiment / "coco" / "train.json").read_text())
    for image in payload["images"]:
        assert not Path(image["file_name"]).is_absolute()
        assert image["file_name"].startswith("rgb/train/")
        assert (experiment / image["file_name"]).is_file()
    assert "dataset root" in payload["info"]["file_name"]


def test_masks_are_run_length_encoded_not_polygons(experiment):
    run(experiment)
    payload = json.loads((experiment / "coco" / "train.json").read_text())
    segmentation = payload["annotations"][0]["segmentation"]
    assert isinstance(segmentation, dict)
    assert segmentation["size"] == [HEIGHT, WIDTH]
    assert isinstance(segmentation["counts"], list)
    assert sum(segmentation["counts"]) == HEIGHT * WIDTH
    trainer = (Path(__file__).resolve().parents[1] / "tools" / "3dfront_training"
               / "train_detector.py").read_text()
    assert 'cfg.INPUT.MASK_FORMAT = "bitmask"' in trainer
