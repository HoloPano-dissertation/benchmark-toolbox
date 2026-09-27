import importlib.util
import json
from pathlib import Path

import pytest

DATASET = Path(__file__).resolve().parents[1] / "tools" / "3dfront_dataset"


@pytest.fixture(scope="module")
def publish():
    spec = importlib.util.spec_from_file_location(
        "publish_annotations", DATASET / "publish_annotations.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def experiment(root):
    for split, view in (("train", "0"), ("val", "1"), ("test", "2")):
        scene = root / "ground_truth" / split / "house" / "Bedroom-1"
        scene.mkdir(parents=True, exist_ok=True)
        (scene / f"{view}.json").write_text(json.dumps({
            "layout": {"height": 2.6},
            "objects": [{"label": "bed", "bbox": {"size": [1, 2, 0.5]},
                         "attributes": {"source_glb": "house/Bedroom-1/Bed_1.glb",
                                        "shape": "/home/me/objects/bed/x/mesh.ply"}}],
            "metadata": {"units": "metres"}}))
        dense = root / "horizonnet" / split / "label_dense"
        corners = root / "horizonnet" / split / "label_cor"
        dense.mkdir(parents=True, exist_ok=True)
        corners.mkdir(parents=True, exist_ok=True)
        (dense / f"house__Bedroom-1__{view}.npz").write_bytes(b"npz")
        (corners / f"house__Bedroom-1__{view}.txt").write_text("1.0 2.0\n")
        manifest = root / "manifests_gt" / f"{split}.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({
            "sample_id": f"house/Bedroom-1/{view}",
            "input": f"{root}/rgb/{split}/house/Bedroom-1/{view}.png",
            "ground_truth": str(scene / f"{view}.json"),
            "metadata": {"split": split}}) + "\n", encoding="utf-8")
        coco = root / "coco"
        coco.mkdir(exist_ok=True)
        (coco / f"{split}.json").write_text(json.dumps({
            "categories": [{"id": 1, "name": "bed"}],
            "images": [{"id": 1, "file_name": f"rgb/{split}/house/Bedroom-1/{view}.png",
                        "sample_id": f"house/Bedroom-1/{view}", "panorama_frame": 0}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                             "bbox": [0, 0, 4, 4]}]}))
    return root


def test_scenes_mirror_the_panorama_layout(publish, tmp_path):
    root = experiment(tmp_path / "experiment")
    target = tmp_path / "annotations"
    records = publish.read_records(root, "train")
    assert publish.copy_views(root, target, records) == 1
    assert (target / "scenes" / "house" / "Bedroom-1" / "0.json").is_file()
    assert (target / "layout" / "house" / "Bedroom-1" / "0.npz").is_file()
    assert (target / "layout" / "house" / "Bedroom-1" / "0.txt").is_file()


def test_the_split_does_not_survive_in_the_paths(publish, tmp_path):
    root = experiment(tmp_path / "experiment")
    target = tmp_path / "annotations"
    for split in ("train", "val", "test"):
        publish.copy_views(root, target, publish.read_records(root, split))
    published = [str(p.relative_to(target)) for p in target.rglob("*.json")]
    assert published
    assert not any(part in path for path in published for part in ("train", "val", "test"))


def test_a_path_into_my_home_directory_never_leaves_with_the_scene(publish, tmp_path):
    root = experiment(tmp_path / "experiment")
    target = tmp_path / "annotations"
    publish.copy_views(root, target, publish.read_records(root, "train"))
    text = (target / "scenes" / "house" / "Bedroom-1" / "0.json").read_text()
    assert "/home/me/" not in text
    assert "shape" not in json.loads(text)["objects"][0]["attributes"]
    assert json.loads(text)["objects"][0]["attributes"]["source_glb"]


def test_the_three_coco_files_become_one_numbered_from_one(publish, tmp_path):
    root = experiment(tmp_path / "experiment")
    payload = publish.joined_coco(root)
    assert [image["id"] for image in payload["images"]] == [1, 2, 3]
    assert [item["id"] for item in payload["annotations"]] == [1, 2, 3]
    assert {item["image_id"] for item in payload["annotations"]} == {1, 2, 3}


def test_coco_points_at_the_panoramas_of_the_share(publish, tmp_path):
    root = experiment(tmp_path / "experiment")
    payload = publish.joined_coco(root)
    for image in payload["images"]:
        assert image["file_name"].startswith("panoramas/png/house/Bedroom-1/")
        assert "train" not in image["file_name"] and "rgb/" not in image["file_name"]


def test_an_unexpected_image_path_is_refused_not_guessed(publish):
    with pytest.raises(ValueError, match="Unexpected image path"):
        publish.panorama_name("/somewhere/else/0.png")
