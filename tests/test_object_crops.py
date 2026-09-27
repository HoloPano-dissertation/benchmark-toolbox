import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

np = pytest.importorskip("numpy")
h5py = pytest.importorskip("h5py")
Image = pytest.importorskip("PIL.Image")

DATASET = Path(__file__).resolve().parents[1] / "tools" / "3dfront_dataset"
spec = importlib.util.spec_from_file_location("export_shape", DATASET / "_lib" / "export_shape.py")
export_shape = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_shape)

WIDTH, HEIGHT = 128, 64


def test_object_id_is_unique_per_instance_and_keeps_the_source_path():
    first = export_shape.object_id("house/room/Bed_uuid_1.glb")
    second = export_shape.object_id("house/other-room/Bed_uuid_1.glb")
    assert first != second
    assert first == "house__room__Bed_uuid_1"


def build(root, instances, objects):
    for split in ("train", "val", "test"):
        rgb = root / "rgb" / split / "0.png"
        rgb.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(
            np.tile(np.arange(WIDTH, dtype=np.uint8)[None, :, None], (HEIGHT, 1, 3))
        ).save(rgb)
        hdf5 = root / "outputs" / split / "0.hdf5"
        hdf5.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(hdf5, "w") as handle:
            handle.create_dataset("instance_segmaps", data=instances)
            handle.create_dataset("instance_attribute_maps", data=json.dumps(
                [{"idx": 1, "source_label": "Bed", "source_file": "Bed_a_0.glb"}]))
        gt = root / "gt" / split / "0.json"
        gt.parent.mkdir(parents=True, exist_ok=True)
        gt.write_text(json.dumps({"objects": objects, "relations": [], "metadata": {}}))
        manifest = root / "manifests_gt" / f"{split}.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({
            "sample_id": f"{split}-house/room/0", "input": str(rgb),
            "ground_truth": str(gt),
            "metadata": {"hdf5": str(hdf5), "split": split}}) + "\n", encoding="utf-8")
    (root / "state").mkdir(exist_ok=True)
    mesh = root / "objects" / "bed" / "house__room__Bed_a_0" / "mesh.ply"
    mesh.parent.mkdir(parents=True, exist_ok=True)
    mesh.write_bytes(b"ply\n")
    return mesh


def run(root):
    result = subprocess.run(
        [sys.executable, str(DATASET / "_lib" / "export_crops.py"), str(root)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return json.loads((root / "state" / "crops.json").read_text())


@pytest.fixture
def experiment(tmp_path):
    instances = np.zeros((HEIGHT, WIDTH), dtype=np.int64)
    instances[20:40, 50:70] = 1
    build(tmp_path, instances, [{
        "object_id": "Bed_a_0", "label": "bed",
        "attributes": {"source_glb": "house/room/Bed_a_0.glb",
                       "shape": str(mesh_path(tmp_path))}}])
    return tmp_path, mesh_path(tmp_path)


def mesh_path(root):
    return root / "objects" / "bed" / "house__room__Bed_a_0" / "mesh.ply"


def test_crop_lands_beside_the_object_mesh(experiment):
    root, _ = experiment
    status = run(root)
    assert status["crops"] == {"train": 1, "val": 1, "test": 1}
    crops = sorted(mesh_path(root).parent.glob("crop-*.png"))
    assert len(crops) == 3


def test_split_files_use_the_class_and_object_folders(experiment):
    root, _ = experiment
    run(root)
    stems = json.loads((root / "objects" / "train.json").read_text())
    assert stems == ["bed/house__room__Bed_a_0/crop-train-house__room__0"]


def test_seam_crossing_object_is_cropped_from_the_rolled_frame(tmp_path):
    instances = np.zeros((HEIGHT, WIDTH), dtype=np.int64)
    instances[20:40, :6] = 1
    instances[20:40, WIDTH - 6:] = 1
    build(tmp_path, instances, [{
        "object_id": "Bed_a_0", "label": "bed",
        "attributes": {"source_glb": "house/room/Bed_a_0.glb",
                       "shape": str(mesh_path(tmp_path))}}])
    run(tmp_path)
    crop = sorted(mesh_path(tmp_path).parent.glob("crop-*.png"))[0]
    with Image.open(crop) as handle:
        assert handle.width < WIDTH * 0.5


def test_an_invisible_object_is_skipped_not_failed(tmp_path):
    instances = np.zeros((HEIGHT, WIDTH), dtype=np.int64)
    build(tmp_path, instances, [{
        "object_id": "Bed_a_0", "label": "bed",
        "attributes": {"source_glb": "house/room/Bed_a_0.glb",
                       "shape": str(mesh_path(tmp_path))}}])
    status = run(tmp_path)
    assert status["crops"] == {}
    assert status["skipped"]["not_visible"] == 3


def run_shard(root, index, count, merge=False):
    command = [sys.executable, str(DATASET / "_lib" / "export_crops.py"), str(root),
               "--shard-count", str(count)]
    command += ["--merge"] if merge else ["--shard-index", str(index)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result


def test_shards_write_the_same_catalogue_as_one_pass(experiment):
    root, _ = experiment
    whole = run(root)
    expected = json.loads((root / "objects" / "train.json").read_text())
    for split in ("train", "val", "test"):
        (root / "objects" / f"{split}.json").unlink()
    for index in range(2):
        run_shard(root, index, 2)
    run_shard(root, 0, 2, merge=True)
    assert json.loads((root / "state" / "crops.json").read_text()) == whole
    assert json.loads((root / "objects" / "train.json").read_text()) == expected


def test_a_lost_shard_stops_the_catalogue(experiment):
    root, _ = experiment
    run_shard(root, 0, 2)
    result = subprocess.run(
        [sys.executable, str(DATASET / "_lib" / "export_crops.py"), str(root),
         "--shard-count", "2", "--merge"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "Shard 1 of train left no report" in result.stderr
