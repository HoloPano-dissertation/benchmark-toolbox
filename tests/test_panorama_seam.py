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
MODULE_PATH = DATASET / "_lib" / "panorama_seam.py"
spec = importlib.util.spec_from_file_location("panorama_seam", MODULE_PATH)
panorama_seam = importlib.util.module_from_spec(spec)
spec.loader.exec_module(panorama_seam)

WIDTH, HEIGHT = 128, 64


def mask_with_columns(columns):
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[20:40, list(columns)] = True
    return mask


def test_object_away_from_the_seam_stays_in_the_rendered_frame():
    frames = panorama_seam.annotation_masks(mask_with_columns(range(50, 70)))
    assert len(frames) == 1 and frames[0][0] == 0
    assert panorama_seam.tight_box(frames[0][1]) == [50, 20, 20, 20]


def test_object_across_the_seam_moves_to_the_rolled_frame():
    mask = mask_with_columns(list(range(0, 6)) + list(range(WIDTH - 6, WIDTH)))
    assert panorama_seam.tight_box(mask)[2] == WIDTH
    frame, rolled = panorama_seam.annotation_masks(mask)[0]
    assert frame == 1
    assert panorama_seam.tight_box(rolled)[2] == 12


def test_object_across_both_seams_is_split_without_losing_pixels():
    columns = list(range(0, 10)) + list(range(56, 72)) + list(range(WIDTH - 10, WIDTH))
    mask = mask_with_columns(columns)
    frames = panorama_seam.annotation_masks(mask)
    assert len(frames) > 1
    joined = np.zeros_like(mask)
    for _, piece in frames:
        assert panorama_seam.tight_box(piece)[2] < WIDTH
        joined |= piece
    assert np.array_equal(joined, mask)


def test_occlusion_gap_alone_does_not_split_an_object():
    mask = mask_with_columns(list(range(30, 40)) + list(range(70, 80)))
    assert len(panorama_seam.annotation_masks(mask)) == 1


def build_experiment(root, instances, attributes, labels=None):
    labels = labels or {"Bed_a_0.glb": "bed", "Chair_b_1.glb": "chair"}
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
            handle.create_dataset("instance_attribute_maps", data=json.dumps(attributes))
        gt = root / "gt" / split / "0.json"
        gt.parent.mkdir(parents=True, exist_ok=True)
        gt.write_text(json.dumps({"objects": [
            {"object_id": name[:-4], "label": label,
             "attributes": {"source_glb": "house/room/" + name}}
            for name, label in labels.items()], "relations": [], "metadata": {}}))
        manifest = root / "manifests_gt" / f"{split}.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({
            "sample_id": f"{split}-house/room/0", "input": str(rgb),
            "ground_truth": str(gt),
            "metadata": {"hdf5": str(hdf5), "split": split}}) + "\n", encoding="utf-8")
    (root / "state").mkdir(exist_ok=True)


def run_export(root):
    result = subprocess.run(
        [sys.executable, str(DATASET / "_lib" / "export_coco.py"), str(root)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return json.loads((root / "coco" / "train.json").read_text())


@pytest.fixture
def seam_experiment(tmp_path):
    instances = np.zeros((HEIGHT, WIDTH), dtype=np.int64)
    instances[20:40, :6] = 1
    instances[20:40, WIDTH - 6:] = 1
    instances[10:30, 50:70] = 2
    build_experiment(tmp_path, instances, [
        {"idx": 1, "source_label": "Bed", "source_file": "Bed_a_0.glb"},
        {"idx": 2, "source_label": "Chair", "source_file": "Chair_b_1.glb"}])
    return tmp_path


def test_export_produces_no_frame_wide_annotation(seam_experiment):
    coco = run_export(seam_experiment)
    assert max(a["bbox"][2] for a in coco["annotations"]) < WIDTH * 0.9


def test_export_writes_the_rolled_frame_it_annotates_in(seam_experiment):
    coco = run_export(seam_experiment)
    rolled = [image for image in coco["images"] if image.get("panorama_frame") == 1]
    assert len(rolled) == 1
    rendered = np.asarray(Image.open(seam_experiment / "rgb" / "train" / "0.png"))
    assert np.array_equal(np.asarray(Image.open(rolled[0]["file_name"])),
                          np.roll(rendered, WIDTH // 2, axis=1))


def test_every_panorama_keeps_its_rendered_frame(tmp_path):
    instances = np.zeros((HEIGHT, WIDTH), dtype=np.int64)
    instances[20:40, :6] = 1
    instances[20:40, WIDTH - 6:] = 1
    build_experiment(tmp_path, instances,
                     [{"idx": 1, "source_label": "Bed", "source_file": "Bed_a_0.glb"}],
                     labels={"Bed_a_0.glb": "bed"})
    coco = run_export(tmp_path)
    assert sorted(image.get("panorama_frame", 0) for image in coco["images"]) == [0, 1]
    assert len(coco["annotations"]) == 1
