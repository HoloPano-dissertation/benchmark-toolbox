import json
from pathlib import Path
import subprocess
import sys

import pytest

np = pytest.importorskip("numpy")
h5py = pytest.importorskip("h5py")
Image = pytest.importorskip("PIL.Image")

DATASET = Path(__file__).resolve().parents[1] / "tools" / "3dfront_dataset"
HEIGHT, WIDTH, VIEWS = 8, 16, 4


def build(root):
    room = "house/Bedroom-1"
    render = root / "outputs" / "train" / room
    render.mkdir(parents=True, exist_ok=True)
    for view in range(VIEWS):
        with h5py.File(render / f"{view}.hdf5", "w") as target:
            target.create_dataset("colors", data=np.full(
                (HEIGHT, WIDTH, 3), view * 10, dtype=np.uint8))
    (render / ".complete").write_text("done\n", encoding="utf-8")
    (render / "render.json").write_text(json.dumps({
        "camera_locations": [[float(v), 0.0, 0.0] for v in range(VIEWS)],
        "bounds_min": [-1.0, -1.0, -1.0], "bounds_max": [9.0, 1.0, 1.0],
        "resolution": [WIDTH, HEIGHT]}))
    splits = root / "splits"
    splits.mkdir(exist_ok=True)
    (splits / "rooms.jsonl").write_text(json.dumps(
        {"room_id": room, "house_id": "house", "split": "train"}) + "\n")
    (root / "state").mkdir(exist_ok=True)
    return root


def run(root):
    result = subprocess.run(
        [sys.executable, str(DATASET / "_lib" / "export_rgb.py"), str(root)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result


def test_every_view_becomes_an_image(tmp_path):
    run(build(tmp_path))
    images = sorted((tmp_path / "rgb" / "train" / "house" / "Bedroom-1").iterdir())
    assert [p.name for p in images] == ["0.png", "1.png", "2.png", "3.png"]
    assert np.asarray(Image.open(images[2]))[0, 0, 0] == 20


def test_an_empty_image_left_by_a_killed_shard_is_written_again(tmp_path):
    run(build(tmp_path))
    torn = tmp_path / "rgb" / "train" / "house" / "Bedroom-1" / "1.png"
    torn.write_bytes(b"")
    run(tmp_path)
    assert torn.stat().st_size
    assert np.asarray(Image.open(torn))[0, 0, 0] == 10


def test_a_finished_image_is_not_written_twice(tmp_path):
    run(build(tmp_path))
    kept = tmp_path / "rgb" / "train" / "house" / "Bedroom-1" / "0.png"
    stamp = kept.stat().st_mtime_ns
    run(tmp_path)
    assert kept.stat().st_mtime_ns == stamp


def test_nothing_half_written_is_left_behind(tmp_path):
    run(build(tmp_path))
    folder = tmp_path / "rgb" / "train" / "house" / "Bedroom-1"
    assert not list(folder.glob("*.part"))
