import importlib.util
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
Image = pytest.importorskip("PIL.Image")

RENDERER = Path(__file__).resolve().parents[1] / "tools" / "3dfront_panorama_renderer"


@pytest.fixture(scope="module")
def export_images():
    spec = importlib.util.spec_from_file_location("export_images", RENDERER / "export_images.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def room_with_views(root, views=4, height=8, width=16, finished=True):
    root.mkdir(parents=True, exist_ok=True)
    for index in range(views):
        colours = np.full((height, width, 3), index * 10, dtype=np.uint8)
        with h5py.File(root / f"{index}.hdf5", "w") as target:
            target.create_dataset("colors", data=colours)
            target.create_dataset("depth", data=np.zeros((height, width), dtype=np.float32))
    if finished:
        (root / ".complete").write_text("done\n", encoding="utf-8")
    return root


def test_every_view_becomes_an_image(export_images, tmp_path):
    room = room_with_views(tmp_path / "hdf5" / "house" / "Bedroom-1")
    target = tmp_path / "png" / "house" / "Bedroom-1"
    assert export_images.export_room(room, target, "png", 92) == 4
    assert sorted(p.name for p in target.iterdir()) == ["0.png", "1.png", "2.png", "3.png"]
    assert np.asarray(Image.open(target / "2.png"))[0, 0, 0] == 20


def test_work_already_done_is_not_repeated(export_images, tmp_path):
    room = room_with_views(tmp_path / "hdf5" / "house" / "Bedroom-1")
    target = tmp_path / "png" / "house" / "Bedroom-1"
    export_images.export_room(room, target, "png", 92)
    assert export_images.export_room(room, target, "png", 92) == 0


def test_an_image_of_zero_length_is_written_again(export_images, tmp_path):
    """An empty file is what a killed job leaves behind; it is not an exported view."""
    room = room_with_views(tmp_path / "hdf5" / "house" / "Bedroom-1")
    target = tmp_path / "png" / "house" / "Bedroom-1"
    export_images.export_room(room, target, "png", 92)
    (target / "1.png").write_bytes(b"")
    assert export_images.export_room(room, target, "png", 92) == 1


def test_only_finished_rooms_are_listed(export_images, tmp_path):
    hdf5_root = tmp_path / "hdf5"
    room_with_views(hdf5_root / "house" / "Bedroom-1")
    room_with_views(hdf5_root / "house" / "Bedroom-2", finished=False)
    listed = export_images.finished_rooms(hdf5_root)
    assert [room.name for room in listed] == ["Bedroom-1"]


def test_no_leftover_part_files(export_images, tmp_path):
    room = room_with_views(tmp_path / "hdf5" / "house" / "Bedroom-1")
    target = tmp_path / "png" / "house" / "Bedroom-1"
    export_images.export_room(room, target, "png", 92)
    assert not list(target.glob("*.part"))


def test_a_room_without_views_is_an_error_not_an_empty_job(export_images, tmp_path):
    room = tmp_path / "hdf5" / "house" / "Bedroom-1"
    room.mkdir(parents=True)
    (room / ".complete").write_text("done\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="no views"):
        export_images.export_room(room, tmp_path / "png" / "house" / "Bedroom-1", "png", 92)
