import importlib.util
import json
from pathlib import Path
import struct

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("shapely")
RENDERER = Path(__file__).resolve().parents[1] / "tools" / "3dfront_panorama_renderer"
spec = importlib.util.spec_from_file_location("camera_policy", RENDERER / "camera_policy.py")
camera_policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camera_policy)

FLOOR_Z, ROOM_HEIGHT = 0.0, 1.5
EYE_Z = FLOOR_Z + 0.6 * ROOM_HEIGHT
BED = ((0.0, 0.0, 0.0), (1.2, 1.0, 0.55))


@pytest.fixture(autouse=True)
def renderer_on_path(monkeypatch):
    monkeypatch.syspath_prepend(str(RENDERER))


def test_camera_over_a_bed_is_rejected_although_it_is_above_the_box():
    candidate = (0.6, 0.5, EYE_Z)
    assert not all(BED[0][axis] <= candidate[axis] <= BED[1][axis] for axis in range(3))
    assert camera_policy.stands_on_furniture(candidate, [BED], FLOOR_Z, ROOM_HEIGHT)


def test_camera_beside_a_bed_and_over_a_rug_are_accepted():
    rug = ((0.0, 0.0, 0.0), (2.0, 2.0, 0.02))
    assert not camera_policy.stands_on_furniture((1.5, 0.5, EYE_Z), [BED], FLOOR_Z, ROOM_HEIGHT)
    assert not camera_policy.stands_on_furniture((1.0, 1.0, EYE_Z), [rug], FLOOR_Z, ROOM_HEIGHT)


def test_fallback_is_taken_only_when_the_room_offers_nothing_else():
    free = [(0.5, (0, 0, 0)), (0.4, (1, 0, 0)), (0.3, (2, 0, 0)), (0.2, (3, 0, 0))]
    over = [(0.9, (4, 0, 0))]
    assert camera_policy.with_furniture_fallback(free, over, views=4) == (free, False)
    assert camera_policy.with_furniture_fallback([], over, views=4) == (over, True)


def rectangle(x0, y0, x1, y1, z):
    return np.array([[[x0, y0, z], [x1, y0, z], [x1, y1, z]],
                     [[x0, y0, z], [x1, y1, z], [x0, y1, z]]], dtype=float)


def write_triangles(path, triangles):
    points = triangles.reshape(-1, 3)[:, [0, 2, 1]].copy()
    points[:, 2] *= -1
    points = points.astype("<f4")
    binary = points.tobytes()
    document = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
                "nodes": [{"mesh": 0}],
                "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
                "buffers": [{"byteLength": len(binary)}],
                "bufferViews": [{"buffer": 0, "byteLength": len(binary)}],
                "accessors": [{"bufferView": 0, "componentType": 5126, "type": "VEC3",
                               "count": len(points), "min": points.min(0).tolist(),
                               "max": points.max(0).tolist()}]}
    content = json.dumps(document).encode()
    content += b" " * (-len(content) % 4)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 28 + len(content) + len(binary))
                     + struct.pack("<II", len(content), 0x4E4F534A) + content
                     + struct.pack("<II", len(binary), 0x004E4942) + binary)


def build_room(tmp_path, with_shell=True):
    room = tmp_path / "house" / "Bedroom-1"
    room.mkdir(parents=True)
    write_triangles(room / "floor.glb", rectangle(0.0, 0.0, 9.0, 3.0, 0.0))
    write_triangles(room / "ceil.glb", rectangle(0.0, 0.0, 9.0, 3.0, 2.6)[:, ::-1])
    write_triangles(room / "Bed_a_0.glb", rectangle(0.8, 0.8, 2.2, 2.2, 0.5))
    if with_shell:
        write_triangles(room.parent / "Bedroom-1.glb",
                        np.concatenate([rectangle(0.0, 0.0, 3.0, 3.0, 0.0),
                                        rectangle(0.0, 0.0, 3.0, 3.0, 2.6)[:, ::-1]]))
    return room


def test_contour_is_clipped_to_the_packaged_room(tmp_path):
    from room_layout import recover_layout
    layout = recover_layout(build_room(tmp_path))
    assert layout["clipped_to_room_shell"] is True
    assert layout["area"] == pytest.approx(9.0, rel=0.1)


def test_without_a_shell_the_contour_is_left_alone(tmp_path):
    from room_layout import recover_layout
    layout = recover_layout(build_room(tmp_path, with_shell=False))
    assert layout["clipped_to_room_shell"] is False
    assert layout["area"] == pytest.approx(27.0, rel=0.1)


def test_a_patch_outside_the_room_is_dropped(tmp_path):
    from room_layout import clip_patches_to_shell, room_shell_footprint
    from shapely.geometry import box
    outside = {"polygon": box(20.0, 20.0, 21.0, 21.0), "area": 1.0,
               "fragment_area": 0.0, "facing": "up", "z": 0.0}
    assert clip_patches_to_shell([outside], room_shell_footprint(build_room(tmp_path))) == []
