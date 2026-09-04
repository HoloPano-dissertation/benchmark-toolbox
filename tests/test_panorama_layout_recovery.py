import importlib.util
import json
from pathlib import Path
import struct

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("shapely")
MODULE_DIR = Path(__file__).resolve().parents[1] / "tools" / "3dfront_panorama_renderer"


def rectangle(x0, y0, x1, y1, z):
    return np.array([[[x0, y0, z], [x1, y0, z], [x1, y1, z]],
                     [[x0, y0, z], [x1, y1, z], [x0, y1, z]]], dtype=float)


def write_triangles(path, triangles):
    points = triangles.reshape(-1, 3)[:, [0, 2, 1]].copy()
    points[:, 2] *= -1  # Blender -> glTF
    points = points.astype("<f4")
    binary = points.tobytes()
    document = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
                "nodes": [{"mesh": 0}], "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
                "buffers": [{"byteLength": len(binary)}],
                "bufferViews": [{"buffer": 0, "byteLength": len(binary)}],
                "accessors": [{"bufferView": 0, "componentType": 5126, "type": "VEC3", "count": len(points),
                               "min": points.min(0).tolist(), "max": points.max(0).tolist()}]}
    content = json.dumps(document).encode()
    content += b" " * (-len(content) % 4)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 28+len(content)+len(binary))
                     + struct.pack("<II", len(content), 0x4E4F534A) + content
                     + struct.pack("<II", len(binary), 0x004E4942) + binary)


def room_fixture(tmp_path, extra=None):
    parts = [rectangle(0, 0, 3, 3, 0), rectangle(0, 0, 3, 3, 2)[:, ::-1]]
    if extra is not None:
        parts.append(extra)
    write_triangles(tmp_path / "ceil.glb", np.concatenate(parts))
    write_triangles(tmp_path / "Bed_example.glb", rectangle(0.8, 0.8, 2.2, 2.2, 0.5))


def test_merged_file_floor_is_not_treated_as_missing(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(MODULE_DIR))
    from room_layout import recover_layout
    room_fixture(tmp_path)
    layout = recover_layout(tmp_path)
    assert layout["floor_z"] == pytest.approx(0)
    assert layout["ceiling_z"] == pytest.approx(2)
    assert layout["area"] == pytest.approx(9)
    assert not layout["reconstruct_floor"]
    assert not layout["reconstruct_ceiling"]
    assert layout["sources"]["floor_height"] == ["ceil.glb"]


def test_unrelated_large_planes_do_not_move_camera_bounds(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(MODULE_DIR))
    from room_layout import recover_layout
    extras = np.concatenate([rectangle(100, 100, 150, 150, -3),
                             rectangle(100, 100, 150, 150, 3)[:, ::-1]])
    room_fixture(tmp_path, extra=extras)
    layout = recover_layout(tmp_path)
    assert layout["bounds_min"] == pytest.approx([0, 0, 0])
    assert layout["bounds_max"] == pytest.approx([3, 3, 2])


def test_missing_ceiling_fails_without_inventing_surface(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(MODULE_DIR))
    from room_layout import recover_layout
    write_triangles(tmp_path / "floor.glb", rectangle(0, 0, 3, 3, 0))
    write_triangles(tmp_path / "Bed_example.glb", rectangle(0.8, 0.8, 2.2, 2.2, 0.5))
    with pytest.raises(ValueError, match="ceiling"):
        recover_layout(tmp_path)


def test_floor_slab_underside_is_not_a_second_room_floor(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(MODULE_DIR))
    from room_layout import recover_layout
    room_fixture(tmp_path, extra=rectangle(-0.2, -0.2, 3.2, 3.2, 0)[:, ::-1])
    layout = recover_layout(tmp_path)
    assert layout["area"] == pytest.approx(9)


def test_glb_header_bounds_match_full_geometry(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(MODULE_DIR))
    from glb_geometry import glb_bounds, glb_triangles
    room_fixture(tmp_path)
    actual = glb_triangles(tmp_path / "ceil.glb").reshape(-1, 3)
    lo, hi = glb_bounds(tmp_path / "ceil.glb")
    np.testing.assert_allclose(lo, actual.min(0))
    np.testing.assert_allclose(hi, actual.max(0))


def test_suspended_ceiling_is_below_the_larger_roof_slab(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(MODULE_DIR))
    from room_layout import recover_layout
    room_fixture(tmp_path, extra=rectangle(0, 0, 3, 1.6, 1.4)[:, ::-1])
    layout = recover_layout(tmp_path)
    assert layout["ceiling_z"] == pytest.approx(1.4)
    assert layout["floor_z"] == pytest.approx(0)
    assert "floor_ceiling_footprints_differ" in layout["warnings"]


def test_wide_shelf_below_furniture_top_is_not_the_ceiling(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(MODULE_DIR))
    from room_layout import recover_layout
    room_fixture(tmp_path, extra=rectangle(0, 0, 3, 1.6, 0.3)[:, ::-1])
    layout = recover_layout(tmp_path)
    assert layout["ceiling_z"] == pytest.approx(2)


def test_camera_near_clipping_scales_with_tiny_glb_rooms(monkeypatch):
    monkeypatch.syspath_prepend(str(MODULE_DIR))
    from camera_policy import camera_clip_planes
    near, far = camera_clip_planes(0.08, 0.003)
    assert 0 < near < 0.003
    assert far > 0.08
    large = camera_clip_planes(8, 0.3)
    assert large == pytest.approx((near*100, far*100))
