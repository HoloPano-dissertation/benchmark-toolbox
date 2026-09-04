import importlib.util
import json
from pathlib import Path
import struct

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("shapely")
MODULE_DIR = Path(__file__).resolve().parents[1] / "tools" / "3dfront_panorama_renderer"
DATASET_DIR = MODULE_DIR.parent / "3dfront_dataset"
spec = importlib.util.spec_from_file_location("panorama_glb_geometry", MODULE_DIR / "glb_geometry.py")
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)


def rectangle(x0, y0, x1, y1, z=0):
    return np.array([[[x0, y0, z], [x1, y0, z], [x1, y1, z]],
                     [[x0, y0, z], [x1, y1, z], [x0, y1, z]]], dtype=float)


def test_l_shaped_floor_rejects_empty_bbox_corner():
    triangles = np.concatenate([rectangle(0, 0, 2, 1), rectangle(0, 1, 1, 2)])
    footprint, horizontal = geometry.floor_footprint(triangles)
    bad = geometry.camera_floor_check([1.5, 1.5, 1], footprint, horizontal)
    good = geometry.camera_floor_check([0.5, 1.5, 1], footprint, horizontal)
    assert not bad["over_floor_footprint"] and not bad["floor_below_camera"]
    assert bad["floor_boundary_margin"] < 0
    assert good["over_floor_footprint"] and good["floor_below_camera"]
    metrics = geometry.footprint_metrics(footprint, [0, 0, 0], [2, 2, 2])
    assert metrics["floor_proxy_iou"] == pytest.approx(0.75)
    assert metrics["floor_oriented_bbox_fill"] == pytest.approx(0.75)


def test_floor_hole_and_disconnected_components():
    triangles = np.concatenate([rectangle(0, 0, 3, 1), rectangle(0, 2, 3, 3),
                                rectangle(0, 1, 1, 2), rectangle(2, 1, 3, 2)])
    footprint, horizontal = geometry.floor_footprint(triangles)
    assert len(footprint.interiors) == 1
    assert not geometry.camera_floor_check([1.5, 1.5, 1], footprint, horizontal)["floor_below_camera"]
    separate, _ = geometry.floor_footprint(np.concatenate([rectangle(0, 0, 1, 1), rectangle(2, 0, 3, 1)]))
    assert geometry.footprint_metrics(separate, [0, 0, 0], [3, 1, 2])["floor_components"] == 2


def test_winding_overlap_and_slab_do_not_double_area():
    floor = rectangle(0, 0, 2, 3)
    slab = np.concatenate([floor, floor[:, ::-1], rectangle(0, 0, 2, 3, -0.1)])
    footprint, horizontal = geometry.floor_footprint(slab)
    assert footprint.area == pytest.approx(6)
    assert geometry.camera_floor_check([1, 1, 1.2], footprint, horizontal)["eye_height_above_floor"] == pytest.approx(1.2)


def test_floor_above_camera_not_accepted():
    footprint, horizontal = geometry.floor_footprint(rectangle(0, 0, 1, 1, 2))
    check = geometry.camera_floor_check([0.5, 0.5, 1], footprint, horizontal)
    assert check["over_floor_footprint"]
    assert not check["floor_below_camera"]


def test_rotated_rectangle_is_not_nonrectangular():
    triangles = rectangle(-1, -1, 1, 1)
    angle = np.pi / 4
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    triangles[:, :, :2] = triangles[:, :, :2] @ rotation.T
    footprint, _ = geometry.floor_footprint(triangles)
    metrics = geometry.footprint_metrics(footprint, [-2**0.5, -2**0.5, 0], [2**0.5, 2**0.5, 2])
    assert metrics["floor_proxy_iou"] == pytest.approx(0.5)
    assert metrics["floor_oriented_bbox_fill"] == pytest.approx(1)


def test_vertical_walls_not_floor():
    wall = np.array([[[0, 0, 0], [1, 0, 0], [1, 0, 2]]], dtype=float)
    with pytest.raises(ValueError, match="horizontal"):
        geometry.floor_footprint(wall)


def write_fixture(path, indexed=True, translation=(0, 0, 0), matrix=None, mode=4):
    points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 0, 1]], dtype="<f4")
    binary = points.tobytes() + np.asarray([0, 1, 2], dtype="<u2").tobytes() + b"\0\0"
    node = {"mesh": 0, "translation": list(translation)}
    if matrix is not None:
        node = {"mesh": 0, "matrix": matrix.T.reshape(-1).tolist()}
    primitive = {"attributes": {"POSITION": 0}, "mode": mode}
    if indexed:
        primitive["indices"] = 1
    document = {"asset": {"version": "2.0"}, "scene": 0,
                "scenes": [{"nodes": [0]}], "nodes": [{"translation": [10, 20, 30], "children": [1]}, node],
                "meshes": [{"primitives": [primitive]}], "buffers": [{"byteLength": len(binary)}],
                "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36},
                                {"buffer": 0, "byteOffset": 36, "byteLength": 6}],
                "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
                              {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}]}
    content = json.dumps(document).encode()
    content += b" " * (-len(content) % 4)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 28+len(content)+len(binary))
                     + struct.pack("<II", len(content), 0x4E4F534A) + content
                     + struct.pack("<II", len(binary), 0x004E4942) + binary)


@pytest.mark.parametrize("indexed", [False, True])
def test_glb_hierarchy_and_axis_conversion(tmp_path, indexed):
    path = tmp_path / "floor.glb"
    write_fixture(path, indexed=indexed, translation=(1, 2, 3))
    triangles = geometry.glb_triangles(path)
    np.testing.assert_allclose(triangles, [[[11, -33, 22], [12, -33, 22], [11, -34, 22]]])


def test_glb_matrix_scale_and_translation(tmp_path):
    path = tmp_path / "floor.glb"
    matrix = np.diag([2., 3., 4., 1.])
    matrix[:3, 3] = [1, 2, 3]
    write_fixture(path, matrix=matrix)
    np.testing.assert_allclose(geometry.glb_triangles(path), [[[11, -33, 22], [13, -33, 22], [11, -37, 22]]])


def test_unsupported_primitives_fail_closed(tmp_path):
    path = tmp_path / "lines.glb"
    write_fixture(path, mode=1)
    with pytest.raises(ValueError, match="TRIANGLES"):
        geometry.glb_triangles(path)


def test_image_flags(tmp_path, monkeypatch):
    h5py = pytest.importorskip("h5py")
    monkeypatch.syspath_prepend(str(MODULE_DIR))
    monkeypatch.syspath_prepend(str(DATASET_DIR))
    from _lib import quality
    path = tmp_path / "0.hdf5"
    with h5py.File(path, "w") as f:
        f["colors"] = np.zeros((16, 32, 3), dtype=np.uint8)
        f["depth"] = np.full((16, 32), np.inf)
    metrics, flags, image = quality.image_checks(path, 2)
    assert "mostly_dark" in flags and "missing_depth" in flags
    assert metrics["depth_valid_fraction"] == 0
    assert image.size == (32, 16)


def test_missing_layout_metadata_is_not_a_successful_audit(tmp_path, monkeypatch):
    h5py = pytest.importorskip("h5py")
    monkeypatch.syspath_prepend(str(MODULE_DIR))
    monkeypatch.syspath_prepend(str(DATASET_DIR))
    import audit as dataset_audit
    root, scene, report = tmp_path / "experiment", tmp_path / "source", tmp_path / "report"
    room = scene / "house" / "bedroom"
    rendered = root / "outputs" / "train" / "house" / "bedroom"
    room.mkdir(parents=True)
    rendered.mkdir(parents=True)
    (report / "previews").mkdir(parents=True)
    (report / "rooms").mkdir()
    write_fixture(room / "wall.glb")
    write_fixture(room / "ceil.glb", translation=(0, 2, 0))
    cameras = [[10.1+i*0.1, -30.2, 21] for i in range(4)]
    (rendered / "render.json").write_text(json.dumps({"camera_locations": cameras}))
    for index in range(4):
        with h5py.File(rendered / f"{index}.hdf5", "w") as f:
            f["colors"] = np.full((16, 32, 3), 50+index, dtype=np.uint8)
            f["depth"] = np.ones((16, 32))
    result = dataset_audit.audit(({"room_id": "house/bedroom", "split": "train", "room_dir": str(room)},
                                  str(root / "outputs"), str(report)))
    assert "audit_error" in result["flags"]
    assert not result["geometry_pass"]
