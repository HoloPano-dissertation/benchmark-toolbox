import importlib.util
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

DATASET = Path(__file__).resolve().parents[1] / "tools" / "3dfront_dataset"
MODULE_PATH = DATASET / "_lib" / "export_shape.py"
spec = importlib.util.spec_from_file_location("export_shape", MODULE_PATH)
export_shape = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_shape)

from benchmark_toolbox.metrics.shape import load_shape_points  # noqa: E402

IDENTITY = np.eye(3)


def cube_triangles(low, high):
    corners = np.array([[x, y, z] for x in (low[0], high[0])
                        for y in (low[1], high[1]) for z in (low[2], high[2])], dtype=float)
    faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
             (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    return np.array([[corners[a], corners[b], corners[c]] for a, b, c in faces])


def test_mesh_is_placed_in_the_unit_box_of_the_instance():
    triangles = cube_triangles((2.0, 2.0, 0.0), (3.0, 4.0, 1.0))
    bbox = {"center": [3.0, 4.0, 0.0], "size": [2.0, 4.0, 2.0], "basis": IDENTITY.tolist()}
    vertices, faces = export_shape.unit_mesh(
        triangles, bbox, camera=np.array([2.0, 2.0, 1.0]), scale=2.0)
    assert vertices.min() == pytest.approx(-0.5)
    assert vertices.max() == pytest.approx(0.5)
    assert len(faces) == len(triangles)


def test_a_rotated_instance_maps_onto_the_same_unit_box():
    triangles = cube_triangles((-0.5, -1.0, -0.25), (0.5, 1.0, 0.25))
    angle = np.pi / 3
    basis = np.array([[np.cos(angle), np.sin(angle), 0.0],
                      [-np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    rotated = triangles.reshape(-1, 3) @ basis
    bbox = {"center": [0.0, 0.0, 0.0], "size": [1.0, 2.0, 0.5], "basis": basis.tolist()}
    vertices, _ = export_shape.unit_mesh(
        rotated.reshape(triangles.shape), bbox, camera=np.zeros(3), scale=1.0)
    assert vertices.min() == pytest.approx(-0.5, abs=1e-9)
    assert vertices.max() == pytest.approx(0.5, abs=1e-9)


def test_written_mesh_is_read_back_by_the_shape_metric(tmp_path):
    triangles = cube_triangles((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))
    bbox = {"center": [0.0, 0.0, 0.0], "size": [1.0, 1.0, 1.0], "basis": IDENTITY.tolist()}
    vertices, faces = export_shape.unit_mesh(triangles, bbox, np.zeros(3), 1.0)
    path = tmp_path / "objects" / "house" / "room" / "Bed_a_0.ply"
    export_shape.write_ply(path, vertices, faces)
    points = load_shape_points(str(path), num_points=256, rng=__import__("random").Random(0))
    assert len(points) == 256
    assert all(-0.6 <= value <= 0.6 for point in points for value in point)
