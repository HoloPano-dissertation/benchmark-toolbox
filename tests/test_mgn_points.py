import importlib.util
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")


def needs_scipy():
    pytest.importorskip("scipy")

DATASET = Path(__file__).resolve().parents[1] / "tools" / "3dfront_dataset"


def load(name):
    sys.path.insert(0, str(DATASET / "_lib"))
    spec = importlib.util.spec_from_file_location(name, DATASET / "_lib" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def exporter():
    return load("export_mgn_points")


@pytest.fixture(scope="module")
def shape():
    return load("export_shape")


def cube(folder, shape):
    folder.mkdir(parents=True, exist_ok=True)
    corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float) - 0.5
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                      [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                      [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3]])
    shape.write_ply(folder / "mesh_watertight.ply", corners, faces)
    return folder


def test_a_mesh_survives_the_round_trip(exporter, shape, tmp_path):
    folder = cube(tmp_path / "objects" / "bed" / "round", shape)
    vertices, faces = exporter.read_ply_mesh(folder / "mesh_watertight.ply")
    assert vertices.shape == (8, 3)
    assert faces.shape == (12, 3)
    assert np.isclose(np.abs(vertices).max(), 0.5)


def test_points_land_on_the_surface_and_carry_a_density(exporter, shape, tmp_path):
    needs_scipy()
    folder = cube(tmp_path / "objects" / "bed" / "one", shape)
    assert exporter.write_for(folder, 500) is None
    points = np.fromfile(folder / "gt_3dpoints.mgn", dtype=np.float64).reshape(-1, 3)
    densities = np.fromfile(folder / "densities.mgn", dtype=np.float64)
    assert points.shape == (500, 3) and densities.shape == (500,)
    assert np.isclose(np.abs(points).max(axis=1), 0.5, atol=1e-9).all()
    assert (densities > 0).all()


def test_every_side_of_the_cube_gets_points(exporter, shape, tmp_path):
    needs_scipy()
    folder = cube(tmp_path / "objects" / "bed" / "sides", shape)
    exporter.write_for(folder, 3000)
    points = np.fromfile(folder / "gt_3dpoints.mgn", dtype=np.float64).reshape(-1, 3)
    for axis in range(3):
        for sign in (-0.5, 0.5):
            assert np.isclose(points[:, axis], sign, atol=1e-9).sum() > 100


def test_the_same_object_gives_the_same_points_twice(exporter, shape, tmp_path):
    needs_scipy()
    folder = cube(tmp_path / "objects" / "bed" / "three", shape)
    exporter.write_for(folder, 200)
    first = (folder / "gt_3dpoints.mgn").read_bytes()
    (folder / "gt_3dpoints.mgn").unlink()
    exporter.write_for(folder, 200)
    assert (folder / "gt_3dpoints.mgn").read_bytes() == first


def test_an_object_without_a_mesh_is_named_not_skipped(exporter, tmp_path):
    folder = tmp_path / "objects" / "bed" / "empty"
    folder.mkdir(parents=True)
    assert exporter.write_for(folder, 100) == "no watertight mesh"


def test_sampling_is_spread_by_area_not_by_face_count(exporter, shape, tmp_path):
    """A long thin face must not win the same share as a large one."""
    folder = tmp_path / "objects" / "bed" / "area"
    folder.mkdir(parents=True)
    corners = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0],
                        [0, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6]])
    shape.write_ply(folder / "mesh_watertight.ply", corners, faces)
    vertices, loaded = exporter.read_ply_mesh(folder / "mesh_watertight.ply")
    points = exporter.sample_surface(vertices, loaded, 4000, seed=1)
    on_small = np.isclose(points[:, 2], 1.0).sum()
    assert 0 < on_small < 100
