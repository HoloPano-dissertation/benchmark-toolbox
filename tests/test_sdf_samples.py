import importlib.util
import struct
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

MODULE_PATH = (Path(__file__).resolve().parents[1] / "tools" / "3dfront_dataset"
               / "_lib" / "sdf_samples.py")
spec = importlib.util.spec_from_file_location("sdf_samples", MODULE_PATH)
sdf_samples = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sdf_samples)

RESOLUTION = 48


def cube_triangles(half):
    corners = np.array([[x, y, z] for x in (-half, half)
                        for y in (-half, half) for z in (-half, half)], dtype=float)
    faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
             (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    return np.array([[corners[a], corners[b], corners[c]] for a, b, c in faces])


@pytest.fixture(scope="module")
def cube_field():
    triangles = cube_triangles(0.25)
    solid, shell = sdf_samples.occupancy(triangles, resolution=RESOLUTION)
    return triangles, solid, sdf_samples.signed_field(solid, shell)


def test_interior_is_filled_and_matches_the_cube_volume(cube_field):
    _, solid, _ = cube_field
    voxel = 2.0 * sdf_samples.HALF_EXTENT / RESOLUTION
    assert solid.sum() * voxel ** 3 == pytest.approx(0.5 ** 3, rel=0.3)


def test_field_is_negative_inside_and_positive_outside(cube_field):
    _, _, field = cube_field
    assert sdf_samples.sample_field(field, [[0.0, 0.0, 0.0]])[0] < 0
    assert sdf_samples.sample_field(field, [[0.6, 0.6, 0.6]])[0] > 0


def test_distance_at_a_known_point_matches_the_cube(cube_field):
    _, _, field = cube_field
    value = sdf_samples.sample_field(field, [[0.45, 0.0, 0.0]])[0]
    assert value == pytest.approx(0.2, abs=0.05)


def test_near_surface_samples_carry_only_the_sign(cube_field):
    triangles, _, field = cube_field
    points, values = sdf_samples.near_surface_samples(triangles, field, count=2000)
    assert set(np.unique(values)) <= {-1.0, 1.0}
    assert 0.2 < (values > 0).mean() < 0.8
    assert np.abs(points).max() <= sdf_samples.HALF_EXTENT + 1e-6


def test_uniform_samples_are_mostly_outside_a_small_object(cube_field):
    _, _, field = cube_field
    points, values = sdf_samples.uniform_samples(field, count=4000)
    assert (values > 0).mean() > 0.8
    assert np.abs(points).max() <= sdf_samples.HALF_EXTENT


def test_grid_and_matrix_round_trip_through_the_reader(tmp_path, cube_field):
    _, _, field = cube_field
    grid = sdf_samples.coarse_grid(field)
    matrix = sdf_samples.world_to_grid()
    path = tmp_path / "coarse_grid.grd"
    sdf_samples.write_grd(path, matrix, grid)
    raw = path.read_bytes()
    resolution = struct.unpack("iii", raw[:12])
    read_matrix = np.array(struct.unpack("f" * 16, raw[12:76])).reshape(4, 4)
    count = resolution[0] * resolution[1] * resolution[2]
    read_grid = np.array(struct.unpack("f" * count, raw[76:76 + 4 * count])).reshape(resolution)
    assert resolution == (32, 32, 32)
    assert read_matrix == pytest.approx(matrix)
    assert read_grid == pytest.approx(grid, abs=1e-6)


def test_world_to_grid_maps_the_cube_corners_onto_the_grid():
    matrix = sdf_samples.world_to_grid()
    low = matrix[:3, :3] @ np.array([-0.7, -0.7, -0.7]) + matrix[:3, 3]
    high = matrix[:3, :3] @ np.array([0.7, 0.7, 0.7]) + matrix[:3, 3]
    assert low == pytest.approx([0.0, 0.0, 0.0], abs=1e-5)
    assert high == pytest.approx([31.0, 31.0, 31.0], abs=1e-5)


def test_samples_are_written_as_four_floats_per_point(tmp_path, cube_field):
    triangles, _, field = cube_field
    points, values = sdf_samples.near_surface_samples(triangles, field, count=128)
    path = tmp_path / "nss_points.sdf"
    sdf_samples.write_samples(path, points, values)
    read = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    assert len(read) == 128
    assert read[:, :3] == pytest.approx(points, abs=1e-6)


def test_matrix_file_holds_sixteen_values(tmp_path):
    path = tmp_path / "orig_to_gaps.txt"
    sdf_samples.write_matrix(path, np.eye(4))
    assert len(path.read_text().split()) == 16


def _export_sdf():
    path = (Path(__file__).resolve().parents[1] / "tools" / "3dfront_dataset"
            / "_lib" / "export_sdf.py")
    loader = importlib.util.spec_from_file_location("export_sdf", path)
    import sys
    sys.path.insert(0, str(path.parent))
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def test_object_transform_maps_the_source_mesh_into_the_sample_frame():
    export_sdf = _export_sdf()
    camera = np.array([0.5, -1.0, 1.4])
    scale, size = 2.0, np.array([2.0, 4.0, 2.0])
    bbox = {"center": [3.0, 4.0, 0.0], "size": size.tolist(),
            "basis": np.eye(3).tolist()}
    matrix = export_sdf.object_transform(bbox, camera, scale)
    corner_room = np.array([3.0, 4.0, 1.0])
    mapped = matrix[:3, :3] @ corner_room + matrix[:3, 3]
    expected = export_sdf.LDIF_MESH_SCALE * (corner_room * scale - camera
                                             - np.array([3.0, 4.0, 0.0])) / size
    assert mapped == pytest.approx(expected)


def test_object_transform_keeps_the_mesh_inside_the_sampled_cube():
    export_sdf = _export_sdf()
    bbox = {"center": [0.0, 0.0, 0.0], "size": [1.0, 1.0, 1.0],
            "basis": np.eye(3).tolist()}
    matrix = export_sdf.object_transform(bbox, np.zeros(3), 1.0)
    corners = np.array([[x, y, z] for x in (-0.5, 0.5)
                        for y in (-0.5, 0.5) for z in (-0.5, 0.5)])
    mapped = corners @ matrix[:3, :3].T + matrix[:3, 3]
    assert np.abs(mapped).max() <= sdf_samples.HALF_EXTENT


def cracked_box_triangles(half, gap):
    triangles = cube_triangles(half).copy()
    top = np.all(np.abs(triangles[:, :, 2] - half) < 1e-9, axis=1)
    triangles[top, :, 2] += gap
    return triangles


def test_a_cracked_mesh_leaks_until_it_is_sealed():
    voxel = 2.0 * sdf_samples.HALF_EXTENT / RESOLUTION
    triangles = cracked_box_triangles(0.25, voxel * 1.5)
    leaking, shell = sdf_samples.occupancy(triangles, resolution=RESOLUTION, seal=0)
    solid, _ = sdf_samples.occupancy(triangles, resolution=RESOLUTION)
    assert leaking.sum() - shell.sum() < shell.sum() * 0.2
    assert solid.sum() > shell.sum() * 1.5


def test_a_sealed_mesh_has_an_interior_and_balanced_surface_samples():
    voxel = 2.0 * sdf_samples.HALF_EXTENT / RESOLUTION
    triangles = cracked_box_triangles(0.25, voxel * 1.5)
    solid, shell = sdf_samples.occupancy(triangles, resolution=RESOLUTION)
    field = sdf_samples.signed_field(solid, shell)
    assert sdf_samples.sample_field(field, [[0.0, 0.0, 0.0]])[0] < 0
    _, values = sdf_samples.near_surface_samples(triangles, field, count=2000)
    assert 0.2 < (values > 0).mean() < 0.8
