import struct

import numpy as np
from scipy import ndimage

HALF_EXTENT = 0.7
GRID_RESOLUTION = 32
FIELD_RESOLUTION = 128
SAMPLE_COUNT = 100000
NEAR_SURFACE_SIGMA = 0.02
SEAL_ITERATIONS = 1


def world_to_grid(resolution=GRID_RESOLUTION, half_extent=HALF_EXTENT):
    scale = (resolution - 1) / (2.0 * half_extent)
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 0] = matrix[1, 1] = matrix[2, 2] = scale
    matrix[0, 3] = matrix[1, 3] = matrix[2, 3] = (resolution - 1) / 2.0
    return matrix


def triangle_samples(triangles, spacing):
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    edge_one, edge_two = b - a, c - a
    area = 0.5 * np.linalg.norm(np.cross(edge_one, edge_two), axis=1)
    steps = np.maximum(2, np.ceil(np.sqrt(area * 4.0) / spacing).astype(int) + 1)
    points = [triangles.reshape(-1, 3)]
    for count in np.unique(steps):
        chosen = steps == count
        grid = [(i / count, j / count) for i in range(count + 1)
                for j in range(count + 1 - i)]
        weights = np.asarray(grid, dtype=float)
        points.append((a[chosen][:, None, :]
                       + weights[None, :, 0, None] * edge_one[chosen][:, None, :]
                       + weights[None, :, 1, None] * edge_two[chosen][:, None, :]
                       ).reshape(-1, 3))
    return np.concatenate(points, axis=0)


def occupancy(triangles, resolution=FIELD_RESOLUTION, half_extent=HALF_EXTENT,
              seal=SEAL_ITERATIONS):
    voxel = 2.0 * half_extent / resolution
    points = triangle_samples(triangles, voxel * 0.5)
    index = np.floor((points + half_extent) / voxel).astype(int)
    index = index[np.all((index >= 0) & (index < resolution), axis=1)]
    shell = np.zeros((resolution,) * 3, dtype=bool)
    if len(index):
        shell[index[:, 0], index[:, 1], index[:, 2]] = True
    sealed = shell
    if seal:
        sealed = ndimage.binary_closing(shell, np.ones((3, 3, 3)), iterations=seal)
    return ndimage.binary_fill_holes(sealed), shell


def signed_field(solid, shell, half_extent=HALF_EXTENT):
    resolution = solid.shape[0]
    voxel = 2.0 * half_extent / resolution
    distance = ndimage.distance_transform_edt(~shell) * voxel
    sign = np.where(solid, -1.0, 1.0)
    return (sign * distance).astype(np.float32)


def sample_field(field, points, half_extent=HALF_EXTENT):
    resolution = field.shape[0]
    coordinates = (np.asarray(points, dtype=float) + half_extent) \
        / (2.0 * half_extent) * (resolution - 1)
    return ndimage.map_coordinates(field, coordinates.T, order=1, mode="nearest")


def coarse_grid(field, resolution=GRID_RESOLUTION, half_extent=HALF_EXTENT):
    axis = np.linspace(-half_extent, half_extent, resolution)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    values = sample_field(field, points.reshape(-1, 3), half_extent)
    return values.reshape((resolution,) * 3).astype(np.float32)


def near_surface_samples(triangles, field, count=SAMPLE_COUNT,
                         sigma=NEAR_SURFACE_SIGMA, half_extent=HALF_EXTENT, seed=0):
    rng = np.random.default_rng(seed)
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total = area.sum()
    if total <= 0:
        raise ValueError("Mesh has no surface area")
    chosen = rng.choice(len(triangles), size=count, p=area / total)
    first, second = rng.random((2, count, 1))
    root = np.sqrt(first)
    points = (a[chosen] * (1 - root) + b[chosen] * root * (1 - second)
              + c[chosen] * root * second)
    points = points + rng.normal(scale=sigma, size=points.shape)
    points = np.clip(points, -half_extent, half_extent)
    values = np.sign(sample_field(field, points, half_extent)).astype(np.float32)
    values[values == 0] = 1.0
    return points.astype(np.float32), values


def uniform_samples(field, count=SAMPLE_COUNT, half_extent=HALF_EXTENT, seed=1):
    rng = np.random.default_rng(seed)
    points = rng.uniform(-half_extent, half_extent, size=(count, 3))
    values = sample_field(field, points, half_extent).astype(np.float32)
    return points.astype(np.float32), values


def write_samples(path, points, values):
    payload = np.concatenate(
        (np.asarray(points, dtype=np.float32),
         np.asarray(values, dtype=np.float32).reshape(-1, 1)), axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload.astype("<f4").tobytes())


def write_grd(path, matrix, grid):
    resolution = grid.shape
    payload = struct.pack("<3i", *resolution)
    payload += np.asarray(matrix, dtype="<f4").reshape(-1).tobytes()
    payload += np.asarray(grid, dtype="<f4").reshape(-1).tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def write_matrix(path, matrix):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(" ".join("%.6g" % value
                             for value in np.asarray(matrix, dtype=float).reshape(-1)),
                    encoding="utf-8")


def watertight_mesh(field, half_extent=HALF_EXTENT):
    from skimage import measure
    vertices, faces, _, _ = measure.marching_cubes(field, level=0.0)
    resolution = field.shape[0]
    vertices = vertices / (resolution - 1) * (2.0 * half_extent) - half_extent
    return vertices.astype(np.float32), faces.astype(np.int64)
