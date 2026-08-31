import math
import random
import struct
import tempfile
import unittest
from pathlib import Path

from benchmark_toolbox.domain import BoundingBox3D, SceneObject, SceneOutput
from benchmark_toolbox.metrics.base import BaseMetric
from benchmark_toolbox.metrics.shape import (
    MeshChamfer,
    MeshFScore,
    _build_kd,
    _nearest_sq,
    chamfer_and_fscore,
    load_shape_points,
    write_f32_points,
)


def _nearest_sq_brute(points, query):
    qx, qy, qz = query
    best = math.inf
    for px, py, pz in points:
        dx, dy, dz = qx - px, qy - py, qz - pz
        distance = dx * dx + dy * dy + dz * dz
        if distance < best:
            best = distance
    return best


def _shape_object(object_id, label, minimum, maximum, shape, score=1.0):
    return SceneObject(
        object_id=object_id,
        label=label,
        score=score,
        bbox=BoundingBox3D(minimum, maximum),
        attributes={"shape": shape},
    )


class KDTreeTest(unittest.TestCase):
    def test_matches_brute_force(self) -> None:
        rng = random.Random(7)
        cloud = [
            (rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-5, 5))
            for _ in range(200)
        ]
        tree = _build_kd(list(cloud))
        for _ in range(80):
            query = (rng.uniform(-6, 6), rng.uniform(-6, 6), rng.uniform(-6, 6))
            self.assertAlmostEqual(
                _nearest_sq(tree, query), _nearest_sq_brute(cloud, query), places=10
            )

    def test_finds_exact_member(self) -> None:
        cloud = [(0.0, 0.0, 0.0), (1.0, 2.0, 3.0), (-4.0, 5.0, 6.0)]
        tree = _build_kd(list(cloud))
        self.assertEqual(_nearest_sq(tree, (1.0, 2.0, 3.0)), 0.0)


class ChamferFScoreTest(unittest.TestCase):
    def test_identical_clouds_are_perfect(self) -> None:
        cloud = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        chamfer, fscore = chamfer_and_fscore(cloud, cloud, fscore_threshold=0.01)
        self.assertEqual(chamfer, 0.0)
        self.assertEqual(fscore, 1.0)

    def test_single_point_offset_squared(self) -> None:
        chamfer, _ = chamfer_and_fscore([(0.3, 0.0, 0.0)], [(0.0, 0.0, 0.0)])
        self.assertAlmostEqual(chamfer, 2.0 * 0.3 * 0.3)

    def test_unsquared_is_euclidean(self) -> None:
        chamfer, _ = chamfer_and_fscore(
            [(0.3, 0.0, 0.0)], [(0.0, 0.0, 0.0)], squared=False
        )
        self.assertAlmostEqual(chamfer, 2.0 * 0.3)

    def test_fscore_threshold_gates_match(self) -> None:
        _, near = chamfer_and_fscore(
            [(0.3, 0.0, 0.0)], [(0.0, 0.0, 0.0)], fscore_threshold=0.5
        )
        _, far = chamfer_and_fscore(
            [(0.3, 0.0, 0.0)], [(0.0, 0.0, 0.0)], fscore_threshold=0.1
        )
        self.assertEqual(near, 1.0)
        self.assertEqual(far, 0.0)

    def test_empty_cloud_is_nan(self) -> None:
        chamfer, fscore = chamfer_and_fscore([], [(0.0, 0.0, 0.0)])
        self.assertTrue(math.isnan(chamfer))
        self.assertTrue(math.isnan(fscore))


class LoadShapePointsTest(unittest.TestCase):
    def test_inline_list(self) -> None:
        points = load_shape_points(
            [[0, 0, 0], [1, 2, 3]], num_points=4, rng=random.Random(0)
        )
        self.assertEqual(points, [(0.0, 0.0, 0.0), (1.0, 2.0, 3.0)])

    def test_dict_points(self) -> None:
        points = load_shape_points(
            {"points": [[1, 1, 1]]}, num_points=4, rng=random.Random(0)
        )
        self.assertEqual(points, [(1.0, 1.0, 1.0)])

    def test_f32_roundtrip(self) -> None:
        cloud = [(0.5, -1.25, 2.0), (3.0, 4.0, 5.0)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pts.f32"
            write_f32_points(path, cloud)
            loaded = load_shape_points(str(path), num_points=4, rng=random.Random(0))
        for got, want in zip(loaded, cloud):
            for a, b in zip(got, want):
                self.assertAlmostEqual(a, b, places=5)

    def test_json_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pts.json"
            path.write_text("[[0,0,0],[2,2,2]]", encoding="utf-8")
            loaded = load_shape_points(str(path), num_points=4, rng=random.Random(0))
        self.assertEqual(loaded, [(0.0, 0.0, 0.0), (2.0, 2.0, 2.0)])

    def test_obj_sampling_on_surface(self) -> None:
        obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3\nf 1 3 4\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quad.obj"
            path.write_text(obj, encoding="utf-8")
            points = load_shape_points(str(path), num_points=200, rng=random.Random(1))
        self.assertEqual(len(points), 200)
        for x, y, z in points:
            self.assertAlmostEqual(z, 0.0, places=6)
            self.assertTrue(-1e-6 <= x <= 1 + 1e-6 and -1e-6 <= y <= 1 + 1e-6)

    def test_sampling_is_deterministic_with_seed(self) -> None:
        obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3\nf 1 3 4\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quad.obj"
            path.write_text(obj, encoding="utf-8")
            first = load_shape_points(str(path), num_points=50, rng=random.Random(3))
            second = load_shape_points(str(path), num_points=50, rng=random.Random(3))
        self.assertEqual(first, second)

    def test_ply_ascii_sampling(self) -> None:
        ply = (
            "ply\nformat ascii 1.0\n"
            "element vertex 4\nproperty float x\nproperty float y\nproperty float z\n"
            "element face 2\nproperty list uchar int vertex_indices\nend_header\n"
            "0 0 0\n1 0 0\n1 1 0\n0 1 0\n3 0 1 2\n3 0 2 3\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quad.ply"
            path.write_text(ply, encoding="ascii")
            points = load_shape_points(str(path), num_points=100, rng=random.Random(2))
        self.assertEqual(len(points), 100)
        for _x, _y, z in points:
            self.assertAlmostEqual(z, 0.0, places=6)

    def test_ply_binary_little_endian(self) -> None:
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            "element vertex 4\nproperty float x\nproperty float y\nproperty float z\n"
            "element face 2\nproperty list uchar int vertex_indices\nend_header\n"
        ).encode("ascii")
        vertices = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        body = b"".join(struct.pack("<fff", *v) for v in vertices)
        body += struct.pack("<Biii", 3, 0, 1, 2) + struct.pack("<Biii", 3, 0, 2, 3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quad_bin.ply"
            path.write_bytes(header + body)
            points = load_shape_points(str(path), num_points=100, rng=random.Random(4))
        self.assertEqual(len(points), 100)
        for x, y, z in points:
            self.assertAlmostEqual(z, 0.0, places=6)
            self.assertTrue(-1e-6 <= x <= 1 + 1e-6 and -1e-6 <= y <= 1 + 1e-6)


class MeshChamferMetricTest(unittest.TestCase):
    def test_registered_in_registry(self) -> None:
        self.assertIsInstance(BaseMetric.create("mesh_chamfer"), MeshChamfer)
        self.assertIsInstance(BaseMetric.create("mesh_fscore"), MeshFScore)

    def test_perfect_shape_is_zero(self) -> None:
        cloud = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        obj = lambda oid: _shape_object(oid, "chair", (0, 0, 0), (1, 1, 1), cloud)
        prediction = SceneOutput(layout=None, objects=(obj("p"),))
        ground_truth = SceneOutput(layout=None, objects=(obj("g"),))
        self.assertEqual(MeshChamfer().compute(prediction, ground_truth), 0.0)
        self.assertEqual(MeshFScore({"fscore_threshold": 0.1}).compute(
            prediction, ground_truth
        ), 1.0)

    def test_averages_only_matched_pairs(self) -> None:
        matched_pred = _shape_object("pa", "chair", (0, 0, 0), (1, 1, 1), [[0, 0, 0]])
        matched_gt = _shape_object("ga", "chair", (0, 0, 0), (1, 1, 1), [[0, 0, 0]])
        unmatched_pred = _shape_object(
            "pb", "table", (50, 50, 50), (51, 51, 51), [[9, 9, 9]]
        )
        unmatched_gt = _shape_object(
            "gb", "table", (0, 0, 0), (1, 1, 1), [[0, 0, 0]]
        )
        prediction = SceneOutput(layout=None, objects=(matched_pred, unmatched_pred))
        ground_truth = SceneOutput(layout=None, objects=(matched_gt, unmatched_gt))
        self.assertEqual(
            MeshChamfer({"iou_threshold": 0.15}).compute(prediction, ground_truth), 0.0
        )

    def test_no_match_returns_nan(self) -> None:
        prediction = SceneOutput(
            layout=None,
            objects=(_shape_object("p", "chair", (0, 0, 0), (1, 1, 1), [[0, 0, 0]]),),
        )
        ground_truth = SceneOutput(
            layout=None,
            objects=(
                _shape_object("g", "chair", (90, 90, 90), (91, 91, 91), [[0, 0, 0]]),
            ),
        )
        self.assertTrue(math.isnan(MeshChamfer().compute(prediction, ground_truth)))

    def test_pair_without_shape_is_skipped(self) -> None:
        prediction = SceneOutput(
            layout=None,
            objects=(_shape_object("p", "chair", (0, 0, 0), (1, 1, 1), [[0, 0, 0]]),),
        )
        gt_no_shape = SceneObject("g", "chair", BoundingBox3D((0, 0, 0), (1, 1, 1)))
        ground_truth = SceneOutput(layout=None, objects=(gt_no_shape,))
        self.assertTrue(math.isnan(MeshChamfer().compute(prediction, ground_truth)))

    def test_normalize_unit_scales_by_gt_diagonal(self) -> None:
        pred = _shape_object("p", "x", (0, 0, 0), (1, 1, 1), [[1, 0, 0], [3, 2, 2]])
        gt = _shape_object("g", "x", (0, 0, 0), (1, 1, 1), [[0, 0, 0], [2, 2, 2]])
        prediction = SceneOutput(layout=None, objects=(pred,))
        ground_truth = SceneOutput(layout=None, objects=(gt,))
        value = MeshChamfer({"normalize": "unit"}).compute(prediction, ground_truth)
        diag = math.sqrt(12.0)
        self.assertAlmostEqual(value, 2.0 * (1.0 / diag) ** 2)

    def test_normalize_canonical_is_scale_invariant(self) -> None:
        unit = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]
        big = [[3 * c + 5 for c in p] for p in unit]
        pred = _shape_object("p", "x", (0, 0, 0), (1, 1, 1), big)
        gt = _shape_object("g", "x", (0, 0, 0), (1, 1, 1), unit)
        prediction = SceneOutput(layout=None, objects=(pred,))
        ground_truth = SceneOutput(layout=None, objects=(gt,))
        value = MeshChamfer({"normalize": "canonical"}).compute(prediction, ground_truth)
        self.assertAlmostEqual(value, 0.0, places=6)
        raw = MeshChamfer({"normalize": "none"}).compute(prediction, ground_truth)
        self.assertGreater(raw, 1.0)

    def test_requires_ground_truth(self) -> None:
        prediction = SceneOutput(layout=None, objects=())
        with self.assertRaises(ValueError):
            MeshChamfer().compute(prediction, None)


if __name__ == "__main__":
    unittest.main()
