import os
import pickle
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

RUNNERS = Path(__file__).resolve().parents[1] / "runners"
sys.path.insert(0, str(RUNNERS))
import dpc_runner  # noqa: E402


class DpcConvertTest(unittest.TestCase):
    def test_convert_data_pkl(self) -> None:
        data = {
            "objs": [
                {
                    "classname": "chair",
                    "score": 0.9,
                    "bdb3d": {
                        "centroid": [0, 0, 0],
                        "basis": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        "size": [2, 2, 2],
                    },
                },
                {"classname": "no-3d-box"},
            ],
            "layout": {
                "manhattan_world": [
                    [2, 3, 4], [2, 3, -4], [2, -3, 4], [2, -3, -4],
                    [-2, 3, 4], [-2, 3, -4], [-2, -3, 4], [-2, -3, -4],
                ],
            },
        }
        scene = dpc_runner.convert_data_pkl(data)
        self.assertEqual(len(scene["objects"]), 1)
        self.assertEqual(scene["objects"][0]["label"], "chair")
        self.assertEqual(scene["objects"][0]["score"], 0.9)
        bbox = scene["objects"][0]["bbox"]
        self.assertEqual(bbox["center"], [0.0, 0.0, 0.0])
        self.assertEqual(bbox["size"], [2.0, 2.0, 2.0])
        self.assertEqual(bbox["basis"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertEqual(scene["layout"]["min_corner"], [-2.0, -3.0, -4.0])
        self.assertEqual(scene["layout"]["max_corner"], [2.0, 3.0, 4.0])

    def test_convert_attaches_mesh_path_as_shape(self) -> None:
        data = {
            "objs": [
                {
                    "classname": "chair",
                    "mesh_path": "/abs/pred/s1/obj-0.ply",
                    "bdb3d": {
                        "centroid": [0, 0, 0],
                        "basis": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        "size": [2, 2, 2],
                    },
                },
                {
                    "classname": "table",
                    "mesh": object(),
                    "bdb3d": {
                        "centroid": [5, 5, 5],
                        "basis": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        "size": [1, 1, 1],
                    },
                },
            ],
            "layout": None,
        }
        scene = dpc_runner.convert_data_pkl(data)
        self.assertEqual(
            scene["objects"][0]["attributes"]["shape"], "/abs/pred/s1/obj-0.ply"
        )
        self.assertNotIn("shape", scene["objects"][1]["attributes"])

    def test_find_data_pkl_matches_sample_id(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            for scene in ("scene-A", "scene-B"):
                pkl = Path(root) / "run1" / "visualization" / scene / "data.pkl"
                pkl.parent.mkdir(parents=True, exist_ok=True)
                pkl.write_bytes(b"")
            found_b = dpc_runner.find_data_pkl("scene-B", search_root=root)
            found_a = dpc_runner.find_data_pkl("scene-A", search_root=root)
            self.assertTrue(found_b.endswith("scene-B/data.pkl"))
            self.assertTrue(found_a.endswith("scene-A/data.pkl"))
            self.assertIsNone(dpc_runner.find_data_pkl("missing", search_root=root))


class LoadDataPklTest(unittest.TestCase):
    def test_unpickles_a_class_defined_in_the_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            repo = Path(root) / "repo"
            (repo / "utils").mkdir(parents=True)
            (repo / "utils" / "__init__.py").write_text("", encoding="utf-8")
            (repo / "utils" / "scene.py").write_text(
                textwrap.dedent(
                    """
                    class IGScene:
                        def __init__(self, data):
                            self.data = data
                    """
                ),
                encoding="utf-8",
            )
            pickle_path = Path(root) / "data.pkl"
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    textwrap.dedent(
                        f"""
                        import pickle, sys
                        sys.path.insert(0, {str(repo)!r})
                        from utils.scene import IGScene
                        with open({str(pickle_path)!r}, "wb") as fh:
                            pickle.dump(IGScene({{"objs": []}}), fh)
                        """
                    ),
                ],
                check=True,
            )

            previous_cwd = Path.cwd()
            previous_path = list(sys.path)
            previous_modules = set(sys.modules)
            try:
                os.chdir(repo)
                with self.assertRaises(ModuleNotFoundError):
                    with open(pickle_path, "rb") as handle:
                        pickle.load(handle)
                self.assertEqual(dpc_runner.load_data_pkl(str(pickle_path)), {"objs": []})
            finally:
                os.chdir(previous_cwd)
                sys.path[:] = previous_path
                for name in set(sys.modules) - previous_modules:
                    del sys.modules[name]


if __name__ == "__main__":
    unittest.main()
