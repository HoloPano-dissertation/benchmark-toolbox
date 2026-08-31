"""Component lookup by config name, and path resolution of config parameters."""

import json
import tempfile
import unittest
from pathlib import Path

from benchmark_toolbox.config import load_experiment_config
from benchmark_toolbox.datasets import BaseDatasetLoader, ManifestDatasetLoader
from benchmark_toolbox.models import BaseSceneEstimator, SubprocessSceneEstimator
from benchmark_toolbox.registry import ComponentRegistry


class RegistryAliasTest(unittest.TestCase):
    def test_dataset_and_model_aliases_resolve_to_one_implementation(self) -> None:
        for name in ("manifest", "igibson"):
            loader = BaseDatasetLoader.create(
                name, {"manifest": str(Path(__file__))}
            )
            self.assertIsInstance(loader, ManifestDatasetLoader)
        for name in ("subprocess", "dpc", "holopano"):
            estimator = BaseSceneEstimator.create(name, {"command": ["true"]})
            self.assertIsInstance(estimator, SubprocessSceneEstimator)

    def test_alias_requires_a_registered_target(self) -> None:
        registry: ComponentRegistry[object] = ComponentRegistry("widget")
        with self.assertRaises(ValueError):
            registry.alias("missing", "other")

    def test_alias_does_not_overwrite_an_existing_name(self) -> None:
        registry: ComponentRegistry[object] = ComponentRegistry("widget")
        registry.register("first")(lambda parameters: "first")
        registry.register("second")(lambda parameters: "second")
        with self.assertRaises(ValueError):
            registry.alias("first", "second")


class ConfigPathResolutionTest(unittest.TestCase):
    def test_relative_paths_resolve_against_the_config_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            (base / "manifest.jsonl").write_text("", encoding="utf-8")
            config_path = base / "experiment.json"
            config_path.write_text(
                json.dumps(
                    {
                        "experiment_name": "t",
                        "model": {
                            "type": "fixture",
                            "parameters": {"prediction_dir": "pred"},
                        },
                        "dataset": {
                            "type": "manifest",
                            "parameters": {"manifest": "manifest.jsonl"},
                        },
                        "metrics": [
                            {
                                "type": "mesh_chamfer",
                                "parameters": {"mesh_root": "meshes"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            config = load_experiment_config(config_path)

        self.assertEqual(
            config.metrics[0].parameters["mesh_root"], str(base / "meshes")
        )
        self.assertEqual(
            config.model.parameters["prediction_dir"], str(base / "pred")
        )


class ProtocolInheritanceTest(unittest.TestCase):
    def _write(self, path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_metrics_are_inherited_and_locally_overridable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            self._write(
                base / "protocol.json",
                {
                    "seed": 7,
                    "metrics": [{"type": "collision_rate"}],
                    "model": {"type": "fixture", "parameters": {"prediction_dir": "p"}},
                    "dataset": {"type": "manifest", "parameters": {"manifest": "m"}},
                },
            )
            self._write(
                base / "method.json",
                {
                    "extends": "protocol.json",
                    "experiment_name": "method",
                    "dataset": {
                        "type": "manifest",
                        "parameters": {"manifest": "other.jsonl"},
                    },
                },
            )
            config = load_experiment_config(base / "method.json")

        self.assertEqual(config.seed, 7)
        self.assertEqual(config.experiment_name, "method")
        self.assertEqual([m.type for m in config.metrics], ["collision_rate"])
        self.assertEqual(
            config.dataset.parameters["manifest"], str(base / "other.jsonl")
        )
        self.assertEqual(config.model.parameters["prediction_dir"], str(base / "p"))

    def test_per_class_expands_into_one_row_per_class(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            self._write(
                base / "c.json",
                {
                    "model": {"type": "fixture", "parameters": {}},
                    "dataset": {"type": "manifest", "parameters": {}},
                    "metrics": [
                        {
                            "type": "object_map_dataset",
                            "parameters": {
                                "iou_threshold": 0.15,
                                "classes": ["chair", "sofa"],
                                "per_class": True,
                                "name": "ap",
                            },
                        }
                    ],
                },
            )
            config = load_experiment_config(base / "c.json")

        self.assertEqual(
            [(m.parameters["name"], m.parameters["classes"]) for m in config.metrics],
            [("ap_chair", ["chair"]), ("ap_sofa", ["sofa"])],
        )
        self.assertTrue(all(m.parameters["iou_threshold"] == 0.15 for m in config.metrics))

    def test_cyclic_extends_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            self._write(base / "a.json", {"extends": "b.json"})
            self._write(base / "b.json", {"extends": "a.json"})
            with self.assertRaisesRegex(ValueError, "Cyclic"):
                load_experiment_config(base / "a.json")


class ShippedConfigsTest(unittest.TestCase):
    def test_every_example_config_loads(self) -> None:
        configs = sorted((Path(__file__).resolve().parent.parent / "configs" / "examples").iterdir())
        self.assertTrue(configs)
        for path in configs:
            with self.subTest(config=path.name):
                config = load_experiment_config(path)
                self.assertTrue(config.metrics)


if __name__ == "__main__":
    unittest.main()
