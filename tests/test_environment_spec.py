import unittest
from pathlib import Path

from benchmark_toolbox.environments import load_environment_spec

REPO = Path(__file__).resolve().parents[1]


class EnvironmentSpecTest(unittest.TestCase):
    def test_load_echo_spec_from_file(self) -> None:
        spec = load_environment_spec(REPO / "configs/environments/echo.yaml")
        self.assertEqual(spec.name, "echo")
        self.assertEqual(spec.backend, "venv")
        self.assertEqual(spec.runner.entry, "runners/echo_runner.py")
        self.assertEqual(spec.variant, "cpu")
        self.assertIsNone(spec.repo)

    def test_load_dpc_spec_real_fields(self) -> None:
        spec = load_environment_spec(REPO / "configs/environments/dpc.yaml")
        self.assertEqual(spec.backend, "conda")
        self.assertEqual(spec.name, "Pano3D")
        self.assertIsNotNone(spec.repo)
        self.assertIn("DeepPanoContext", spec.repo.url)
        self.assertTrue(spec.checkpoints)
        self.assertEqual(spec.runner.wrapper[0], "xvfb-run")
        self.assertEqual(spec.runner.args[:2], ("--config", "configs/pano3d_igibson.yaml"))

    def test_inline_dict_requires_runner_entry(self) -> None:
        with self.assertRaises(ValueError):
            load_environment_spec({"name": "x", "backend": "venv"}, base_dir=REPO)

    def test_fingerprint_is_stable_and_variant_sensitive(self) -> None:
        base = {
            "name": "x",
            "backend": "conda",
            "python": "3.8",
            "runner": {"entry": "r.py"},
        }
        cpu = load_environment_spec({**base, "variant": "cpu"}, base_dir=REPO)
        cpu_again = load_environment_spec({**base, "variant": "cpu"}, base_dir=REPO)
        gpu = load_environment_spec({**base, "variant": "gpu"}, base_dir=REPO)
        self.assertEqual(cpu.fingerprint(), cpu_again.fingerprint())
        self.assertNotEqual(cpu.fingerprint(), gpu.fingerprint())


if __name__ == "__main__":
    unittest.main()
