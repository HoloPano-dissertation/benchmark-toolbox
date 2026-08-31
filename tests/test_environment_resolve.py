import os
import unittest
from pathlib import Path

from benchmark_toolbox.environments import load_environment_spec
from benchmark_toolbox.environments.conda import CondaEnvironmentManager
from benchmark_toolbox.environments.venv import VenvEnvironmentManager

REPO = Path(__file__).resolve().parents[1]


class ResolveTest(unittest.TestCase):
    def test_conda_resolve_builds_conda_run_prefix(self) -> None:
        spec = load_environment_spec(REPO / "configs/environments/dpc.yaml")
        handle = CondaEnvironmentManager().resolve(spec)
        self.assertIn("run", handle.command_prefix)
        self.assertIn("Pano3D", handle.command_prefix)
        self.assertEqual(handle.command_prefix[-1], "python")
        self.assertNotEqual(handle.env.get("CUDA_VISIBLE_DEVICES"), "")
        self.assertTrue(str(handle.cwd).endswith(os.path.join("repos", "Pano3D")))
        self.assertEqual(handle.wrapper[0], "xvfb-run")

    def test_venv_resolve_points_at_venv_python(self) -> None:
        spec = load_environment_spec(REPO / "configs/environments/echo.yaml")
        handle = VenvEnvironmentManager().resolve(spec)
        self.assertEqual(len(handle.command_prefix), 1)
        python = handle.command_prefix[0]
        self.assertTrue(
            python.endswith(os.path.join("venvs", "echo", "bin", "python"))
            or python.endswith(os.path.join("venvs", "echo", "Scripts", "python.exe"))
        )
        self.assertIsNone(handle.cwd)


if __name__ == "__main__":
    unittest.main()
