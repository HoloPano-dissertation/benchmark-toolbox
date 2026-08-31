import unittest
from pathlib import Path

from benchmark_toolbox.environments import generate_submit_script, load_environment_spec

REPO = Path(__file__).resolve().parents[1]


class SubmitScriptTest(unittest.TestCase):
    def test_slurm_script_for_gpu(self) -> None:
        spec = load_environment_spec(REPO / "configs/environments/dpc.yaml")
        script = generate_submit_script(
            "configs/environments/dpc.yaml",
            "configs/examples/dpc.yaml",
            spec=spec,
            scheduler="slurm",
            gpus=1,
        )
        self.assertIn("#!/bin/bash", script)
        self.assertIn("#SBATCH", script)
        self.assertIn("--gres=gpu:1", script)
        self.assertIn("module load anaconda", script)
        self.assertIn(
            "benchmark-toolbox env prepare --env configs/environments/dpc.yaml",
            script,
        )
        self.assertIn(
            "benchmark-toolbox run --config configs/examples/dpc.yaml", script
        )

    def test_bash_script_has_no_sbatch(self) -> None:
        spec = load_environment_spec(REPO / "configs/environments/echo.yaml")
        script = generate_submit_script(
            "configs/environments/echo.yaml",
            spec=spec,
            scheduler="bash",
            module_load="",
        )
        self.assertNotIn("#SBATCH", script)
        self.assertIn("benchmark-toolbox env prepare", script)


if __name__ == "__main__":
    unittest.main()
