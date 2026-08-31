import os
import tempfile
import unittest
from pathlib import Path

from benchmark_toolbox.environments.spec import CheckpointSpec
from benchmark_toolbox.environments.steps import download_checkpoint


class CheckpointStagingTest(unittest.TestCase):
    def _run(self, artifacts: Path, spec: CheckpointSpec, repo: Path):
        os.environ["BENCHMARK_TOOLBOX_ARTIFACTS"] = str(artifacts)
        try:
            return download_checkpoint(spec, repo)
        finally:
            os.environ.pop("BENCHMARK_TOOLBOX_ARTIFACTS", None)

    def test_uses_local_artifact_instead_of_url(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            (artifacts / "model.pth").write_bytes(b"weights")
            repo = root / "repo"
            repo.mkdir()
            spec = CheckpointSpec(
                url="https://invalid.example/never-fetched", dest="ckpt/model.pth"
            )
            out = self._run(artifacts, spec, repo)
            self.assertEqual(out.read_bytes(), b"weights")

    def test_local_artifact_checksum_is_verified(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            (artifacts / "model.pth").write_bytes(b"weights")
            repo = root / "repo"
            repo.mkdir()
            spec = CheckpointSpec(
                url="https://invalid.example/never-fetched",
                dest="ckpt/model.pth",
                sha256="0" * 64,  # wrong digest
            )
            with self.assertRaises(RuntimeError):
                self._run(artifacts, spec, repo)

    def test_no_artifacts_dir_means_no_local_staging(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            repo = Path(root) / "repo"
            repo.mkdir()
            spec = CheckpointSpec(
                url="https://invalid.example/never-fetched", dest="ckpt/model.pth"
            )
            os.environ.pop("BENCHMARK_TOOLBOX_ARTIFACTS", None)
            with self.assertRaises(Exception):
                download_checkpoint(spec, repo)


if __name__ == "__main__":
    unittest.main()
