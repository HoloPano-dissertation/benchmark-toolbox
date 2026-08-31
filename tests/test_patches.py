import tempfile
import unittest
from pathlib import Path

from benchmark_toolbox.environments.patches import apply_patches


class PatchesTest(unittest.TestCase):
    def _env_file(self, repo: Path) -> Path:
        path = repo / "environment.yml"
        path.write_text(
            "\n".join(
                [
                    "dependencies:",
                    "  - python=3.8",
                    "  - pytorch=1.1.0=py3.6_cuda9.0.176_cudnn7.5.1_0",
                    "  - cudatoolkit=9.0",
                    "  - numpy",
                ]
            ),
            encoding="utf-8",
        )
        return path

    def test_depin_drop_and_relax(self) -> None:
        with tempfile.TemporaryDirectory() as repo:
            env = self._env_file(Path(repo))
            apply_patches(
                [
                    {
                        "file": "environment.yml",
                        "drop": ["cudatoolkit=9.0"],
                        "relax": ["pytorch=1.1.0"],
                    }
                ],
                Path(repo),
            )
            text = env.read_text(encoding="utf-8")
            self.assertNotIn("cudatoolkit", text)
            self.assertNotIn("cuda9.0.176", text)
            self.assertIn("pytorch=1.1.0", text)
            self.assertIn("numpy", text)

    def test_depin_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as repo:
            env = self._env_file(Path(repo))
            patch = [
                {
                    "file": "environment.yml",
                    "drop": ["cudatoolkit=9.0"],
                    "relax": ["pytorch=1.1.0"],
                }
            ]
            apply_patches(patch, Path(repo))
            once = env.read_text(encoding="utf-8")
            apply_patches(patch, Path(repo))
            self.assertEqual(once, env.read_text(encoding="utf-8"))

    def test_prepend_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as repo:
            main = Path(repo) / "main.py"
            main.write_text("print('x')\n", encoding="utf-8")
            patch = [{"file": "main.py", "prepend": "import cpu_patch\n"}]
            apply_patches(patch, Path(repo))
            apply_patches(patch, Path(repo))
            self.assertEqual(main.read_text(encoding="utf-8").count("import cpu_patch"), 1)

    def test_insert_after_lands_under_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as repo:
            env = self._env_file(Path(repo))
            patch = [
                {
                    "file": "environment.yml",
                    "after": "dependencies:",
                    "lines": ["  - setuptools<58", "  - wheel<0.38"],
                }
            ]
            apply_patches(patch, Path(repo))
            lines = env.read_text(encoding="utf-8").splitlines()
            # The pins must sit right under dependencies:, i.e. be conda deps,
            # not dangle outside the section.
            anchor = lines.index("dependencies:")
            self.assertEqual(lines[anchor + 1], "  - setuptools<58")
            self.assertEqual(lines[anchor + 2], "  - wheel<0.38")

    def test_insert_after_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as repo:
            env = self._env_file(Path(repo))
            patch = [
                {
                    "file": "environment.yml",
                    "after": "dependencies:",
                    "lines": ["  - setuptools<58"],
                }
            ]
            apply_patches(patch, Path(repo))
            once = env.read_text(encoding="utf-8")
            apply_patches(patch, Path(repo))
            self.assertEqual(once, env.read_text(encoding="utf-8"))
            self.assertEqual(once.count("setuptools<58"), 1)


if __name__ == "__main__":
    unittest.main()
