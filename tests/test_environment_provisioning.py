import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from benchmark_toolbox.environments.base import BaseEnvironmentManager, EnvHandle
from benchmark_toolbox.environments.plan import (
    DONE,
    NETWORK,
    TODO,
    build_plan,
    render_plan,
    verify_checkpoints,
)
from benchmark_toolbox.environments.spec import CheckpointSpec, load_environment_spec
from benchmark_toolbox.environments.steps import checkpoint_path, download_checkpoint


@contextmanager
def environment(**values: "str | None"):
    previous = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class _FakeManager(BaseEnvironmentManager):
    def __init__(self, exists: bool = False) -> None:
        super().__init__()
        self.exists = exists

    def prepare(self, spec):
        raise AssertionError("prepare must not run in these tests")

    def resolve(self, spec):
        return EnvHandle()

    def _env_exists(self, spec) -> bool:
        return self.exists


def _spec(tmp: Path, **overrides):
    data = {
        "name": "demo",
        "backend": "venv",
        "workspace": str(tmp / "workspace"),
        "runner": {"entry": "runners/echo_runner.py"},
    }
    data.update(overrides)
    return load_environment_spec(data, base_dir=tmp)


class CheckpointDestinationTest(unittest.TestCase):
    def test_relative_destination_lands_in_the_repository(self) -> None:
        checkpoint = CheckpointSpec(url="https://example/x", dest="out/model.pth")
        self.assertEqual(
            checkpoint_path(checkpoint, Path("/repo")), Path("/repo/out/model.pth")
        )

    def test_absolute_and_home_destinations_are_taken_literally(self) -> None:
        backbone = CheckpointSpec(
            url="https://example/resnet.pth", dest="~/.cache/torch/hub/checkpoints/r.pth"
        )
        self.assertEqual(
            checkpoint_path(backbone, Path("/repo")),
            Path.home() / ".cache/torch/hub/checkpoints/r.pth",
        )
        absolute = CheckpointSpec(url="https://example/x", dest="/opt/weights/m.pth")
        self.assertEqual(checkpoint_path(absolute, Path("/repo")), Path("/opt/weights/m.pth"))

    def test_staged_file_reaches_a_destination_outside_the_repository(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            staging = root / "staged"
            staging.mkdir()
            (staging / "r.pth").write_bytes(b"backbone")
            target = root / "cache" / "r.pth"
            checkpoint = CheckpointSpec(
                url="https://invalid.example/never-fetched", dest=str(target)
            )
            with environment(BENCHMARK_TOOLBOX_ARTIFACTS=str(staging)):
                download_checkpoint(checkpoint, root / "repo")
            self.assertEqual(target.read_bytes(), b"backbone")


class OfflineModeTest(unittest.TestCase):
    def test_missing_checkpoint_fails_fast_and_says_what_to_stage(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            checkpoint = CheckpointSpec(
                url="https://example.invalid/weights.pth", dest="ckpt/weights.pth"
            )
            with environment(
                BENCHMARK_TOOLBOX_OFFLINE="1", BENCHMARK_TOOLBOX_ARTIFACTS=None
            ):
                with self.assertRaises(RuntimeError) as caught:
                    download_checkpoint(checkpoint, Path(root))
        message = str(caught.exception)
        self.assertIn("Offline mode", message)
        self.assertIn("https://example.invalid/weights.pth", message)
        self.assertIn("weights.pth", message)

    def test_staged_checkpoint_still_works_offline(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            staging = root / "staged"
            staging.mkdir()
            (staging / "weights.pth").write_bytes(b"w")
            checkpoint = CheckpointSpec(
                url="https://example.invalid/weights.pth", dest="ckpt/weights.pth"
            )
            with environment(
                BENCHMARK_TOOLBOX_OFFLINE="1",
                BENCHMARK_TOOLBOX_ARTIFACTS=str(staging),
            ):
                result = download_checkpoint(checkpoint, root)
            self.assertEqual(result.read_bytes(), b"w")


class PlanTest(unittest.TestCase):
    def test_plan_separates_what_is_done_from_what_needs_the_network(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            staging = root / "staged"
            staging.mkdir()
            (staging / "staged.pth").write_bytes(b"w")
            spec = _spec(
                root,
                repo={"url": "https://example/model.git"},
                pip=["torch==1.7.1"],
                checkpoints=[
                    {"url": "https://example/a.pth", "dest": "ckpt/staged.pth"},
                    {"url": "https://example/b.pth", "dest": "ckpt/missing.pth"},
                ],
                system=["xvfb"],
            )
            with environment(BENCHMARK_TOOLBOX_ARTIFACTS=str(staging)):
                steps = build_plan(spec, _FakeManager(exists=True))
                report = render_plan(spec, steps)

        by_summary = {step.summary: step for step in steps}
        self.assertEqual(by_summary["venv environment 'demo' exists"].state, DONE)
        self.assertEqual(by_summary["clone repository"].state, NETWORK)
        # Already staged locally: no network needed for this one, unlike the other.
        self.assertEqual(by_summary["checkpoint ckpt/staged.pth"].state, TODO)
        self.assertEqual(by_summary["checkpoint ckpt/missing.pth"].state, NETWORK)
        self.assertIn("step(s) will use the network", report)
        self.assertIn("xvfb", report)

    def test_plan_reports_a_fully_staged_build_as_offline_capable(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            spec = _spec(Path(root))
            steps = build_plan(spec, _FakeManager(exists=True))
            report = render_plan(spec, steps)
        self.assertIn("Nothing left to fetch", report)


class VerifyAndAdoptTest(unittest.TestCase):
    def test_verify_flags_a_corrupted_staged_file(self) -> None:
        # The classic failure: an expired share link returns an HTML page saved under
        # the weights' filename, and nothing notices until the model loads garbage.
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            manager = _FakeManager(exists=True)
            spec = _spec(
                root,
                repo={"url": "https://example/model.git"},
                checkpoints=[
                    {
                        "url": "https://example/a.pth",
                        "dest": "ckpt/model.pth",
                        "sha256": "0" * 64,
                    }
                ],
            )
            target = manager.repo_dir(spec) / "ckpt" / "model.pth"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"<html>Sign in</html>")

            problems = verify_checkpoints(spec, manager)
        self.assertEqual(len(problems), 1)
        self.assertIn("sha256", problems[0])

    def test_adopt_marks_an_externally_built_environment_as_prepared(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            spec = _spec(Path(root))
            manager = _FakeManager(exists=True)
            self.assertFalse(manager.is_prepared(spec))
            manager.adopt(spec)
            self.assertTrue(manager.is_prepared(spec))

    def test_adopt_refuses_when_the_environment_does_not_exist(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            spec = _spec(Path(root))
            with self.assertRaises(RuntimeError):
                _FakeManager(exists=False).adopt(spec)


if __name__ == "__main__":
    unittest.main()
