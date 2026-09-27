import importlib.util
import sys
import types
from pathlib import Path

import pytest

TRAINING = Path(__file__).resolve().parents[1] / "tools" / "3dfront_training"


@pytest.fixture(scope="module")
def detector():
    for name in ("detectron2", "detectron2.config", "detectron2.data",
                 "detectron2.data.datasets", "detectron2.engine",
                 "detectron2.evaluation", "detectron2.model_zoo", "torch"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["detectron2.engine"].DefaultTrainer = type("DefaultTrainer", (), {})
    sys.modules["detectron2.evaluation"].COCOEvaluator = type("COCOEvaluator", (), {})
    sys.modules["detectron2.config"].get_cfg = lambda: None
    sys.modules["detectron2.data.datasets"].register_coco_instances = lambda *a, **k: None
    sys.modules["detectron2.model_zoo"].model_zoo = None
    sys.path.insert(0, str(TRAINING))
    spec = importlib.util.spec_from_file_location(
        "train_detector", TRAINING / "train_detector.py")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        pytest.skip("detectron2 is not installed here")
    return module


def schedule_fits(max_iter, steps):
    """detectron2 refuses a schedule whose phases outnumber the updates."""
    return max_iter > len(steps) + 1


def test_a_two_iteration_smoke_gets_no_decay(detector):
    assert detector.decay_milestones(2) == ()
    assert schedule_fits(2, detector.decay_milestones(2))


def test_a_full_run_decays_late_twice(detector):
    assert detector.decay_milestones(14000) == (9800, 12600)
    assert schedule_fits(14000, detector.decay_milestones(14000))


def test_every_short_run_keeps_a_schedule_detectron_accepts(detector):
    for max_iter in range(2, 40):
        steps = detector.decay_milestones(max_iter)
        assert schedule_fits(max_iter, steps), (max_iter, steps)
        assert len(set(steps)) == len(steps)
        assert all(0 < step < max_iter for step in steps)


def test_one_iteration_is_refused_rather_than_quietly_reshaped(detector):
    assert detector.decay_milestones(1) == ()
    assert not schedule_fits(1, ())
