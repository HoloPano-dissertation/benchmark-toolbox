"""Prevent a successful smoke job from silently authorizing unvalidated full data."""
import json
from pathlib import Path


def require_training_approval(experiment_root, smoke=False):
    if smoke:
        return
    path = Path(experiment_root) / "state" / "training_gate.json"
    if not path.is_file():
        raise RuntimeError("Full training blocked: no data-quality approval at %s" % path)
    report = json.loads(path.read_text())
    if report.get("training_approved") is not True:
        raise RuntimeError("Full training blocked by data-quality gate: %s" % report.get("reason", "unapproved"))
