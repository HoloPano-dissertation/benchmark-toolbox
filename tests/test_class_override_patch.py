import yaml
from pathlib import Path

from benchmark_toolbox.environments.patches import apply_patches

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = """import os

IG56CLASSES = ['basket', 'bathtub', 'bed']

IG59CLASSES = IG56CLASSES + ['walls', 'floors', 'ceilings']
"""


def class_patch():
    spec = yaml.safe_load((ROOT / "configs/environments/dpc.yaml").read_text())
    return [p for p in spec["patches"] if p.get("file") == "configs/data_config.py"]


def prepared(tmp_path):
    target = tmp_path / "configs" / "data_config.py"
    target.parent.mkdir(parents=True)
    target.write_text(ORIGINAL, encoding="utf-8")
    apply_patches(class_patch(), tmp_path)
    return target


def run(target):
    namespace = {}
    exec(compile(target.read_text(), str(target), "exec"), namespace)
    return namespace


def test_the_patch_is_declared_for_the_class_list():
    assert len(class_patch()) == 1


def test_without_the_variable_the_igibson_classes_are_kept(tmp_path, monkeypatch):
    monkeypatch.delenv("PANO3D_CLASSES", raising=False)
    namespace = run(prepared(tmp_path))
    assert namespace["IG56CLASSES"] == ['basket', 'bathtub', 'bed']
    assert namespace["IG59CLASSES"][-3:] == ['walls', 'floors', 'ceilings']


def test_with_the_variable_our_classes_replace_them(tmp_path, monkeypatch):
    monkeypatch.setenv("PANO3D_CLASSES", "bed,chair,table")
    namespace = run(prepared(tmp_path))
    assert namespace["IG56CLASSES"] == ["bed", "chair", "table"]
    assert namespace["IG59CLASSES"] == ["bed", "chair", "table",
                                        "walls", "floors", "ceilings"]


def test_the_patch_does_not_repeat_a_line_that_already_exists(tmp_path):
    target = prepared(tmp_path)
    body = target.read_text()
    assert "PANO3D_CLASSES" in body, "the guard on repeated lines must not skip it"


def test_applying_the_patch_twice_changes_nothing(tmp_path):
    target = prepared(tmp_path)
    once = target.read_text()
    apply_patches(class_patch(), tmp_path)
    assert target.read_text() == once
