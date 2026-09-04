import yaml
from pathlib import Path

CONFIGS = sorted((Path(__file__).resolve().parents[1] / "configs" / "dpc").glob("*front3d*.yaml"))


def loaded():
    return {path.name: yaml.safe_load(path.read_text(encoding="utf-8")) for path in CONFIGS}


def test_every_component_has_a_configuration():
    assert {path.name for path in CONFIGS} == {
        "bdb3d_estimation_front3d.yaml", "ldif_front3d.yaml",
        "mgnet_front3d.yaml", "relation_scene_gcn_front3d.yaml"}


def test_no_path_still_points_at_igibson_data():
    for name, config in loaded().items():
        for key, value in config.get("data", {}).items():
            if isinstance(value, str) and "/" in value:
                assert "igibson" not in value, f"{name}: {key} = {value}"
        for weight in config.get("weight", []):
            assert "igibson" not in weight, name


def test_scene_and_object_datasets_are_the_expected_ones():
    configs = loaded()
    assert configs["bdb3d_estimation_front3d.yaml"]["data"]["split"].endswith("dpc_scenes")
    assert configs["relation_scene_gcn_front3d.yaml"]["data"]["split"].endswith("dpc_scenes")
    assert configs["ldif_front3d.yaml"]["data"]["split"].endswith("objects")
    assert configs["mgnet_front3d.yaml"]["data"]["split"].endswith("objects")


def test_box_estimation_starts_from_scratch():
    config = loaded()["bdb3d_estimation_front3d.yaml"]
    assert config["weight"] == []
    assert config["finetune"] is False


def test_the_relation_stage_declares_its_two_inputs():
    weights = loaded()["relation_scene_gcn_front3d.yaml"]["weight"]
    assert len(weights) == 2
    assert any("bdb3d_estimation_front3d" in w for w in weights)
    assert any("ldif_front3d" in w for w in weights)


def test_log_paths_do_not_collide_with_the_igibson_runs():
    for name, config in loaded().items():
        path = config.get("log", {}).get("path", "")
        assert path.endswith("front3d"), name
