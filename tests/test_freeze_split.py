import importlib.util
import json
from pathlib import Path

import pytest

DATASET = Path(__file__).resolve().parents[1] / "tools" / "3dfront_dataset"
spec = importlib.util.spec_from_file_location(
    "freeze", DATASET / "_lib" / "freeze.py",
    submodule_search_locations=[str(DATASET / "_lib")])


@pytest.fixture(scope="module")
def freeze():
    import sys
    sys.path.insert(0, str(DATASET))
    from _lib import freeze as module
    return module


def rooms_for(count, per_house=2):
    return ["house-%02d/Bedroom-%d" % (index // per_house, index)
            for index in range(count)]


def test_split_keeps_a_house_in_one_part(freeze):
    assigned = freeze.house_disjoint_split(rooms_for(120), seed=1)
    owner = {}
    for name, values in assigned.items():
        for room_id in values:
            owner.setdefault(room_id.split("/")[0], name)
            assert owner[room_id.split("/")[0]] == name


def test_split_respects_the_requested_shares(freeze):
    assigned = freeze.house_disjoint_split(rooms_for(400), val=0.2, test=0.1, seed=3)
    total = sum(len(v) for v in assigned.values())
    assert total == 400
    assert 0.14 < len(assigned["val"]) / total < 0.26
    assert 0.05 < len(assigned["test"]) / total < 0.16


def test_split_is_reproducible_for_a_seed(freeze):
    first = freeze.house_disjoint_split(rooms_for(200), seed=7)
    second = freeze.house_disjoint_split(rooms_for(200), seed=7)
    other = freeze.house_disjoint_split(rooms_for(200), seed=8)
    assert first == second
    assert first != other


def test_shares_that_leave_no_training_data_are_rejected(freeze):
    with pytest.raises(ValueError):
        freeze.house_disjoint_split(rooms_for(20), val=0.6, test=0.5)


def test_a_room_without_an_exact_scale_is_excluded_with_a_reason(freeze, tmp_path, monkeypatch):
    ring = [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]
    monkeypatch.setattr(freeze, "recover", lambda root, room: {
        "floor_z": -0.7, "ceiling_z": 0.7, "area": 4.0,
        "bounds_min": [-1, -1, -0.7], "bounds_max": [1, 1, 0.7],
        "polygon": {"type": "Polygon", "coordinates": [ring]},
        "camera_region": {"type": "Polygon", "coordinates": [ring]}})
    monkeypatch.setattr(freeze, "surface_tree", lambda *a, **k: _FarTree())
    kept, excluded = freeze.review_rooms(
        tmp_path, ["house/a", "house/b"], table={"house/a": 2.0})
    assert kept == ["house/a"]
    assert len(excluded) == 1
    assert "cannot be measured" in excluded[0]["reason"]


def test_a_room_whose_geometry_fails_is_recorded_not_raised(freeze, tmp_path, monkeypatch):
    def explode(root, room):
        raise ValueError("no ceiling patch")
    monkeypatch.setattr(freeze, "recover", explode)
    kept, excluded = freeze.review_rooms(tmp_path, ["house/a"])
    assert kept == []
    assert "no ceiling patch" in excluded[0]["reason"]


def test_writing_refuses_to_replace_a_frozen_split(freeze, tmp_path):
    assigned = {"train": ["h/a"], "val": ["h/b"], "test": ["h/c"]}
    freeze.write_split(tmp_path, assigned, [], tmp_path)
    with pytest.raises(FileExistsError):
        freeze.write_split(tmp_path, assigned, [], tmp_path)
    report = freeze.write_split(tmp_path, assigned, [], tmp_path, force=True)
    assert report["rooms"] == 3


def test_the_policy_carries_the_expected_count_the_loader_checks(freeze, tmp_path):
    assigned = {"train": ["h/a", "h/b"], "val": ["i/c"], "test": ["j/d"]}
    excluded = [{"room_id": "k/e", "reason": "too small", "evidence": "checks"}]
    freeze.write_split(tmp_path, assigned, excluded, tmp_path)
    policy = json.loads((tmp_path / "excluded_rooms.json").read_text(encoding="utf-8"))
    assert policy["expected_retained_rooms"] == 4
    assert policy["expected_original_rooms"] == 5
    assert policy["rooms"][0]["house_id"] == "k"
    assert (tmp_path / "train.txt").read_text().split() == ["h/a", "h/b"]


def test_a_missing_source_is_reported_plainly(freeze, tmp_path):
    with pytest.raises(NotADirectoryError, match="not found"):
        freeze.source_rooms(tmp_path / "absent")


def test_a_source_without_rooms_is_reported_plainly(freeze, tmp_path):
    (tmp_path / "house").mkdir()
    with pytest.raises(ValueError, match="No rooms"):
        freeze.freeze(tmp_path, tmp_path / "splits")


def room_layout(half=1.5, height=2.6):
    ring = [[-half, -half], [half, -half], [half, half], [-half, half], [-half, -half]]
    return {"polygon": {"type": "Polygon", "coordinates": [ring]},
            "camera_region": {"type": "Polygon", "coordinates": [ring]},
            "floor_z": 0.0, "ceiling_z": height}


def test_an_open_room_offers_camera_poses(freeze, tmp_path, monkeypatch):
    monkeypatch.setattr(freeze, "surface_tree", lambda *a, **k: _FarTree())
    assert freeze.camera_failure(tmp_path, "house/room", room_layout()) is None


def test_a_room_without_clearance_is_named_before_rendering(freeze, tmp_path, monkeypatch):
    monkeypatch.setattr(freeze, "surface_tree", lambda *a, **k: _NearTree())
    failure = freeze.camera_failure(tmp_path, "house/room", room_layout())
    assert failure and "clearance" in failure


def test_a_room_without_geometry_is_named_too(freeze, tmp_path, monkeypatch):
    monkeypatch.setattr(freeze, "surface_tree", lambda *a, **k: None)
    assert "clearance" in freeze.camera_failure(tmp_path, "house/room", room_layout())


class _FarTree:
    def query(self, point):
        return 5.0, 0


class _NearTree:
    def query(self, point):
        return 0.001, 0
