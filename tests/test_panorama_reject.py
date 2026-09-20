import json
from pathlib import Path

import pytest

DATASET = Path(__file__).resolve().parents[1] / "tools" / "3dfront_dataset"


@pytest.fixture(scope="module")
def reject():
    import sys
    sys.path.insert(0, str(DATASET))
    from _lib import reject as module
    return module


def rendered_room(root, views=4, complete=True, metadata=True, empty_frame=None):
    root.mkdir(parents=True, exist_ok=True)
    for index in range(views):
        size = b"" if empty_frame == index else b"frame"
        (root / f"{index}.hdf5").write_bytes(size)
    (root / "render.json").write_text("{}" if metadata else "", encoding="utf-8")
    if complete:
        (root / ".complete").write_text("done\n", encoding="utf-8")
    return root


def rows(*room_ids):
    return [{"room_id": room_id, "house_id": room_id.split("/")[0], "split": "train"}
            for room_id in room_ids]


def test_a_room_counts_as_rendered_only_when_everything_is_there(reject, tmp_path):
    assert reject.usable_render(rendered_room(tmp_path / "ok"), 4)
    assert not reject.usable_render(rendered_room(tmp_path / "a", complete=False), 4)
    assert not reject.usable_render(rendered_room(tmp_path / "b", metadata=False), 4)
    assert not reject.usable_render(rendered_room(tmp_path / "c", empty_frame=2), 4)
    assert not reject.usable_render(tmp_path / "never", 4)


def test_the_plan_names_every_room_that_has_no_render(reject, tmp_path):
    rendered_room(tmp_path / "h1" / "Bedroom-1")
    reasons = {"h2/Bedroom-2": "no usable camera poses"}
    plan = reject.plan_exclusions(rows("h1/Bedroom-1", "h2/Bedroom-2"), tmp_path, 4,
                                 reasons, "the renderer")
    assert [item["room_id"] for item in plan] == ["h2/Bedroom-2"]
    assert plan[0]["reason"] == "no usable camera poses"
    assert plan[0]["split"] == "excluded"


def test_a_failure_without_a_reason_stops_the_work(reject, tmp_path):
    rendered_room(tmp_path / "h1" / "Bedroom-1")
    with pytest.raises(ValueError, match="h2/Bedroom-2"):
        reject.plan_exclusions(rows("h1/Bedroom-1", "h2/Bedroom-2"), tmp_path, 4, {},
                               "the renderer")


def test_a_reason_for_a_room_that_was_repaired_is_ignored(reject, tmp_path):
    """Logs keep every failure, including the ones a later run put right."""
    rendered_room(tmp_path / "h1" / "Bedroom-1")
    plan = reject.plan_exclusions(rows("h1/Bedroom-1"), tmp_path, 4,
                                  {"h1/Bedroom-1": "failed once, rendered later"},
                                  "the renderer")
    assert plan == []


def test_applying_the_plan_keeps_the_policy_counting(reject, tmp_path):
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "train.txt").write_text("h1/Bedroom-1\nh2/Bedroom-2\n", encoding="utf-8")
    (splits / "val.txt").write_text("h3/Bedroom-3\n", encoding="utf-8")
    (splits / "test.txt").write_text("h4/Bedroom-4\n", encoding="utf-8")
    (splits / "excluded_rooms.json").write_text(json.dumps({
        "policy_id": "excluded-rooms", "expected_retained_rooms": 4,
        "expected_original_rooms": 5,
        "rooms": [{"room_id": "h5/Bedroom-5", "house_id": "h5", "split": "excluded",
                   "reason": "the contour collapsed", "evidence": "preparation"}]}),
        encoding="utf-8")
    report = reject.apply_to_splits(splits, [
        {"room_id": "h2/Bedroom-2", "house_id": "h2", "split": "excluded",
         "reason": "no usable camera poses", "evidence": "the renderer"}])
    assert report == {"rooms": 3, "excluded": 2, "excluded_now": 1}
    assert (splits / "train.txt").read_text().split() == ["h1/Bedroom-1"]
    policy = json.loads((splits / "excluded_rooms.json").read_text())
    assert policy["expected_retained_rooms"] == 3
    assert policy["expected_original_rooms"] == 5
    assert {item["room_id"] for item in policy["rooms"]} == {"h5/Bedroom-5", "h2/Bedroom-2"}


def test_applying_the_same_plan_twice_changes_nothing_more(reject, tmp_path):
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "train.txt").write_text("h1/Bedroom-1\nh2/Bedroom-2\n", encoding="utf-8")
    (splits / "val.txt").write_text("", encoding="utf-8")
    (splits / "test.txt").write_text("", encoding="utf-8")
    (splits / "excluded_rooms.json").write_text(json.dumps({
        "policy_id": "excluded-rooms", "expected_retained_rooms": 2,
        "expected_original_rooms": 2, "rooms": []}), encoding="utf-8")
    plan = [{"room_id": "h2/Bedroom-2", "house_id": "h2", "split": "excluded",
             "reason": "no usable camera poses", "evidence": "the renderer"}]
    first = reject.apply_to_splits(splits, plan)
    second = reject.apply_to_splits(splits, plan)
    assert first == second
    policy = json.loads((splits / "excluded_rooms.json").read_text())
    assert len(policy["rooms"]) == 1
