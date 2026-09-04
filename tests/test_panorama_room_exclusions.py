import json
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "tools" / "3dfront_dataset" / "_lib"


@pytest.fixture
def exclusion_case(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT_DIR))
    root = tmp_path / "experiment"
    (root / "splits").mkdir(parents=True)
    rows = [{"room_id": "a/keep", "house_id": "a", "split": "train"},
            {"room_id": "a/drop", "house_id": "a", "split": "train"},
            {"room_id": "b/keep", "house_id": "b", "split": "test"}]
    (root / "splits/rooms.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    (root / "splits/summary.json").write_text('{"seed":123}')
    for split in ("train", "val", "test"):
        (root / "splits" / (split+".txt")).write_text("".join(r["room_id"]+"\n" for r in sorted(rows, key=lambda r: r["room_id"]) if r["split"] == split))
    policy = {"policy_id": "test", "expected_original_rooms": 3, "expected_retained_rooms": 2,
              "views_per_room": 4, "rooms": [{**rows[1], "reason": "geometry unresolved"}]}
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))
    inventory = tmp_path / "unresolved.jsonl"
    inventory.write_text(json.dumps(rows[1])+"\n")
    return root, policy_path, inventory, rows


def test_dry_run_makes_no_changes(exclusion_case):
    from selection import apply_policy, read_rows
    root, policy, inventory, rows = exclusion_case
    report = apply_policy(root, policy, inventory)
    assert report["retained_rooms"] == 2 and report["expected_panoramas"] == 8
    assert not report["applied"]
    assert read_rows(root / "splits/rooms.jsonl") == rows
    assert not (root / "state").exists()


def test_apply_preserves_backup_membership_and_blocks_training(exclusion_case):
    from selection import apply_policy, read_rows
    root, policy, inventory, rows = exclusion_case
    report = apply_policy(root, policy, inventory, apply=True)
    assert report["applied"] and report["house_disjoint"]
    assert read_rows(root / "splits/rooms.jsonl") == [rows[0], rows[2]]
    assert read_rows(root / "state/splits-before-approved-exclusions/rooms.jsonl") == rows
    assert json.loads((root / "state/training_gate.json").read_text())["training_approved"] is False
    assert apply_policy(root, policy, inventory, apply=True)["already_applied"]


def test_inventory_mismatch_fails_before_changes(exclusion_case):
    from selection import apply_policy
    root, policy, inventory, rows = exclusion_case
    inventory.write_text(json.dumps(rows[0])+"\n")
    with pytest.raises(ValueError, match="inventory"):
        apply_policy(root, policy, inventory, apply=True)
    assert not (root / "state").exists()


def test_stale_derived_data_is_not_silently_kept(exclusion_case):
    from selection import apply_policy
    root, policy, inventory, _ = exclusion_case
    (root / "coco").mkdir()
    (root / "coco/train.json").write_text("{}")
    with pytest.raises(ValueError, match="derived data"):
        apply_policy(root, policy, inventory, apply=True)


def test_house_leakage_is_rejected(exclusion_case):
    from selection import filter_rooms
    _, policy_path, _, rows = exclusion_case
    rows[2]["house_id"] = "a"
    with pytest.raises(ValueError, match="House leakage"):
        filter_rooms(rows, json.loads(policy_path.read_text()))


def test_idempotent_run_detects_reintroduced_excluded_ids(exclusion_case):
    from selection import apply_policy
    root, policy, inventory, _ = exclusion_case
    apply_policy(root, policy, inventory, apply=True)
    (root / "splits/train.txt").write_text("a/drop\na/keep\n")
    with pytest.raises(ValueError, match="Inconsistent split"):
        apply_policy(root, policy, inventory, apply=True)
