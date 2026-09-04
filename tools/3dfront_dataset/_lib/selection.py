import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


SPLITS = ("train", "val", "test")


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def index_rows(rows):
    result = {}
    houses = {}
    for row in rows:
        if row["room_id"] in result:
            raise ValueError("Duplicate room: " + row["room_id"])
        if row["split"] not in SPLITS:
            raise ValueError("Unknown split")
        house = row["house_id"]
        if houses.setdefault(house, row["split"]) != row["split"]:
            raise ValueError("House leakage: " + house)
        result[row["room_id"]] = row
    return result


def filter_rooms(original, policy):
    by_id = index_rows(original)
    excluded = index_rows(policy["rooms"])
    if len(by_id) != policy["expected_original_rooms"]:
        raise ValueError("Original room count does not match the reviewed policy")
    for room_id, row in excluded.items():
        if room_id not in by_id:
            raise ValueError("Excluded room absent from original split: " + room_id)
        if any(row[key] != by_id[room_id][key] for key in ("house_id", "split")):
            raise ValueError("Exclusion would refer to a different house/split: " + room_id)
        if not row.get("reason"):
            raise ValueError("Every exclusion requires a recorded reason")
    retained = [r for r in original if r["room_id"] not in excluded]
    if len(retained) != policy["expected_retained_rooms"]:
        raise ValueError("Retained count does not match the reviewed policy")
    index_rows(retained)
    counts = {s: sum(r["split"] == s for r in retained) for s in SPLITS}
    summary = {
        "policy_id": policy["policy_id"], "original_rooms": len(original),
        "excluded_rooms": len(excluded), "retained_rooms": len(retained),
        "room_counts": counts,
        "excluded_room_counts": dict(Counter(r["split"] for r in excluded.values())),
        "house_counts": {s: len({r["house_id"] for r in retained if r["split"] == s}) for s in SPLITS},
        "panorama_counts": {s: n*policy["views_per_room"] for s, n in counts.items()},
        "expected_panoramas": len(retained)*policy["views_per_room"],
        "house_disjoint": True, "source_assets_deleted": False,
        "geometry_reconstructed": False, "training_approved": False,
    }
    return retained, summary


def validate_active_split(split_dir, expected):
    if read_rows(split_dir / "rooms.jsonl") != expected:
        raise ValueError("Active room manifest differs from the reviewed filtered split")
    for split in SPLITS:
        actual = (split_dir / (split+".txt")).read_text().splitlines()
        wanted = sorted(r["room_id"] for r in expected if r["split"] == split)
        if actual != wanted:
            raise ValueError("Inconsistent split text file: " + split)


def apply_policy(root, policy_path, inventory, apply=False):
    root, policy_path, inventory = map(Path, (root, policy_path, inventory))
    policy = json.loads(policy_path.read_text())
    if set(index_rows(read_rows(inventory))) != set(index_rows(policy["rooms"])):
        raise ValueError("Final unresolved inventory differs from the approved exclusion IDs")
    digest = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()
    split_dir = root / "splits"
    state_dir = root / "state"
    backup = state_dir / "splits-before-approved-exclusions"
    active_policy_path = split_dir / "exclusion_policy.json"
    if active_policy_path.is_file():
        active = json.loads(active_policy_path.read_text())
        if active.get("policy_sha256") != digest:
            raise ValueError("A different exclusion policy is already active")
        retained, summary = filter_rooms(read_rows(backup / "rooms.jsonl"), policy)
        validate_active_split(split_dir, retained)
        return {**summary, "already_applied": True, "applied": True}
    if backup.exists():
        raise FileExistsError("Previous exclusion transaction requires inspection: " + str(backup))
    for name in ("manifests", "manifests_gt", "coco", "horizonnet", "dpc_dataset", "ground_truth", "rgb"):
        path = root / name
        if path.exists() and any(path.iterdir()):
            raise ValueError("Existing derived data must be archived/rebuilt first: " + str(path))
    original = read_rows(split_dir / "rooms.jsonl")
    retained, summary = filter_rooms(original, policy)
    report = {**summary, "policy_sha256": digest,
              "backup": str(backup), "applied": apply}
    if not apply:
        return report
    state_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(split_dir, backup)
    with tempfile.TemporaryDirectory(prefix="filtered-splits-", dir=str(state_dir)) as tmp:
        staged = Path(tmp)
        (staged / "rooms.jsonl").write_text("".join(json.dumps(r)+"\n" for r in retained))
        for split in SPLITS:
            ids = sorted(r["room_id"] for r in retained if r["split"] == split)
            (staged / (split+".txt")).write_text("".join(s+"\n" for s in ids))
        previous_summary = json.loads((backup / "summary.json").read_text())
        (staged / "summary.json").write_text(json.dumps({**previous_summary, **summary}, indent=2)+"\n")
        (staged / "excluded_rooms.jsonl").write_text("".join(json.dumps(r)+"\n" for r in policy["rooms"]))
        (staged / "exclusion_policy.json").write_text(json.dumps({**policy, **report}, indent=2)+"\n")
        validate_active_split(staged, retained)
        gate = {"training_approved": False, "policy_id": policy["policy_id"],
                "reason": "Filtered dataset requires a fresh full audit and derived-data QA"}
        (state_dir / "training_gate.json").write_text(json.dumps(gate, indent=2)+"\n")
        for path in staged.iterdir():
            os.replace(str(path), str(split_dir / path.name))
    validate_active_split(split_dir, retained)
    (state_dir / "dataset_selection.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--policy", type=Path,
                        default=Path(__file__).resolve().parents[1] / "splits" / "excluded_rooms.json")
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    inventory = args.inventory or args.experiment_root / "validation/plan-inventory-v3/unresolved.jsonl"
    print(json.dumps(apply_policy(args.experiment_root, args.policy, inventory, args.apply), indent=2))


if __name__ == "__main__":
    main()
