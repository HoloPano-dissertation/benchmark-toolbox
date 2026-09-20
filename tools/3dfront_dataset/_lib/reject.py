from __future__ import annotations

import json
from pathlib import Path

SPLITS = ("train", "val", "test")


def usable_render(room_dir, views):
    room_dir = Path(room_dir)
    if not (room_dir / ".complete").is_file():
        return False
    try:
        json.loads((room_dir / "render.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    frames = [room_dir / f"{index}.hdf5" for index in range(views)]
    return all(frame.is_file() and frame.stat().st_size for frame in frames)


def unrendered(rooms, rendered_root, views):
    rendered_root = Path(rendered_root)
    return [row for row in rooms
            if not usable_render(rendered_root / row["room_id"], views)]


def read_reasons(path):
    reasons = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(reasons, dict) or not all(
            isinstance(key, str) and isinstance(value, str) and value.strip()
            for key, value in reasons.items()):
        raise ValueError("Reasons must be an object of room id -> what the renderer said")
    return reasons


def plan_exclusions(rooms, rendered_root, views, reasons, evidence):
    missing = unrendered(rooms, rendered_root, views)
    by_id = {row["room_id"]: row for row in missing}
    unexplained = sorted(set(by_id) - set(reasons))
    if unexplained:
        raise ValueError(
            "%d rooms have no usable render and no recorded reason; excluding a room "
            "without saying why is how a set loses count of itself: %s"
            % (len(unexplained), ", ".join(unexplained[:5])))
    return [{"room_id": row["room_id"], "house_id": row["room_id"].split("/")[0],
             "split": "excluded", "reason": reasons[row["room_id"]], "evidence": evidence}
            for row in sorted(missing, key=lambda r: r["room_id"])]


def apply_to_splits(splits_dir, exclusions):
    splits_dir = Path(splits_dir)
    leaving = {item["room_id"] for item in exclusions}
    retained = 0
    for name in SPLITS:
        path = splits_dir / (name + ".txt")
        kept = [room_id for room_id in path.read_text(encoding="utf-8").split()
                if room_id not in leaving]
        path.write_text("".join(room_id + "\n" for room_id in kept), encoding="utf-8")
        retained += len(kept)
    policy_path = splits_dir / "excluded_rooms.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    known = {item["room_id"] for item in policy["rooms"]}
    policy["rooms"].extend(item for item in exclusions if item["room_id"] not in known)
    policy["expected_retained_rooms"] = retained
    policy["expected_original_rooms"] = retained + len(policy["rooms"])
    policy_path.write_text(json.dumps(policy, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    return {"rooms": retained, "excluded": len(policy["rooms"]),
            "excluded_now": len(exclusions)}
