#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from _lib.freeze import DEFAULT_TEST, DEFAULT_VAL
from _lib.selection import index_rows

HERE = Path(__file__).resolve().parent
STAGES = ("rgb", "ground_truth", "relations", "shape", "crops", "coco",
          "horizonnet", "dpc")

VIEWS_PER_ROOM = 4


def exclusion_policy():
    return json.loads((HERE / "splits/excluded_rooms.json").read_text(encoding="utf-8"))


def frozen_rooms():
    rows = []
    for split in ("train", "val", "test"):
        for room_id in (HERE / "splits" / (split+".txt")).read_text().splitlines():
            if Path(room_id).is_absolute() or ".." in Path(room_id).parts or len(Path(room_id).parts) != 2:
                raise ValueError("Invalid frozen room ID")
            rows.append({"room_id": room_id, "house_id": room_id.split("/")[0], "split": split})
    index_rows(rows)
    policy = exclusion_policy()
    if len(rows) != policy["expected_retained_rooms"] \
            or set(index_rows(rows)) & set(index_rows(policy["rooms"])):
        raise ValueError("Frozen split is inconsistent with the reviewed exclusions")
    return sorted(rows, key=lambda r: (r["split"], r["room_id"]))


def initialize(scene_root, root):
    if root.exists():
        raise FileExistsError("Use a new experiment directory")
    scene_root = scene_root.resolve()
    rows = frozen_rooms()
    sys.path.insert(0, str(HERE.parent / "3dfront_panorama_renderer"))
    from room_layout import recover_layout
    for row in rows:
        room = scene_root / row["room_id"]
        layout = recover_layout(room)
        height = layout["ceiling_z"]-layout["floor_z"]
        row.update(room_dir=str(room), min_clearance=min(0.1, 0.04*height))
    for name in ("splits", "source", "outputs", "state"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "source/3D-FRONT-TEST-SCENE").symlink_to(scene_root, target_is_directory=True)
    for split in ("train", "val", "test"):
        shutil.copyfile(HERE / "splits" / (split+".txt"), root / "splits" / (split+".txt"))
    (root / "splits/rooms.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    policy = exclusion_policy()
    (root / "splits/excluded_rooms.jsonl").write_text("".join(json.dumps(r)+"\n" for r in policy["rooms"]))
    (root / "splits/summary.json").write_text(json.dumps({
        "room_counts": {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")},
        "house_disjoint": True, "excluded_rooms": len(policy["rooms"]),
        "expected_panoramas": len(rows)*VIEWS_PER_ROOM,
        "source_root": str(scene_root), "policy_id": policy["policy_id"]}, indent=2)+"\n")
    (root / "state/training_gate.json").write_text('{"training_approved":false,"reason":"New dataset requires rendering and QA"}\n')
    print("Initialized %d rooms with frozen house-disjoint splits; source GLBs unchanged"
          % len(rows))


def run_module(name, root, extra=()):
    subprocess.run([sys.executable, str(HERE / "_lib" / (name+".py")), str(root), *extra], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Create a new experiment from source GLB rooms")
    init.add_argument("scene_root", type=Path)
    init.add_argument("experiment_root", type=Path)
    export = commands.add_parser("export", help="Build RGB, boxes, COCO, HorizonNet and DPC inputs")
    export.add_argument("experiment_root", type=Path)
    export.add_argument("--stage", choices=("all", *STAGES), default="all")
    export.add_argument("--scale-table", help="Exact per-room metric scales from the original 3D-FRONT")
    export.add_argument("--class-table", help="Object classes from the original 3D-FRONT")
    export.add_argument("--reference-height", help="Ceiling height in metres assumed by the scale anchor")
    export.add_argument("--allow-unscalable", action="store_true",
                        help="Record rooms failing a scale check instead of stopping")
    export.add_argument("--expand-distance", help="Slack in metres for a touching relation")
    validate = commands.add_parser("validate", help="Validate exported training/evaluation inputs")
    validate.add_argument("experiment_root", type=Path)
    freeze = commands.add_parser(
        "freeze", help="Build the frozen house-disjoint split from a source of GLB rooms")
    freeze.add_argument("scene_root", type=Path)
    freeze.add_argument("--metadata", help="Scale and class report of the original 3D-FRONT")
    freeze.add_argument("--val", type=float, default=DEFAULT_VAL)
    freeze.add_argument("--test", type=float, default=DEFAULT_TEST)
    freeze.add_argument("--seed", type=int, default=0)
    freeze.add_argument("--reference-height", type=float,
                        help="Ceiling height in metres assumed when no exact scale is given")
    freeze.add_argument("--force", action="store_true",
                        help="Replace an existing split; earlier results then describe another set")
    args = parser.parse_args()
    if args.command == "freeze":
        from _lib.freeze import freeze as build_split
        print(json.dumps(build_split(
            args.scene_root, HERE / "splits", args.metadata, args.val, args.test,
            args.seed, args.reference_height, args.force), indent=2, ensure_ascii=False))
        return
    root = args.experiment_root.resolve()
    if args.command == "init":
        initialize(args.scene_root, root)
    else:
        actual = index_rows([json.loads(s) for s in (root / "splits/rooms.jsonl").read_text().splitlines() if s.strip()])
        expected = index_rows(frozen_rooms())
        if set(actual) != set(expected) or any(actual[k]["split"] != expected[k]["split"] for k in expected):
            raise ValueError("Active dataset must match the frozen 981-room split")
        if args.command == "export":
            (root / "state").mkdir(exist_ok=True)
            (root / "state/training_gate.json").write_text('{"training_approved":false,"reason":"Exports changed; review and validation required"}\n')
            scale_options = []
            for name in ("scale_table", "reference_height", "class_table"):
                value = getattr(args, name)
                if value:
                    scale_options += ["--"+name.replace("_", "-"), str(value)]
            if args.allow_unscalable:
                scale_options.append("--allow-unscalable")
            relation_options = ["--expand-distance", str(args.expand_distance)] \
                if args.expand_distance else []
            for stage in STAGES if args.stage == "all" else (args.stage,):
                extra = {"ground_truth": scale_options, "relations": relation_options}
                run_module("export_"+stage, root, extra.get(stage, ()))
        else:
            run_module("validate_horizonnet", root)
            from _lib.validate_dataset import validate_dataset
            print(json.dumps(validate_dataset(root), indent=2))


if __name__ == "__main__":
    main()
