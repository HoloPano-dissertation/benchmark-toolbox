#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def failure_reason(text):
    reason = ""
    for line in (text or "").splitlines():
        line = line.strip()
        if line.startswith(("ValueError:", "RuntimeError:", "FileExistsError:",
                            "FileNotFoundError:", "KeyError:", "MemoryError",
                            "NotImplementedError:", "OSError:")):
            reason = line
    return reason[:500]


def discard_room(output):
    shutil.rmtree(output, ignore_errors=True)


def resume_state(output, expected, views):
    try:
        metadata = json.loads((output / "render.json").read_text())
    except (OSError, ValueError):
        return "redo", None
    frames = [output / f"{index}.hdf5" for index in range(views)]
    if not all(frame.is_file() and frame.stat().st_size for frame in frames):
        return "redo", metadata
    if (metadata.get("implementation_sha256") != expected["implementation_sha256"]
            or metadata.get("samples") != expected["samples"]
            or metadata.get("views") != views
            or metadata.get("requested_min_clearance") != expected["requested_min_clearance"]
            or metadata.get("camera_height_fraction", 0.6) != expected["camera_height_fraction"]):
        return "stale", metadata
    return "keep", metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rooms_jsonl", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--min-clearance", type=float, default=0.1)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--architecture-root", type=Path,
                        help="Root of the architecture rebuilt from the original "
                             "3D-FRONT; each room is taken from <root>/<house>/<room>")
    parser.add_argument("--ambient", type=float)
    parser.add_argument("--light-temperature")
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("Invalid shard index/count")
    records = [json.loads(line) for line in args.rooms_jsonl.read_text().splitlines() if line.strip()]
    records = records[args.shard_index::args.shard_count]
    if args.limit:
        records = records[:args.limit]
    failed = []
    runner = Path(__file__).resolve().parent / "run_room.sh"
    for record in records:
        room_id, split = record["room_id"], record["split"]
        clearance = float(record.get("min_clearance", args.min_clearance))
        height_fraction = float(record.get("camera_height_fraction", 0.6))
        output = args.output_root / split / room_id
        if (output / ".complete").is_file() and not args.plan_only:
            expected = {
                "implementation_sha256": {
                    name: hashlib.sha256((runner.parent / name).read_bytes()).hexdigest()
                    for name in ("render.py", "camera_policy.py", "room_layout.py", "glb_geometry.py")},
                "samples": args.samples,
                "requested_min_clearance": clearance,
                "camera_height_fraction": height_fraction,
            }
            state, _ = resume_state(output, expected, args.views)
            if state == "stale":
                raise RuntimeError("Completed room was made with different code/settings or is incomplete; use a new output root")
            if state == "keep":
                print(f"Already complete: {room_id}", flush=True)
                continue
            print(f"INCOMPLETE {room_id}: marker without a usable render behind it, "
                  f"rendering again", flush=True)
        discard_room(output)
        command = ["bash", str(runner), record["room_dir"], str(output),
                   "--views", str(args.views), "--min-clearance", str(clearance),
                   "--camera-height-fraction", str(height_fraction),
                   "--samples", str(args.samples)]
        if args.architecture_root:
            architecture = args.architecture_root / room_id
            if not (architecture / "architecture.obj").is_file():
                failed.append({"room_id": room_id, "split": split, "exit_code": 0,
                               "reason": "no rebuilt architecture for this room"})
                print(f"FAILED {room_id}: no architecture", flush=True)
                continue
            command += ["--architecture", str(architecture)]
        if args.ambient is not None:
            command += ["--ambient", str(args.ambient)]
        if args.light_temperature:
            command += ["--light-temperature", str(args.light_temperature)]
        if args.plan_only:
            command.append("--plan-only")
        print(f"START {split}/{room_id}", flush=True)
        result = subprocess.run(command, check=False, stderr=subprocess.PIPE, text=True)
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr, flush=True)
        if result.returncode:
            reason = failure_reason(result.stderr)
            failed.append({"room_id": room_id, "split": split,
                           "exit_code": result.returncode, "reason": reason})
            print(f"FAILED {room_id}: {result.returncode} {reason}", flush=True)
        else:
            if not args.plan_only:
                metadata = json.loads((output / "render.json").read_text())
                if metadata["plan_only"] or not all((output / f"{i}.hdf5").is_file() for i in range(args.views)):
                    raise RuntimeError("Renderer exited without complete data")
                (output / ".complete").write_text("floor-supported render complete\n")
            print(f"DONE {room_id}", flush=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    report = args.output_root / f"shard-{args.shard_index:03d}.json"
    report.write_text(json.dumps({"rooms": len(records), "failed": failed}, indent=2)+"\n")
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
