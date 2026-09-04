#!/usr/bin/env python3
"""Make reproducible review sheets from an existing audit, without rerendering."""

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--all", action="store_true", help="Review every room of a small QA subset")
    parser.add_argument("--all-flagged", action="store_true",
                        help="Include every ceiling/object/image anomaly, plus the fixed random sample")
    args = parser.parse_args()
    root, output = args.report_dir.resolve(), args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Use an empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    rooms = [json.loads(line) for line in (root / "rooms.jsonl").read_text().splitlines()]
    by_id = {room["room_id"]: room for room in rooms}
    selected = {}

    def add(room, reason):
        entry = selected.setdefault(room["room_id"], {"room_id": room["room_id"],
                                    "preview": room["preview"], "reasons": []})
        entry["reasons"].append(reason)

    if args.all:
        for record in rooms:
            add(record, "complete_QA_subset")
    elif (root / "random_sample.json").is_file():
        for record in json.loads((root / "random_sample.json").read_text()):
            add(by_id[record["room_id"]], "fixed_seed_random_sample")
    else:
        generator = random.Random(20260902)
        for split in ("train", "val", "test"):
            candidates = sorted((r for r in rooms if r["split"] == split), key=lambda r: r["room_id"])
            for record in generator.sample(candidates, min(8, len(candidates))):
                add(record, "fixed_seed_random_sample")
    criteria = {
        "geometry_or_io_failure": lambda r: not r.get("geometry_pass", False) or any(
            f in r["flags"] for f in ("audit_error", "image_read_error", "camera_clipping_unverified")),
        "camera_outside_floor": lambda r: "no_floor_below_camera" in r["flags"],
        "missing_floor": lambda r: "missing_floor_geometry" in r["flags"],
        "nonrectangular_floor": lambda r: r.get("floor_oriented_bbox_fill", 1) < 0.95 or "nonrectangular_floor" in r["flags"],
        "rectangular_but_wrong_proxy": lambda r: "layout_proxy_mismatch" in r["flags"] and "nonrectangular_floor" not in r["flags"],
        "image_anomaly": lambda r: any(f in r["flags"] for f in ("mostly_dark", "mostly_clipped", "near_surface_dominates", "almost_uniform_rgb")),
        "floor_target_pass": lambda r: r.get("proxy_xy_pass", r.get("geometry_pass", False)),
        "source_object_outlier": lambda r: "oversized_source_object_requires_review" in r["flags"],
        "ceiling_height": lambda r: "ceiling_height_requires_review" in r["flags"],
    }
    for reason, criterion in criteria.items():
        candidates = [r for r in rooms if criterion(r)]
        # Diverse rooms, deterministically spread across the candidate list.
        if args.all_flagged and reason in {"geometry_or_io_failure", "camera_outside_floor", "missing_floor",
                                           "image_anomaly", "source_object_outlier", "ceiling_height"}:
            indices = range(len(candidates))
        else:
            indices = sorted({0, len(candidates)//2, len(candidates)-1}) if candidates else []
        for index in indices:
            add(candidates[index], reason)
    entries = list(selected.values())
    for page, start in enumerate(range(0, len(entries), 4), 1):
        sheet = Image.new("RGB", (2440, 1280), "#151c28")
        draw = ImageDraw.Draw(sheet)
        for slot, entry in enumerate(entries[start:start+4]):
            x, y = (slot % 2)*1220, (slot//2)*640
            draw.text((x+12, y+8), ", ".join(entry["reasons"]), fill="white")
            with Image.open(root / entry["preview"]) as image:
                sheet.paste(image, (x, y+28))
            entry["page"] = f"page-{page:02d}.jpg"
            entry["slot"] = slot
        sheet.save(output / f"page-{page:02d}.jpg", quality=90)
    (output / "selection.json").write_text(json.dumps(entries, indent=2) + "\n")
    print(f"Prepared {len(entries)} rooms on {(len(entries)+3)//4} review sheets")


if __name__ == "__main__":
    main()
