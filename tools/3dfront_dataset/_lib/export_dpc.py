#!/usr/bin/env python3
"""Create collision-free Pano3D image splits from MIDI benchmark manifests."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.experiment_root.resolve()
    manifest_root = root / "manifests_gt"
    dpc_root = root / "dpc_dataset"
    image_root = dpc_root / "images"
    dpc_root.mkdir(parents=True, exist_ok=True)
    mapping_records = []
    counts = {}

    for split in ("train", "val", "test"):
        records = [
            json.loads(line)
            for line in (manifest_root / f"{split}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        split_images = image_root / split
        split_images.mkdir(parents=True, exist_ok=True)
        dpc_entries = []
        for record in records:
            source = Path(record["input"]).resolve()
            safe_name = SAFE.sub("__", record["sample_id"]).strip("_") + ".png"
            destination = split_images / safe_name
            if destination.is_symlink():
                if destination.resolve() != source:
                    raise RuntimeError(f"Conflicting symlink: {destination}")
            elif destination.exists():
                raise RuntimeError(f"Refusing to replace existing path: {destination}")
            else:
                destination.symlink_to(source)
            relative = destination.relative_to(dpc_root).as_posix()
            dpc_entries.append(relative)
            mapping_records.append(
                {
                    "dpc_name": destination.stem,
                    "sample_id": record["sample_id"],
                    "split": split,
                    "input": str(source),
                    "ground_truth": record["ground_truth"],
                }
            )
        (dpc_root / f"{split}.json").write_text(
            json.dumps(dpc_entries, indent=2) + "\n", encoding="utf-8"
        )
        counts[split] = len(dpc_entries)

    first_test = json.loads((dpc_root / "test.json").read_text(encoding="utf-8"))[0]
    (dpc_root / "test_smoke.json").write_text(
        json.dumps([first_test], indent=2) + "\n", encoding="utf-8"
    )
    (dpc_root / "sample_map.jsonl").write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in sorted(mapping_records, key=lambda item: item["dpc_name"])
        ),
        encoding="utf-8",
    )
    status = {
        "ready": True,
        "root": str(dpc_root),
        "counts": counts,
        "unique_dpc_names": len({record["dpc_name"] for record in mapping_records}),
        "symlinks": len(mapping_records),
    }
    (root / "state" / "dpc_inputs.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
