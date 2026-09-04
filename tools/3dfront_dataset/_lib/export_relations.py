#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from relations import DEFAULT_EXPAND_DISTANCE, camera_frame_layout, scene_relations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--expand-distance", type=float, default=DEFAULT_EXPAND_DISTANCE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.experiment_root.resolve()
    counts: Counter[str] = Counter()
    sample_counts: Counter[str] = Counter()
    scenes_without_relations = 0

    for split in ("train", "val", "test"):
        records = [
            json.loads(line)
            for line in (root / "manifests_gt" / f"{split}.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        for record in records:
            path = Path(record["ground_truth"])
            scene = json.loads(path.read_text(encoding="utf-8"))
            metadata = scene["metadata"]
            layout = camera_frame_layout(metadata["layout_geometry_world"],
                                         metadata["camera_location_world"])
            relations, walls = scene_relations(scene["objects"], layout,
                                               args.expand_distance)
            scene["relations"] = relations
            metadata["walls"] = walls
            metadata["relation_expand_distance"] = args.expand_distance
            path.write_text(json.dumps(scene, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
            for relation in relations:
                counts[relation["relation_type"]] += 1
            if not relations:
                scenes_without_relations += 1
            sample_counts[split] += 1

    status = {
        "ready": True,
        "sample_counts": dict(sample_counts),
        "expand_distance": args.expand_distance,
        "relation_counts": dict(sorted(counts.items())),
        "scenes_without_relations": scenes_without_relations,
        "relation_types": sorted(counts),
        "walls": "one box per edge of the recovered contour, not a Manhattan cuboid",
    }
    (root / "state" / "relations.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
