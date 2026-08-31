#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys


def _oriented_box(centroid, size, yaw: float) -> dict:
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    return {
        "center": [float(value) for value in centroid],
        "size": [float(value) for value in size],
        "basis": [
            [cos_yaw, sin_yaw, 0.0],
            [-sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
    }


def _entries(pred_dir: str, stage1: str | None, split: str) -> list[tuple[str, str]]:
    if stage1:
        from holopano.data.object_dataset import _split_entries

        return list(_split_entries(stage1, split))
    found: list[tuple[str, str]] = []
    for scene in sorted(os.listdir(pred_dir)):
        scene_dir = os.path.join(pred_dir, scene)
        if not os.path.isdir(scene_dir):
            continue
        for name in sorted(os.listdir(scene_dir)):
            if name.endswith(".npz"):
                found.append((scene, name[: -len(".npz")]))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pred", required=True, help="HoloPano prediction dir (<scene>/<cam>.npz)")
    parser.add_argument("--det-export", required=True, help="export dir with ground truth")
    parser.add_argument("--out", required=True, help="output dir (pred/, gt/, manifest.jsonl)")
    parser.add_argument("--stage1", help="iGibson stage1 dir — restrict to a dataset split")
    parser.add_argument("--split", default="test")
    parser.add_argument("--score-threshold", type=float, default=0.0,
                        help="drop detections below this score (default: keep all)")
    parser.add_argument("--rgb-root", help="panorama root for the manifest 'input' field; "
                                           "by default it points at the export .npz")
    parser.add_argument("--holopano-root", help="path to the holo-pano repo (if not on PYTHONPATH)")
    args = parser.parse_args()

    if args.holopano_root:
        sys.path.insert(0, os.path.abspath(args.holopano_root))
    try:
        import numpy as np

        from holopano.data.classes import classname
        from holopano.geometry.boxes import world_box_params
    except ImportError as error:  # pragma: no cover - environment problem, not logic
        print(f"Cannot import holopano/numpy ({error}). Run inside the holo-pano env, "
              f"or pass --holopano-root.", file=sys.stderr)
        return 2

    pred_root = os.path.abspath(args.pred)
    export_root = os.path.abspath(args.det_export)
    out_dir = os.path.abspath(args.out)
    pred_out = os.path.join(out_dir, "pred")
    gt_out = os.path.join(out_dir, "gt")
    os.makedirs(pred_out, exist_ok=True)
    os.makedirs(gt_out, exist_ok=True)

    written = missing = 0
    manifest_path = os.path.join(out_dir, "manifest.jsonl")
    with open(manifest_path, "w", encoding="utf-8") as manifest:
        for scene, cam in _entries(pred_root, args.stage1, args.split):
            export_npz = os.path.join(export_root, scene, cam + ".npz")
            prediction_npz = os.path.join(pred_root, scene, cam + ".npz")
            if not (os.path.exists(export_npz) and os.path.exists(prediction_npz)):
                missing += 1
                continue
            sample_id = f"{scene}-{cam}"

            with np.load(export_npz) as export:
                cam3d2world = export["cam3d2world"]
                gt_objects = []
                for index in range(len(export["gt_centroid"])):
                    centroid, size, yaw = world_box_params(
                        export["gt_centroid"][index],
                        export["gt_basis"][index],
                        export["gt_size"][index],
                        cam3d2world,
                    )
                    gt_objects.append(
                        {
                            "object_id": f"gt{index}",
                            "label": classname(int(export["gt_label"][index])),
                            "score": 1.0,
                            "bbox": _oriented_box(centroid, size, yaw),
                            "attributes": {},
                        }
                    )

            with np.load(prediction_npz) as prediction:
                pred_objects = []
                for index in range(len(prediction["pred_centroid"])):
                    score = float(prediction["det_score"][index])
                    if score < args.score_threshold:
                        continue
                    pred_objects.append(
                        {
                            "object_id": f"det{index}",
                            "label": classname(int(prediction["det_label"][index])),
                            "score": score,
                            "bbox": _oriented_box(
                                prediction["pred_centroid"][index],
                                prediction["pred_size"][index],
                                float(prediction["pred_yaw"][index]),
                            ),
                            "attributes": {},
                        }
                    )

            for path, objects in (
                (os.path.join(gt_out, sample_id + ".json"), gt_objects),
                (os.path.join(pred_out, sample_id + ".json"), pred_objects),
            ):
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(
                        {"layout": None, "objects": objects, "relations": [],
                         "metadata": {"scene": scene, "camera": cam}},
                        handle,
                        ensure_ascii=False,
                    )

            input_path = (
                os.path.join(os.path.abspath(args.rgb_root), scene, cam, "rgb.png")
                if args.rgb_root
                else export_npz
            )
            manifest.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "input": os.path.relpath(input_path, out_dir),
                        "ground_truth": os.path.relpath(
                            os.path.join(gt_out, sample_id + ".json"), out_dir
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1

    print(f"Done: {written} samples in {manifest_path}"
          + (f" (skipped {missing} without prediction/export)" if missing else ""))
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
