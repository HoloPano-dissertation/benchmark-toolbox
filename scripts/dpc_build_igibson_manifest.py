#!/usr/bin/env python3
from __future__ import annotations

import glob
import json
import os
import pickle
import sys


def _load_split_filter(split_file: str) -> set:
    with open(split_file, encoding="utf-8") as handle:
        entries = json.load(handle)
    allow = set()
    for entry in entries:
        item = str(entry).replace("\\", "/").strip("/")
        if item.startswith("data/igibson/"):
            item = item[len("data/igibson/"):]
        if item.endswith(".pkl"):
            item = item.rsplit("/", 1)[0]
        allow.add(item.replace("/", "-"))
    return allow


def main() -> int:
    if len(sys.argv) not in (3, 4):
        print(
            "usage: dpc_build_igibson_manifest.py <repo-DPC> <out-dir> [split.json]\n"
            "  split.json — e.g. data/igibson/test.json, to take only the test split"
        )
        return 2
    repo = os.path.abspath(sys.argv[1])
    out_dir = os.path.abspath(sys.argv[2])
    allow = _load_split_filter(os.path.abspath(sys.argv[3])) if len(sys.argv) == 4 else None
    if allow is not None:
        print(f"Split filter: {len(allow)} samples")
    shape_root = os.environ.get("IGIBSON_OBJ_ROOT")
    if shape_root:
        shape_root = os.path.abspath(shape_root)
        print(f"GT shape: meshes from {shape_root}/<model_path>/mesh_watertight.ply")

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, os.pardir, "runners"))
    sys.path.insert(0, repo)
    os.chdir(repo)

    import dpc_runner  # noqa: E402  reuse the same conversion as for predictions

    samples = sorted(
        glob.glob(os.path.join(repo, "data", "igibson", "*", "*", "data.pkl"))
    )
    if not samples:
        print(f"No samples found in {repo}/data/igibson/*/*/data.pkl")
        return 1

    gt_dir = os.path.join(out_dir, "gt")
    os.makedirs(gt_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.jsonl")

    written = skipped = 0
    with open(manifest_path, "w", encoding="utf-8") as manifest:
        for pkl in samples:
            sample_dir = os.path.dirname(pkl)
            scene = os.path.basename(os.path.dirname(sample_dir))
            name = os.path.basename(sample_dir)
            sample_id = f"{scene}-{name}"
            if allow is not None and sample_id not in allow:
                continue
            try:
                with open(pkl, "rb") as handle:
                    data = pickle.load(handle)
                data = data.data if hasattr(data, "data") else data
                if shape_root:
                    for obj in data.get("objs", []):
                        model_path = obj.get("model_path")
                        if not model_path:
                            continue
                        mesh = os.path.join(shape_root, model_path, "mesh_watertight.ply")
                        if os.path.exists(mesh):
                            obj["mesh_path"] = mesh
                scene_dict = dpc_runner.convert_data_pkl(data)
            except Exception as error:  # noqa: BLE001  one broken sample must not kill the build
                print(f"[skip] {sample_id}: {type(error).__name__}: {error}")
                skipped += 1
                continue

            gt_path = os.path.join(gt_dir, sample_id + ".json")
            with open(gt_path, "w", encoding="utf-8") as gt_file:
                json.dump(scene_dict, gt_file, ensure_ascii=False)

            manifest.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "input": os.path.relpath(
                            os.path.join(sample_dir, "rgb.png"), out_dir
                        ),
                        "ground_truth": os.path.relpath(gt_path, out_dir),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1

    print(f"Done: {written} samples in {manifest_path}" + (
        f" (skipped {skipped})" if skipped else ""
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
