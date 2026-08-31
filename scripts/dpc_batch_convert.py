#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-convert DPC predictions")
    parser.add_argument("manifest", help="toolbox manifest.jsonl (sample_id/...)")
    parser.add_argument("out_dir", help="directory for <sample_id>.json")
    parser.add_argument(
        "--repo",
        default=os.getcwd(),
        help="cloned DPC repo root (to locate predictions and unpickle utils.*; "
        "defaults to the current directory)",
    )
    parser.add_argument(
        "--pred-root",
        default=os.path.join("out", "pano3d"),
        help="predictions root (<root>/<run>/visualization/.../data.pkl); "
        "a relative path is resolved from --repo. Defaults to out/pano3d (full DPC). "
        "For Im3D-Pano point it at a separate inference root, e.g. out/pano3d_im3d, "
        "otherwise predictions mix with DPC.",
    )
    parser.add_argument(
        "--obj-mesh-sibling",
        default=None,
        help="name of the subdirectory next to data.pkl holding CANONICAL per-object "
        "pred meshes (<i>.ply, exported by DPC_PATCH_EXPORT_OBJ_MESH under --mode test). "
        "If set, object i's mesh path is written to attributes['shape'] for the shape "
        "metric (mesh_chamfer). Index i matches the order of objs in data.pkl.",
    )
    args = parser.parse_args()

    repo = os.path.abspath(args.repo)
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, repo)
    sys.path.insert(0, os.path.join(here, os.pardir, "runners"))
    import dpc_runner

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.abspath(args.manifest)
    pred_root = args.pred_root
    if not os.path.isabs(pred_root):
        pred_root = os.path.join(repo, pred_root)
    os.chdir(repo)

    predictions = sorted(
        glob.glob(
            os.path.join(pred_root, "*", "visualization", "**", "data.pkl"),
            recursive=True,
        )
    )
    by_id: dict[str, str] = {}
    for path in predictions:
        by_id.setdefault(dpc_runner.prediction_sample_id(path), path)

    base = os.path.basename(out_dir.rstrip("/"))
    manifest_pred = os.path.join(os.path.dirname(out_dir.rstrip("/")), f"manifest_{base}.jsonl")
    converted = missing = errors = 0
    kept: list[str] = []
    for line in open(manifest_path, encoding="utf-8"):
        sample_id = json.loads(line)["sample_id"]
        pkl = by_id.get(sample_id)
        if pkl is None:
            missing += 1
            continue
        try:
            with open(pkl, "rb") as handle:
                data = pickle.load(handle)
            data = data.data if hasattr(data, "data") else data
            if args.obj_mesh_sibling:
                mesh_dir = os.path.join(os.path.dirname(pkl), args.obj_mesh_sibling)
                for index, obj in enumerate(data.get("objs", [])):
                    candidate = os.path.join(mesh_dir, f"{index}.ply")
                    if os.path.exists(candidate):
                        obj["mesh_path"] = candidate
            scene = dpc_runner.convert_data_pkl(data)
            with open(os.path.join(out_dir, sample_id + ".json"), "w", encoding="utf-8") as out:
                json.dump(scene, out, ensure_ascii=False)
            converted += 1
            kept.append(line)
        except Exception as error:  # noqa: BLE001  one broken scene must not kill the batch
            errors += 1
            print(f"[err] {sample_id}: {type(error).__name__}: {error}", file=sys.stderr)

    with open(manifest_pred, "w", encoding="utf-8") as out:
        out.writelines(kept)

    print(f"converted: {converted} | missing: {missing} | errors: {errors}")
    print(f"predictions -> {out_dir}")
    print(f"manifest (converted only) -> {manifest_pred}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
