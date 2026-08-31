#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
import sys
import traceback
from multiprocessing import Pool

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def _mgn_paths(folder: str):
    return (os.path.join(folder, "gt_3dpoints.mgn"),
            os.path.join(folder, "densities.mgn"))


def _process_one(task):
    folder, n_points, k, overwrite = task
    pts_path, dens_path = _mgn_paths(folder)
    if not overwrite and os.path.exists(pts_path) and os.path.exists(dens_path):
        return folder, True, "skip-done"
    ply = os.path.join(folder, "mesh_watertight.ply")
    if not os.path.exists(ply):
        return folder, False, "no mesh_watertight.ply"
    try:
        from utils.mesh_utils import sample_pnts_from_obj

        mesh = trimesh.load(ply, force="mesh", process=False)
        data = {
            "v": np.asarray(mesh.vertices, dtype=np.float64),
            "f": [[str(int(i) + 1) for i in face] for face in np.asarray(mesh.faces)],
        }
        pts = np.asarray(sample_pnts_from_obj(data, n_points, mode="random"), dtype=np.float64)
        pts = pts[:, :3]
        pts.tofile(pts_path)

        tree = cKDTree(pts)
        dists, indices = tree.query(pts, k=k)
        densities = np.array([max(dists[point_set, 1]) ** 2 for point_set in indices],
                             dtype=np.float64)
        densities.tofile(dens_path)
        return folder, True, f"{len(pts)} pts"
    except Exception:
        return folder, False, traceback.format_exc().splitlines()[-1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate gt_3dpoints.mgn + densities.mgn for MGN / Total3D-Pano")
    parser.add_argument("repo", help="cloned DPC (Pano3D) repo root")
    parser.add_argument("--data-dir", default="data/igibson_obj",
                        help="objects directory relative to repo (default data/igibson_obj)")
    parser.add_argument("--n-points", type=int, default=10000,
                        help="points per object (as in process_mgnet)")
    parser.add_argument("--k", type=int, default=30,
                        help="neighbors for the density estimate (as in process_mgnet)")
    parser.add_argument("--processes", type=int, default=8,
                        help="parallel processes (0 = sequential)")
    parser.add_argument("--overwrite", action="store_true",
                        help="regenerate even if .mgn already exists (done ones are skipped by default)")
    parser.add_argument("--limit", type=int, default=0,
                        help="process only the first N (for a quick check)")
    args = parser.parse_args()

    repo = os.path.abspath(args.repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    data_root = os.path.join(repo, args.data_dir)

    folders = sorted(os.path.dirname(p) for p in
                     glob.glob(os.path.join(data_root, "*", "*", "mesh_watertight.ply")))
    if args.limit:
        folders = folders[:args.limit]
    overwrite = args.overwrite
    print(f"objects with mesh_watertight.ply: {len(folders)} (data_root={data_root})")
    if not folders:
        print("NOTHING to process — check --data-dir and that mesh_watertight.ply exists")
        return 1

    tasks = [(f, args.n_points, args.k, overwrite) for f in folders]
    results = []
    if args.processes and args.processes > 0:
        with Pool(processes=args.processes) as pool:
            for i, r in enumerate(pool.imap_unordered(_process_one, tasks), 1):
                results.append(r)
                if i % 50 == 0 or i == len(tasks):
                    print(f"  {i}/{len(tasks)}")
    else:
        for i, t in enumerate(tasks, 1):
            results.append(_process_one(t))
            if i % 50 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}")

    ok = [r for r in results if r[1]]
    fail = [r for r in results if not r[1]]
    print(f"DONE: ok={len(ok)} fail={len(fail)}")
    for folder, _, msg in fail[:20]:
        print(f"  FAIL {os.path.relpath(folder, data_root)}: {msg}")
    return 0 if not fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
