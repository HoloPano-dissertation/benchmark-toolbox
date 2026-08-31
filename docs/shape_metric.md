# Shape metric: Chamfer distance and F-score (LDIF vs MGN)

Why it is needed. Without relation optimization, **Im3D-Pano (LDIF)** and
**Total3D-Pano (MGN)** produce **exactly the same object boxes** — verified file by file
over all 500 scenes, because the trunk (detector + HorizonNet + BEN) is shared and the
shape head never touches the box. No box metric (`object_map`, `collision_rate`,
`layout_penetration`) can therefore tell them apart — see [results.md](results.md). The only
meaningful difference is **mesh quality**, which these two core metrics measure:

| Metric | What | Direction |
|---|---|---|
| `mesh_chamfer` | symmetric Chamfer distance: predicted object mesh ↔ GT mesh | ↓ lower is better |
| `mesh_fscore`  | surface F-score at threshold `fscore_threshold` | ↑ higher is better |

Both are implemented in [`src/benchmark_toolbox/metrics/shape.py`](../src/benchmark_toolbox/metrics/shape.py)
in **pure stdlib** (like the rest of the core — no numpy): nearest neighbour via an own
3D KD-tree, surface sampling proportional to triangle areas.

## How it is computed

1. **Matching.** Predicted objects are greedily matched to ground truth by **oriented
   3D-IoU** (protocol threshold `iou_threshold=0.15`) — the same matching as `object_map`
   (`geometry._match_scene`). Shape is computed **only over matched pairs**: shape quality
   is meaningful only for detected objects; detection errors are caught by `object_map`.
2. **Points.** For each object in a pair, its shape point cloud is obtained (see "Shape
   contract").
3. **Chamfer / F-score.** `chamfer = mean_i d(pred_i→gt) + mean_j d(gt_j→pred)` (squared
   L2 by default, `squared=True`); F-score is the fraction of points within
   `fscore_threshold`. Both come from ONE nearest-neighbour pass.
4. **Aggregation.** Per scene — the mean over its matched pairs. If there are no matched
   pairs with shape, the metric returns **NaN** and the `Evaluator` **excludes** the scene
   from aggregation (it does not count it as 0: a Chamfer of 0 means "perfect", which would
   bias the mean). The result is `mean/median/95% CI` (bootstrap) over scenes, like other
   per-scene metrics.

### Comparison frame (IMPORTANT)

Chamfer is computed **as is** in the frame of the stored points, so the predicted and
ground-truth mesh of one object must lie in **one canonical object frame**. This is
natural: `extract_mesh()` returns the mesh in the network's normalized frame (before the
object is placed into the scene), and the ground-truth `mesh_watertight.ply` is
canonical; the DPC reference `external.ldif.inference.metrics.mesh_chamfer_via_points`
compares them the same way, without re-normalization.

If, on inspection, the frames turn out to be misaligned in scale/centre, enable
`normalize: unit` — the metric centres both shapes at the GT centroid and divides by the
GT box diagonal (Chamfer is then in GT-normalized units). **Validate the alignment on
2–3 objects** before a full run (e.g. dump a pair's points and eyeball the overlap).

For the LDIF-vs-MGN comparison the two heads use a different canonical convention (MGN
emits metric-scale meshes, LDIF unit-scale), so use `normalize: canonical`, which
normalizes each mesh independently (bbox centre + unit diagonal) and isolates shape from
the scale that the box already encodes.

### Parameters

`iou_threshold` (0.15) · `num_points` (2048; surface samples per object — more is more
accurate but slower in pure Python; DPC uses 10000) · `seed` (0) · `squared` (True →
L2²) · `fscore_threshold` (0.1; in the units of the point frame) · `normalize`
(`none` | `canonical` | `unit`) · `mesh_root` (prefix for relative shape paths).

## Shape contract (`attributes["shape"]`)

An object references its shape through the free-form field `attributes["shape"]` — **no
domain/serialization change needed**, the field already round-trips through JSON. The
value:

- a **path string** to a file: `.ply` / `.obj` (the surface is sampled on the fly),
  `.f32` / `.bin` (raw little-endian float32, xyz in order, N = size/12), `.json` (a list
  of points);
- a **list** `[[x,y,z], ...]` — points inline in JSON (for tests / small cases);
- a **dict** `{"points": [...]}` or `{"file": "..."}` / `{"mesh": "..."}`.

Relative paths are resolved against `mesh_root` (by default used as is; on a cluster it
is convenient to write absolute paths). If either side has no shape, the pair is skipped.

The helper `scene_from_oriented_objects` (in `runners/_common.py`) passes shape through:
it is enough to put `item["shape"] = "<path>"` (or `item["attributes"]`) into an object
item. `dpc_runner.convert_data_pkl` picks up `item["mesh_path"]` / `item["mesh"]` (if it
is a **string**) into `attributes["shape"]` — that is the injection point for paths at the
export step.

## Running (re-score)

The config template is
[`configs/examples/shape_chamfer_rescore.yaml`](../configs/examples/shape_chamfer_rescore.yaml).
No GPU/conda needed (pure core). The LDIF-vs-MGN comparison = **two runs** of this config
(`pred_im3d` and `pred_total3d`) against a **shared** GT manifest with shape:

```bash
./.venv/bin/python -m benchmark_toolbox run --config configs/examples/shape_chamfer_rescore.yaml
# -> artifacts/shape_chamfer/{summary.json,report.md}
```

## What to do on the cluster to feed it

A plain `qtest` saves **boxes only** (`IGScene.to_pickle` drops the mesh field), so the
shape metric "starves". The recipe (the `Pano3D` environment; `$DPC`/`$TBX` are the DPC
clone / the toolbox):

1. **Mesh-export patch.** `scripts/dpc_postsetup.py` (marker `DPC_PATCH_EXPORT_OBJ_MESH`
   inside `patch_safe_viz`) appends to `visualize_step`: in full mode, export
   `est_scene.mesh_io[i]` (the canonical object mesh, BEFORE `obj2frame` to world — see
   `merge_mesh`, `utils/igibson_utils.py`) to `obj_mesh/<i>.ply` next to `data.pkl`.
   Applied during environment prep; idempotent. _(The native `chamfer_scene` in
   `testing.py` is unusable: it is nested under `if 'relation' in gt_data`, which is off
   for Total3D/Im3D → dead block; and it scores the whole SCENE in world coordinates, not
   per object.)_

2. **Inference with export (GPU).** `--mode test` enables `full` → `extract_mesh` + the
   patch write `obj_mesh/<i>.ply`. The index `<i>` = the objs order in `data.pkl`.
   ```bash
   conda run -n Pano3D python -u main.py configs/pano3d_igibson_total3d.yaml \
     --data.split data/igibson --mode test --log.path out/t3d_export_full
   # Im3D-Pano (LDIF): configs/pano3d_igibson_im3d.yaml, --log.path out/im3d_export_full
   ```

3. **GT with shape.** The manifest builder with `IGIBSON_OBJ_ROOT` attaches each GT
   object's watertight mesh `<root>/<obj['model_path']>/mesh_watertight.ply` (the
   `model_path` key of the IGScene object; the same canonical frame as the predicted
   meshes — `obj2frame` places both sides identically):
   ```bash
   cd $DPC && IGIBSON_OBJ_ROOT=data/igibson_obj conda run -n Pano3D python \
     $TBX/scripts/dpc_build_igibson_manifest.py $DPC $DPC/data/igibson_bench_shape data/igibson/test.json
   ```

4. **Predictions with shape.** Batch-convert with `--obj-mesh-sibling obj_mesh` picks up
   `obj_mesh/<i>.ply` → `attributes['shape']` (via `obj['mesh_path']` in
   `convert_data_pkl`):
   ```bash
   conda run -n Pano3D python $TBX/scripts/dpc_batch_convert.py \
     $DPC/data/igibson_bench_shape/manifest.jsonl $DPC/data/igibson_bench_shape/pred_total3d \
     --repo $DPC --pred-root out/t3d_export_full --obj-mesh-sibling obj_mesh
   ```

5. **Scoring (no GPU).** Run `mesh_chamfer`/`mesh_fscore` (the shared GT manifest with
   shape, one run per method): `$TBX/.venv/bin/python -m benchmark_toolbox run --config <shape-rescore>.yaml`.

The LDIF-vs-MGN comparison = steps 2–5 for both (pred_im3d / pred_total3d). FRAME
VALIDATION (confirmed for MGN on a smoke run: predicted/GT extents match, Chamfer is
small): check extent+Chamfer of the predicted mesh vs. the watertight GT on 2–3 objects;
on a mismatch use `normalize: canonical`.
