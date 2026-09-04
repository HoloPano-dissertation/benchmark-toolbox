# Results: iGibson, one protocol

A comparison of the methods on the **iGibson test split (500 scenes)** under one toolbox
protocol. Protocol: ORIENTED 3D-IoU of objects at threshold **0.15**; layout — a shared
HorizonNet (axis-aligned 3D-IoU); physics — collision (object↔object), violation and
penetration (object↔layout, 5% volume tolerance). **Box metrics do not separate shape
(LDIF vs MGN)** — that requires the shape metric (Chamfer), see the dedicated section.
Reproduction — [REPRODUCING.md](REPRODUCING.md).

Every column is produced by **the same code** and differs only in the DeepPanoContext
inference configuration. Two routes lead to these numbers — driving the model inside its
isolated environment, or re-scoring the converted predictions — and they were verified
to agree on all 500 scenes.

> **Class set.** The detection numbers below average AP over **every** label present in the
> ground truth. The published DPC / PanoContext-Former mAPs (52.69 / 67.35) average over
> **11 categories**, so they belong in a different column and must not be read against these
> ones. To produce rows on the published protocol — including per-class AP — use the
> `classes` parameter (`configs/examples/dpc_igibson_pcf.yaml` for the DPC control row,
> `configs/examples/holopano_pcf_rescore.yaml` for HoloPano); see [metrics.md](metrics.md)
> and [REPRODUCING.md](REPRODUCING.md).

## Summary table (500 scenes, mean)

| Metric | DPC@100 (release) | DPC@20 (stab.) | Im3D-Pano (abl.) | Im3D-Pano (trained) | Total3D-Pano |
|---|---:|---:|---:|---:|---:|
| object_map_dataset ↑ | 0.292 | **0.356** | 0.293 | 0.224 | 0.224 |
| object_map (per-scene) ↑ | 0.314 | **0.381** | 0.279 | 0.227 | 0.227 |
| layout_iou_3d, mean ↑ | 0.748 | 0.897 | 0.897 | 0.897 | 0.897 |
| layout_iou_3d, median ↑ | 0.919 | 0.926 | 0.926 | 0.926 | 0.926 |
| collision_rate (obj↔obj) ↓ | 0.0044 | 0.0077 | 0.0173 | 0.0227 | 0.0227 |
| layout_violation@0.05 ↓ | 0.413 \* | 0.501 | 0.377 | 0.799 | 0.799 |
| layout_penetration ↓ | 0.258 \* | 0.298 | 0.220 | 0.576 | 0.576 |

- **Weights:** DPC and Im3D-Pano (ablation) — authors'; Im3D-Pano (trained) and Total3D-Pano — **trained by us** (the shape head).
- **Relation/GCN:** on for DPC; for Im3D-Pano (ablation) the optimization is off but **the GCN is present**; for Im3D-Pano (trained) and Total3D-Pano **the GCN is absent entirely**.

\* For DPC@100 these two metrics are **deflated** (86 scenes with degenerate boxes are
skipped by the counter); the honest object↔layout numbers for DPC are in the DPC@20
column (0 degenerate scenes).

## Key observations

1. **Layout is a shared control.** `layout_iou_3d` agrees to six decimals across all
   stable configs (0.897414 / 0.926493) — the same HorizonNet estimate. It is not bitwise
   equal: each configuration was a separate inference run, and the saved room corners
   differ in the sixth decimal (float non-determinism on the GPU), which moves
   `layout_iou_3d` only at the seventh significant digit. The comparison is therefore
   controlled (as in the DPC authors' protocol). DPC@100's .748 is a divergence artifact,
   not "it estimates the room worse".

2. **LDIF ≡ MGN on boxes — empirical proof of controllability.** Im3D-Pano (trained,
   LDIF) and Total3D-Pano (MGN) **agree to 6 decimals on ALL box metrics**, and the
   underlying object boxes are **exactly identical in all 500 scenes** (verified
   file-by-file over the predictions of both runs; the metrics that depend on objects
   alone — `object_map`, `object_map_dataset`, `collision_rate` — are therefore equal to
   the last bit, while the two layout-dependent ones inherit the sixth-decimal layout
   noise of observation 1). Boxes come
   from the shared BEN, and without relation the shape head does not affect the box. So
   **box metrics physically cannot separate shape** → LDIF vs MGN is judged by the **shape
   metric** (Chamfer, below). This also proves the trunk is shared not in words but to 6
   decimals.

3. **Contribution of relation optimization** (DPC@20 vs Im3D-Pano ablation): higher
   detection (object_map_dataset 0.356 vs 0.293), fewer object↔object collisions (0.0077 vs
   0.0173), **but** higher penetration (0.298 vs 0.220) — a clear trade-off.

4. **Role of relation/GCN on boxes.** Removing the GCN entirely (the trained rows) →
   detection drops (0.224 vs 0.293 for the ablation) and penetration rises (0.576 vs
   0.220): the GCN refinement physically edits the boxes (pushes objects into the room).
   So the trained rows **cannot be compared to the ablation directly** — the ablation has
   the GCN present; the clean pair for shape is the trained rows against each other.

5. **Instability of the released DPC config — an honest-reproduction finding.**

   | optimize_steps | degenerate layouts (of 500) | object_map_dataset |
   |---:|---:|---:|
   | 100 (release) | **86** | 0.292 |
   | 20 | **0** | 0.356 |

   The instability **nullifies the method's own gain** (0.292 ≈ the no-relation 0.293);
   dropping to 20 steps removes the degeneracy and restores the gain (0.356). Tunable, not
   a property of the method.

## Shape comparison (LDIF vs MGN): the shape metric

Since the box metrics of Im3D-Pano (trained) and Total3D-Pano are **identical**, the only
meaningful difference between the methods is **mesh quality**. The metrics **`mesh_chamfer`**
(symmetric Chamfer, predicted mesh ↔ GT mesh) and **`mesh_fscore`** (surface F-score) are
implemented in the toolbox core (pure stdlib, matching by 3D-IoU 0.15, KD-tree, surface
sampling, `normalize: canonical` — each mesh normalized independently, centre + unit
diagonal); the protocol and commands are in [shape_metric.md](shape_metric.md).

**Result (500 iGibson test scenes, 462 with matched pairs, `normalize: canonical`):**

| Method | Chamfer ↓ (mean / median) | F-score ↑ (mean / median) |
|---|---:|---:|
| **Total3D-Pano (MGN, explicit mesh)** | **0.00445** / 0.00245 | **0.939** / 0.983 |
| **Im3D-Pano (LDIF, implicit)** | 0.00492 / 0.00302 | 0.930 / 0.967 |

**MGN is significantly more accurate than LDIF on shape.** Since the methods share boxes
(only the head differs), the comparison is **paired** per scene (much lower variance than
independent CIs): Chamfer(MGN−LDIF) = **−0.00047**, 95% CI **[−0.00070, −0.00023]** (CI<0;
MGN better in 304/462=66% of scenes); F-score(MGN−LDIF) = **+0.0086**, 95% CI **[+0.0042,
+0.0128]** (CI>0). Both CIs **exclude 0** → the difference is significant. **This is the
discriminator that box metrics cannot provide:** the explicit-mesh head (MGN) reconstructs
object shape more accurately than the implicit one (LDIF) in a controlled setup (shared
trunk, no GCN). Export of canonical predicted meshes (patch `DPC_PATCH_EXPORT_OBJ_MESH`) +
watertight GT via `model_path` — see [shape_metric.md](shape_metric.md).

## Honest-reproduction caveats

- **Im3D-Pano (ablation)** = DPC with relation optimization off (the GCN present,
  authors' weights) — a clean measurement of the relation contribution against DPC.
- **Im3D-Pano (trained)** and **Total3D-Pano** = our from-scratch shape-head training,
  **without the GCN**.
- Comparisons are made **within pairs of equal weight provenance** (see the "ladder" in
  the paper); a change of provenance is a separate control row.
- layout_iou is a shared input, not a method-discriminating metric.

## How to reproduce each column

Inference on GPU once (the `Pano3D` environment), then either route — through the
environment manager, or by re-scoring the converted predictions. Both were verified to
produce these numbers identically on all 500 scenes. Details in
[REPRODUCING.md](REPRODUCING.md).

| Column | Inference config | Through the environment | Re-scoring |
|---|---|---|---|
| DPC@100 | `pano3d_igibson.yaml` (`relation_adjust True`, `optimize_steps 100`) | `dpc_igibson.yaml` | `dpc_igibson_rescore.yaml` |
| DPC@20 | `pano3d_igibson.yaml` (`optimize_steps 20`) | `dpc_s20_igibson.yaml` | `dpc_s20_rescore.yaml` |
| Im3D-Pano (ablation) | `pano3d_igibson.yaml` (`relation_adjust False`) | `im3d_igibson.yaml` | `im3d_igibson_rescore.yaml` |
| Im3D-Pano (trained) | `pano3d_igibson_im3d.yaml` (trunk + our LDIF, no GCN) | `im3d_trained_igibson.yaml` | `im3d_trained_rescore.yaml` |
| Total3D-Pano | `pano3d_igibson_total3d.yaml` (trunk + our MGN, no GCN) | `total3d_igibson.yaml` | `total3d_igibson_rescore.yaml` |

## Status

- **DPC** — reproduced (@100 + @20). ✅
- **Im3D-Pano (ablation)** — done. ✅
- **Im3D-Pano (trained, LDIF)** — trained + scored. ✅
- **Total3D-Pano (MGN)** — trained + scored. ✅
- **Shape metric (Chamfer/F-score, LDIF vs MGN)** — scored, MGN significantly better. ✅
