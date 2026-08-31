# Metrics and extending the toolbox

Every metric is computed by **the same core code** from a single representation,
`SceneOutput = <layout, objects, relations>` (see [architecture.md](architecture.md)),
so numbers across methods are comparable. A metric is enabled in the config via the
`metrics:` list (see [usage.md](usage.md)); each takes its own `parameters` block.

```yaml
metrics:
  - type: object_map_dataset
    parameters: { iou_threshold: 0.15 }
  - type: mesh_chamfer
    parameters: { iou_threshold: 0.15, num_points: 2048, normalize: canonical }
```

## What each metric reports

Notation: **↑** higher is better, **↓** lower is better. "GT" — whether ground truth is
required (`requires_ground_truth`); metrics without GT measure self-consistency of the
prediction. "Level" — per-scene (averaged over scenes) or dataset (computed once over
the whole set). Implementation: `src/benchmark_toolbox/metrics/`.

### Object detection (3D)

| `type` | What it measures | Dir. | GT | Level | Parameters (default) |
|---|---|:--:|:--:|---|---|
| `object_map` | per-scene mAP over objects at **oriented 3D-IoU**; AP per class within a scene, averaged over scenes | ↑ | yes | scene | `iou_threshold` (0.5), `classes`, `class_map`, `name` |
| `object_map_dataset` | dataset-level mAP: the per-class PR curve is built over **all** scenes (COCO/VOC style) — the headline detection number | ↑ | yes | dataset | `iou_threshold` (0.5), `classes`, `class_map`, `name` |

Matching predictions to ground truth is greedy by descending `score`, each label is
matched once; a hit requires IoU ≥ `iou_threshold`. **IoU is oriented**: `OrientedBox3D`
computes the intersection of rotated rectangles on the floor plane (Sutherland–Hodgman)
× the vertical overlap. The paper protocol uses a threshold of **0.15** (not 0.5, which
would be incomparable with the publications for these methods).

#### The class set of a protocol (`classes`, `class_map`, `name`)

By default mAP is averaged over **every** label present in the ground truth. The
publications average over a **fixed category list** — 11 categories for the DPC /
PanoContext-Former iGibson protocol (`chair, sofa, table, fridge, sink, door, floor_lamp,
bottom_cabinet, top_cabinet, sofa_chair, dryer`) — so the two numbers answer different
questions and must not share a table.

| Parameter | Meaning |
|---|---|
| `classes` | whitelist of evaluation classes; everything else is dropped |
| `class_map` | rename raw label → evaluation class (several source labels may collapse into one, e.g. `office_chair, stool → chair`); labels absent from the map are dropped |
| `name` | row name in the summary (default: the metric type) |

Filtering removes objects from **both** sides before matching, so an out-of-protocol
detection is not counted as a false positive. Under a class protocol a scene holding no
in-protocol ground truth returns NaN (**excluded** from the average) instead of a 0 that
would deflate the mean; without a protocol the historical 0.0 is kept.

`name` exists because summary rows are keyed by the metric name: giving each entry its own
name is what lets a single run report **per-class AP** next to the headline mAP (the
evaluator rejects duplicate names instead of silently overwriting a row).

Writing those rows out by hand means a dozen near-identical blocks, and a category that
drifts between two configs quietly makes their numbers incomparable — so `per_class: true`
expands ONE entry into a `<name>_<class>` row per class, over the very list the headline
mAP uses. The ready protocol covering all 11 categories is
`configs/protocols/igibson_pcf.yaml`.

```yaml
metrics:
  - type: object_map_dataset          # headline: mAP over the protocol
    parameters: &pcf
      iou_threshold: 0.15
      classes: [chair, sofa, table, fridge, sink, door, floor_lamp,
                bottom_cabinet, top_cabinet, sofa_chair, dryer]
  - type: object_map_dataset          # -> ap_chair, ap_sofa, ... (one row per class)
    parameters:
      <<: *pcf
      per_class: true
      name: ap
```

### Layout

| `type` | What it measures | Dir. | GT | Level | Parameters |
|---|---|:--:|:--:|---|---|
| `layout_iou_3d` | 3D-IoU of predicted vs. ground-truth room layout | ↑ | yes | scene | — |

For all methods the layout comes from a shared HorizonNet → it is a **shared control**,
not a method-discriminating metric (reported for completeness, see [results.md](results.md)).

### Physical plausibility (no ground truth)

| `type` | What it measures | Dir. | GT | Level | Parameters |
|---|---|:--:|:--:|---|---|
| `collision_rate` | fraction of overlapping **object pairs** (object↔object) | ↓ | no | scene | `minimum_intersection` (0.0) |
| `layout_violation_rate` | fraction of objects sticking out of the layout (floor/walls/ceiling) beyond a tolerance | ↓ | no | scene | `tolerance` (0.0) — volume fraction |
| `layout_penetration` | mean fraction of object volume **outside** the room (continuous depth) | ↓ | no | scene | — |

`tolerance` in `layout_violation_rate` ignores millimetre wall contacts (furniture
standing flush); the paper protocol uses 0.05. `layout_penetration` is fairer than the
binary rate for reporting.

### Shape quality (mesh)

| `type` | What it measures | Dir. | GT | Level | Parameters (default) |
|---|---|:--:|:--:|---|---|
| `mesh_chamfer` | symmetric Chamfer distance: predicted object mesh ↔ GT mesh | ↓ | yes | scene | `iou_threshold` (0.15), `num_points` (2048), `seed` (0), `squared` (True), `normalize` (none), `mesh_root` (—) |
| `mesh_fscore` | surface F-score over matched objects at threshold `fscore_threshold` | ↑ | yes | scene | + `fscore_threshold` (0.1) |

Objects are matched by 3D-IoU (as in `object_map`); for each pair the meshes are
surface-sampled and compared. `normalize: canonical` normalizes **each** mesh
independently (bbox centre + unit diagonal) — a fair comparison of heads with different
canonical-frame conventions (needed for LDIF-vs-MGN, see [shape_metric.md](shape_metric.md)).
Shape is passed to an object via `attributes["shape"]` (a path to `.ply`/`.obj`/`.f32`
or inline points). Scenes with no matched pairs → the metric returns NaN and is
**excluded** from aggregation (a detection failure is not reported as a perfect shape).

## Aggregation and artifacts

- **per-scene** metrics: `mean`, `median`, **95% bootstrap CI** over scenes;
- **dataset** metrics (`object_map_dataset`): computed once over all scenes;
- NaN values (metric not applicable to a scene) are excluded from aggregation.

Each run writes to `artifacts/<name>/`: `summary.json` (aggregates), `report.md`
(table), `metrics.jsonl` (per scene), `run.json` (config, seed, git revision).

## Adding your own metric

1. Subclass `BaseMetric` (`src/benchmark_toolbox/metrics/base.py`) and implement
   `compute(prediction, ground_truth) -> float`. For a dataset-level metric, set
   `dataset_level = True` and implement `compute_dataset(predictions, ground_truths)`.
   If ground truth is not needed, set `requires_ground_truth = False` (the metric is then
   also computed on scenes without GT).
2. Register it with the decorator `@BaseMetric.registry.register("my_metric")`.
3. Import it in `metrics/__init__.py` (so the registration runs) — done: the metric is
   available in a config as `type: my_metric`.

```python
from benchmark_toolbox.metrics.base import BaseMetric

@BaseMetric.registry.register("bbox_count_ratio")
class BBoxCountRatio(BaseMetric):
    name = "bbox_count_ratio"
    requires_ground_truth = True
    def __init__(self, parameters=None):
        del parameters
    def compute(self, prediction, ground_truth):
        gt = len(ground_truth.objects) or 1
        return len(prediction.objects) / gt
```

Return `float("nan")` if the metric is not applicable to a particular scene — it is then
excluded from the average (rather than counted as zero).

## Adding your own model

The core does **not** import the model — it runs in its own environment and exchanges
JSON files. Two paths:

- **fixture** (ready-made predictions): convert the model output to `SceneOutput` JSON
  and place it in a directory; config `model.type: fixture`, `prediction_dir: <dir>`.
  Convenient for debugging metrics and for re-scoring.
- **subprocess** (live inference): write a runner per the contract
  ([`runners/README.md`](../runners/README.md)) (it reads `request`, writes `SceneOutput`
  via the `runners/_common.py` helpers), and describe the environment in
  `configs/environments/<env>.yaml`. An end-to-end example is `echo` (a GPU-free demo):
  `runners/echo_runner.py` + `configs/environments/echo.yaml`. All three paper methods
  share one converter, `runners/dpc_runner.py` (common `data.pkl` format).

Extension points are registered in registries (`registry.py`): `BaseSceneEstimator`,
`BaseDatasetLoader`, `BaseMetric`, `BaseEnvironmentManager` — a new component is plugged
in **without changing the core**. Details: [architecture.md](architecture.md).
