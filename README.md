# Benchmark Toolbox

A single, **reproducible** framework for running and comparing holistic indoor-scene
understanding methods from one 360° panorama. The accompanying paper compares three
methods — **DeepPanoContext (DPC)**, **Im3D-Pano**, and **Total3D-Pano** — on the
**iGibson** dataset under one evaluation protocol.

Key idea: the three methods are **configurations of a single DeepPanoContext code
base**, not three separate repositories. They share a detector, a layout estimator
(HorizonNet), and a 3D box estimator (BEN), and differ only in the shape head (LDIF vs.
MGN) and the optional Scene-GCN relation optimization. The toolbox therefore runs them
through one path and maps every output to a single format,
`<layout, objects, relations>`.

## Getting started

```bash
make setup    # creates .venv and installs the package with test deps (needs internet once)
make test     # unit and smoke tests — should finish with "passed"
make smoke    # demo benchmark (no GPU, no models)
```

`make smoke` prints `Evaluated 1 scene(s). Artifacts: .../artifacts/smoke` and writes
the report to `artifacts/smoke/report.md` (perfect metrics on a fixture scene:
`layout_iou_3d=1.0`, `object_map=1.0`, `collision_rate=0.0`).

Processed MIDI-3D/3D-FRONT GLB rooms can be converted into equirectangular RGB,
depth, normal, and instance-segmentation panoramas with the standalone
[3D-FRONT panorama renderer](tools/3dfront_panorama_renderer/README.md).

Without `make` (standard library only, no install):

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m benchmark_toolbox run --config configs/examples/smoke.json
```

## What the benchmark does

A single `benchmark-toolbox run --config <cfg>` runs the chain:

```text
config -> dataset (manifest + ground truth) -> estimator (model) -> metrics -> report
```

The same metric code is applied to every method, so the comparison rows are
commensurable. Each run writes to `artifacts/<name>/`:

- `predictions/<scene>.json` — unified predictions;
- `metrics.jsonl` — per-scene values;
- `summary.json` — aggregates (mean / median / 95% bootstrap CI);
- `report.md` — a human-readable table;
- `run.json` — config, seed, git revision, and environment provenance (for reproducibility).

**Example (no GPU, any OS):**

```bash
benchmark-toolbox run --config configs/examples/smoke.json          # fixture prediction -> metrics in seconds
benchmark-toolbox run --config configs/examples/echo_local.yaml     # end-to-end path through the environment manager (venv echo)
```

**Example of a paper row.** Inference runs once on a GPU; scoring it is then either of
two routes, which produce identical numbers:

```bash
# through the isolated model environment (the framework's own path)
benchmark-toolbox run --config configs/examples/dpc_igibson.yaml
# or by re-scoring the converted predictions — no GPU, no model, any OS
benchmark-toolbox run --config configs/examples/dpc_igibson_rescore.yaml
```

See [docs/REPRODUCING.md](docs/REPRODUCING.md) for the full reproduction recipe.

## The three methods

| Method | What differs (within the DPC code base) | Shape | Relation Scene-GCN |
|---|---|---|---|
| **DPC** | full method | LDIF | on |
| **Im3D-Pano** | DPC ablation: relation optimization off | LDIF | off |
| **Total3D-Pano** | Im3D-Pano with MGN instead of LDIF | MGN | off |

How to run each is in [docs/models_guide.md](docs/models_guide.md).

## Metrics and protocol

The comparison uses **oriented 3D-IoU of objects at threshold 0.15** (not 0.5, which
would be incomparable with the published numbers for these methods). Reported metrics:
`layout_iou_3d`, per-scene `object_map`, dataset-level `object_map_dataset`, and
physical plausibility — `collision_rate` (object↔object), `layout_violation_rate` and
`layout_penetration` (object↔layout). Shape quality (which separates LDIF and MGN —
indistinguishable on boxes) — `mesh_chamfer` and `mesh_fscore` (predicted mesh ↔ GT
mesh), see [docs/shape_metric.md](docs/shape_metric.md). **A description of every metric
and how to add your own metric/model** — [docs/metrics.md](docs/metrics.md).

## Trained weights and reproducing the results

Every number in the paper is reproducible.

- **Weights trained by us** (shape heads, trained on iGibson, A100):
  - `total3d-pano_mgn_igibson.pth` — MGN (`DensTMNet`) for **Total3D-Pano** (config `configs/dpc/mgnet_igibson.yaml`);
  - `im3d-pano_ldif_igibson.pth` — LDIF (LIEN+LDIF) for **Im3D-Pano (trained)** (config `configs/dpc/ldif_igibson_scratch.yaml`).
- **DPC released weights** (detector, HorizonNet, BEN/Scene-GCN) — public, from the DeepPanoContext release; not redistributed here (the assembly recipe is in the release `MANIFEST.md`).

📦 **Download weights:** [GitHub Release `weights-v1`](https://github.com/HoloPano-dissertation/benchmark-toolbox/releases/tag/weights-v1) — `total3d-pano_mgn_igibson.pth`, `im3d-pano_ldif_igibson.pth` (+ `SHA256SUMS.txt`, `MANIFEST.md`).

**How to reproduce:**
1. Box metrics (5 rows) — either through the model's isolated environment or by re-scoring the converted predictions without a GPU; both give the same numbers. See [docs/REPRODUCING.md](docs/REPRODUCING.md) and the per-column table in [docs/results.md](docs/results.md).
2. Shape metric (Chamfer/F-score, MGN vs. LDIF): export canonical predicted meshes (`--mode test`) + watertight GT via `model_path`, then `mesh_chamfer`/`mesh_fscore` with `normalize: canonical` — the full recipe is in [docs/shape_metric.md](docs/shape_metric.md).

Final numbers are in [docs/results.md](docs/results.md).

## Architecture principle

The core works with a single output representation:

```text
SceneOutput = <layout, objects, relations>
```

Each model implements the `BaseSceneEstimator.predict(sample)` contract but is **not
imported** into the toolbox process: an adapter runs it in a separate environment and
exchanges JSON files with the runner process. This lets models use incompatible
versions of Python, PyTorch, and CUDA. Extension points (`BaseSceneEstimator`,
`BaseDatasetLoader`, `BaseMetric`, `BaseEnvironmentManager`) are registered in a
registry — a new method is plugged in without touching the core. Diagrams are in
[docs/architecture.md](docs/architecture.md).

For network-restricted hosts (e.g. an offline HPC node), set
`BENCHMARK_TOOLBOX_ARTIFACTS=<dir>` so `env prepare` uses pre-staged weights/backbones
from that directory instead of downloading them (see [docs/usage.md](docs/usage.md)).

## Available run configs

| Config | What it does | Requires |
|---|---|---|
| `configs/examples/smoke.json` | metrics on a fixture prediction | nothing (any OS, no GPU) |
| `configs/examples/echo_local.yaml` | end-to-end path through the environment manager (venv echo) | nothing |
| `configs/examples/*_igibson_rescore.yaml` | a paper row from converted predictions | converted predictions (no GPU) |
| `configs/examples/{dpc,im3d,total3d,...}_igibson.yaml` | the same row through the model environment | Linux + conda + that method's inference on disk |
| `configs/examples/dpc_igibson_pcf.yaml` | the 11-category control row against published mAP | converted predictions (no GPU) |
| `configs/examples/shape_chamfer_rescore.yaml` | shape metric, LDIF vs MGN | exported meshes (no GPU) |

Every method config inherits its metric list from `configs/protocols/` via `extends`, so
the comparison rows are produced by one protocol definition instead of copies of it.

## Layout

```text
benchmark-toolbox/
├── configs/                 experiment and model-environment configs
│   ├── protocols/           the shared evaluation protocols every method inherits
│   ├── examples/            ready-to-run configs (smoke, echo, *_igibson_rescore, ...)
│   └── environments/        isolated model-environment specs (dpc, echo, holopano)
├── data/examples/           manifests and small test data
├── docs/                    documentation (see below)
├── runners/                 model adapters for isolated environments (contract + dpc/echo)
├── scripts/                 prediction batch-conversion, GT/manifest building, DPC patches
├── src/benchmark_toolbox/   Python core implementation
├── tests/                   unit and smoke tests
└── pyproject.toml
```

## Documentation

- [docs/usage.md](docs/usage.md) — how to run: configs, datasets, environments, artifacts.
- [docs/metrics.md](docs/metrics.md) — **a description of every metric + how to add your own metric and model** (extensibility).
- [docs/architecture.md](docs/architecture.md) — components, contracts, extension points.
- [docs/results.md](docs/results.md) — final paper numbers (iGibson, 500 scenes) + findings.
- [docs/shape_metric.md](docs/shape_metric.md) — shape metric (Chamfer/F-score) and mesh export.
- [docs/REPRODUCING.md](docs/REPRODUCING.md) — how to reproduce the paper numbers.
- [docs/models_guide.md](docs/models_guide.md) — the three methods as configurations of one code base.

## License

MIT, see [LICENSE](LICENSE).
