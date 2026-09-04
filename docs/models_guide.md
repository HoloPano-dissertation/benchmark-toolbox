# Methods and how to run them

The three paper methods — **DPC**, **Im3D-Pano**, **Total3D-Pano** — are
**configurations of one DeepPanoContext code base**, not separate repositories. They
share the detector (detectron2), the layout estimator (HorizonNet), and the 3D boxes
(BEN), differing only in the shape head and the Scene-GCN relation optimization. So all
three produce the same `data.pkl`, which the single converter `dpc_runner.convert_data_pkl`
maps to `SceneOutput`, and the metrics are computed under one protocol.

| Method | Shape | Relation Scene-GCN | How to obtain | Status |
|---|---|---|---|---|
| **DPC** | LDIF | on | inference on released weights | ✅ reproduced |
| **Im3D-Pano** | LDIF | **off** (`relation_adjust False`) | DPC ablation + LDIF trained from scratch | ✅ scored (ablation and trained) |
| **Total3D-Pano** | **MGN** | off | MGN head in the DPC code base + training | ✅ trained + scored |
| echo / fixture | — | — | demo without a model | ✅ for core checks |

All rows are produced and summarized in [results.md](results.md) (+ the Chamfer/F-score
shape metric); the trained-head weights are in a GitHub Release (see [README](../README.en.md)).

---

## Core demo checks (no GPU)

**fixture** — the fastest `config → dataset → metrics → report` check on ready-made
predictions from a directory, with no environment or subprocess:

```bash
benchmark-toolbox run --config configs/examples/smoke.json
```

**echo** — the end-to-end "environment manager → subprocess runner → metrics" path on an
empty venv (the runner returns a fixed scene). A baseline for isolating problems: if DPC
breaks, run echo to confirm the issue is in the model environment, not the core.

```bash
benchmark-toolbox env prepare --env configs/environments/echo.yaml
benchmark-toolbox run --config configs/examples/echo_local.yaml
```

---

## The three methods, concretely

**DPC** — the full method: LDIF shape head, Scene-GCN relation optimization on, the
authors' released weights. **Im3D-Pano** — the same code and the same weights with the
relation optimization off (`--model.scene_gcn.relation_adjust False`); the on-disk DPC
predictions were computed with it on, so this row needs its own inference run into its
own output root. **Total3D-Pano** — the shared trunk with the MGN head instead of LDIF
and no Scene-GCN; MGN lives in the DeepPanoContext code base and was trained by us.

Commands for each row are in [REPRODUCING.md](REPRODUCING.md); the numbers are in
[results.md](results.md).

### Debugging the converter without inference

If a `data.pkl` already exists, the runner will convert just that one:

```bash
conda run -n Pano3D python runners/dpc_runner.py \
  --request <request.json> --output <out.json> \
  --data-pkl path/to/out/pano3d/<run>/visualization/<scene>/data.pkl
```

### Gotchas that cost real time

- DPC needs an old stack (torch 1.7.1+cu110, Python 3.7); `configs/environments/dpc.yaml`
  unpins the authors' hard pins for it. `scripts/dpc_postsetup.py` applies the source
  patches idempotently.
- System packages (`xvfb`, `ninja-build`, ...) are not installed automatically — they
  need an administrator. `env prepare --dry-run` lists them.
- Do **not** strip the `.module.` prefix from checkpoint keys: the model wraps its
  subnets in `DataParallel` and expects it. Without it the bdb3d/shape/layout weights
  silently fail to load and the boxes come out as garbage.
- Torchvision backbones are declared as checkpoints in the spec, so they are staged
  rather than downloaded at the first forward pass — which matters on a node with no
  network.

## Out of the paper's scope

- **HoloPano** — the dissertation's own method (a DPC fork), maintained separately;
  `runners/holopano_runner.py:run_inference()` currently raises `NotImplementedError`. Its
  metrics path is checked with the mock config `configs/examples/holopano_mock.yaml`
  (fixture predictions, no model) — which also serves as an example of how to plug in a new
  model (see [metrics.md](metrics.md#adding-your-own-model)).

---

## Choosing a config

Every method has two configs, and they are two routes to ONE protocol, not two
protocols: the `*_rescore.yaml` pair scores predictions already converted to JSON
(`model.type: fixture`, no model environment involved), while the `*_igibson.yaml` pair
drives the model itself through the environment manager (`model.type: dpc`). Both were
verified to give identical numbers on all 500 scenes.

| Goal | Command |
|---|---|
| Check the core in seconds | `run --config configs/examples/smoke.json` |
| Check the environment manager locally | `run --config configs/examples/echo_local.yaml` |
| DPC row (re-scoring) | `run --config configs/examples/dpc_igibson_rescore.yaml` |
| Im3D-Pano row (re-scoring) | `run --config configs/examples/im3d_igibson_rescore.yaml` |
| Total3D-Pano row (re-scoring) | `run --config configs/examples/total3d_igibson_rescore.yaml` |
| DPC row (through the environment) | `run --config configs/examples/dpc_igibson.yaml` |
| Im3D-Pano row (through the environment) | `run --config configs/examples/im3d_igibson.yaml` |
| Total3D-Pano row (through the environment) | `run --config configs/examples/total3d_igibson.yaml` |
| Im3D-Pano trained / DPC@20 | `im3d_trained_igibson.yaml` / `dpc_s20_igibson.yaml` |
| Real DPC inference | see [REPRODUCING.md](REPRODUCING.md) |

### Why there is no runner per method

The three methods are configurations of one code base and emit the same `data.pkl`, so
they share one environment spec and one runner (`runners/dpc_runner.py`). What selects
the method is the invocation, passed from the experiment config:

```yaml
model:
  type: dpc
  parameters:
    environment: ../environments/dpc.yaml
    runner_args: ["--pred-root", "out/pano3d_im3d", "--model", "im3d-pano",
                  "--relation-adjust", "False"]
```

`--pred-root` is the method selector when scoring existing inference: each configuration
writes its predictions to its own root (`out/pano3d` for DPC, `out/pano3d_im3d`,
`out/pano3d_total3d`, ...), which is also what keeps two methods' predictions from being
mixed into one row. `--relation-adjust` and `--config` matter only when the runner has
to produce the inference itself. The chosen arguments are recorded in `run.json`, so a
row can be traced back to the configuration that produced it.

Adding a genuinely different method — one that is not a DPC configuration — means a new
runner plus its own environment spec, and nothing in the core changes; `runners/echo_runner.py`
is the minimal worked example and `runners/holopano_runner.py` the realistic skeleton.
