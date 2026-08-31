# Reproducing the results

Every number in [results.md](results.md) can be re-derived from this repository. There are
two routes to the same numbers, and they were verified to agree on all 500 scenes, to the
last digit:

| Route | What it needs | Time (500 scenes) | Use it to |
|---|---|---|---|
| **A. Through the environment manager** | the built model environment + that method's inference on disk | ~1 min | reproduce a row the way the framework is meant to work |
| **B. Fixture re-scoring** | only the converted predictions (JSON) — produced once by route A's conversion | ~20 s | check the metric code alone, on any OS, without the model or its environment |

**Why a model environment is involved at all, when the metrics need no GPU.** The metrics
are pure standard library and run anywhere — but they score a *unified* representation,
and getting there means reading the model's native output. DeepPanoContext writes each
prediction as a pickle of its own classes: deserializing one imports `utils.*`, which
imports gibson2. The toolbox core has no numpy, let alone gibson2, so it cannot read that
file at all — the conversion has to happen inside the model's environment, GPU or no GPU.
That conversion is the only step that needs it. Once done, everything downstream is both
GPU-free and dependency-free.

Route A does the conversion through the runner and records the environment fingerprint,
repository commit and checkpoint hashes into `run.json`. Route B is the same conversion
done once ahead of time and cached as JSON, which is why it then needs neither the model,
nor its environment, nor a dataset licence — that is what makes it checkable by a reader
who has none of them.

## What you need, and where it comes from

| Input | Where it comes from | Needed by |
|---|---|---|
| This repository | `git clone`, then `make setup` | both routes |
| iGibson dataset | official iGibson/BEHAVIOR site (**licence-gated, not redistributed**), preprocessed as in the DeepPanoContext repo | route A, inference |
| DeepPanoContext code + released weights | `benchmark-toolbox env prepare --env configs/environments/dpc.yaml` (clones and provisions it) | route A |
| Our trained shape heads (MGN, LDIF) | [GitHub Release `weights-v1`](https://github.com/HoloPano-dissertation/benchmark-toolbox/releases/tag/weights-v1) — `total3d-pano_mgn_igibson.pth`, `im3d-pano_ldif_igibson.pth`, `SHA256SUMS.txt` | Total3D-Pano and Im3D-Pano (trained) |
| Inference output (`data.pkl` per scene) | produced once on a GPU, see [Producing the inference](#producing-the-inference) | route A |
| Converted predictions (JSON) + GT manifest | `scripts/dpc_build_igibson_manifest.py` + `scripts/dpc_batch_convert.py` | route B |

## Route A — through the environment manager

Set `$TBX` to this repository and `$DPC` to the provisioned DeepPanoContext clone
(`$TBX/.benchmark_toolbox/repos/Pano3D` after `env prepare`).

```bash
# 0. one-off: the toolbox itself
cd $TBX && make setup

# 1. one-off: the model environment. --dry-run first: it reports what is already in
#    place and what still needs the network, without changing anything.
benchmark-toolbox env prepare --env configs/environments/dpc.yaml --dry-run
benchmark-toolbox env prepare --env configs/environments/dpc.yaml
#    If the environment was built by hand (or by an admin), adopt it instead:
benchmark-toolbox env prepare --env configs/environments/dpc.yaml --adopt
#    Sanity-check the weights on disk at any time:
benchmark-toolbox env prepare --env configs/environments/dpc.yaml --verify

# 2. one-off: ground truth + manifest, built by the SAME converter as the predictions
conda run -n Pano3D python $TBX/scripts/dpc_build_igibson_manifest.py \
    $DPC $DPC/data/igibson_bench $DPC/data/igibson/test.json

# 3. per row: point the config's manifest at $DPC/data/igibson_bench/manifest.jsonl,
#    then run. Each command is one row of the table.
benchmark-toolbox run --config configs/examples/dpc_igibson.yaml           # DPC@100
benchmark-toolbox run --config configs/examples/dpc_s20_igibson.yaml       # DPC@20
benchmark-toolbox run --config configs/examples/im3d_igibson.yaml          # Im3D-Pano (ablation)
benchmark-toolbox run --config configs/examples/im3d_trained_igibson.yaml  # Im3D-Pano (trained)
benchmark-toolbox run --config configs/examples/total3d_igibson.yaml       # Total3D-Pano

cat artifacts/<name>/report.md        # the row
cat artifacts/<name>/run.json         # provenance: env fingerprint, repo commit, checkpoints
```

Each of those configs differs only in `runner_args` — which inference root to score and
which method label to record. There is no runner and no environment per method, because
the methods are configurations of one code base; see
[models_guide.md](models_guide.md#why-there-is-no-runner-per-method).

Step 2 must be re-run after any change to `runners/_common.py` or
`dpc_runner.convert_data_pkl`: ground truth is built by that same code, and a converter
change that is not reflected in the ground truth makes the two sides incomparable.

## Route B — fixture re-scoring

Convert the inference to JSON once, then score without the model environment at all:

```bash
# in the model environment: convert every prediction (gibson2 is imported once)
conda run -n Pano3D python $TBX/scripts/dpc_batch_convert.py \
    $DPC/data/igibson_bench/manifest.jsonl $DPC/data/igibson_bench/pred \
    --repo $DPC --pred-root out/pano3d
# -> $DPC/data/igibson_bench/pred/<sample_id>.json + manifest_pred.jsonl

# anywhere, no GPU, no model: set the two paths in the config, then
benchmark-toolbox run --config configs/examples/dpc_igibson_rescore.yaml
```

Use `--pred-root out/pano3d_im3d --out-dir .../pred_im3d` (and the matching
`*_rescore.yaml`) for the other methods, so predictions of two methods never land in one
row.

## The published protocol (11 categories) and the control row

The configs above average AP over **every** label in the ground truth. DPC and
PanoContext-Former report mAP over **11 categories**, which is a different number:

```bash
benchmark-toolbox run --config configs/examples/dpc_igibson_pcf.yaml
```

This is the **control row**: scoring the authors' own predictions under their own class
set puts this toolbox's metric next to a published one (DPC reports mAP 52.69 at oriented
3D-IoU 0.15). It also emits per-class `ap_*` rows that line up with PanoContext-Former's
Table 1. The class list lives once in `configs/protocols/igibson_pcf.yaml`.

## Producing the inference

Route A scores inference that already exists; this is how it is produced (one GPU run per
method, into its own output root so the methods cannot mix):

```bash
cd $DPC
# DPC@100 — the released default
WANDB_MODE=dryrun python main.py configs/pano3d_igibson.yaml \
    --data.split data/igibson --mode qtest --log.path out/pano3d
# Im3D-Pano — the same weights with the relation optimization off
WANDB_MODE=dryrun python main.py configs/pano3d_igibson.yaml \
    --data.split data/igibson --mode qtest \
    --model.scene_gcn.relation_adjust False --log.path out/pano3d_im3d
# Total3D-Pano — shared trunk + our MGN head
WANDB_MODE=dryrun python main.py configs/pano3d_igibson_total3d.yaml \
    --data.split data/igibson --mode qtest --log.path out/pano3d_total3d
```

Shape meshes (for `mesh_chamfer`) need `--mode test` plus the export patch — see
[shape_metric.md](shape_metric.md).

## Training the shape heads

```bash
# GT for MGN: surface points + densities from the watertight meshes
conda run -n Pano3D python $TBX/scripts/mgn_make_gt.py $DPC --processes 8
# training (A100; submit through your scheduler)
conda run -n Pano3D python main.py configs/mgnet_igibson.yaml --mode qtrain          # MGN  -> Total3D-Pano
conda run -n Pano3D python main.py configs/ldif_igibson_scratch.yaml --mode qtrain   # LDIF -> Im3D-Pano (trained)
```

The ChamferDistance CUDA extension must actually build for training (it is degraded to
`None` for inference only).

## Which config produces which row

| Row | Weights | Through the environment | Fixture re-scoring |
|---|---|---|---|
| DPC@100 (release) | authors' | `dpc_igibson.yaml` | `dpc_igibson_rescore.yaml` |
| DPC@20 (stabilised) | authors' | `dpc_s20_igibson.yaml` | `dpc_s20_rescore.yaml` |
| Im3D-Pano (ablation) | authors' | `im3d_igibson.yaml` | `im3d_igibson_rescore.yaml` |
| Im3D-Pano (trained) | ours | `im3d_trained_igibson.yaml` | `im3d_trained_rescore.yaml` |
| Total3D-Pano | ours | `total3d_igibson.yaml` | `total3d_igibson_rescore.yaml` |
| DPC, 11 categories (control) | authors' | — | `dpc_igibson_pcf.yaml` |
| Shape: LDIF vs MGN | ours | — | `shape_chamfer_rescore.yaml` |

## Methods outside the paper

HoloPano is the dissertation's own method, kept here as the worked example of plugging in
a method that is *not* a DPC configuration. Its predictions are per-camera `.npz` files;
`scripts/holopano_to_fixtures.py` converts both sides into fixtures, which
`configs/examples/holopano_pcf_rescore.yaml` then scores under the same 11-category
protocol. Its export carries no room layout, so the layout rows come out as `n/a`. See
[models_guide.md](models_guide.md).

## Environment and determinism

- **Model environment** (`Pano3D`): torch 1.7.1+cu110, CUDA 11, detectron2 (cu110),
  gibson2. Source patches are applied idempotently by `scripts/dpc_postsetup.py`.
- **Toolbox core**: pure standard library on Python 3.9 — the metrics need no GPU, no
  numpy, and no model.
- **Seeds** are fixed in the configs; `run.json` records the config, seed, git revision
  and environment provenance for every run.
- **Hardware**: NVIDIA A100-40GB. Two inference runs of the same code differ slightly:
  the saved layout corners move in the sixth decimal, which is why layout-dependent
  metrics agree across configurations to six decimals rather than bitwise (the object
  boxes themselves are exactly equal — see observation 2 in [results.md](results.md)).

## Honest reproduction (important caveats)

- The **released DPC config** (`optimize_steps 100`, `optimize_lr 1.0`) is **unstable on
  ~17% of scenes**: it degenerates the layout. This is reproducible, not noise —
  `optimize_steps 20` gives 0 degenerate layouts out of 500. Both operating points are
  reported.
- **layout_iou is a shared control**, not a discriminating metric: every method takes the
  same HorizonNet layout (as in the DPC authors' protocol).
- **Im3D-Pano (ablation)** is DPC with the relation optimization off on the same weights;
  **Im3D-Pano (trained)** is an LDIF head retrained from scratch. Different rows, and the
  ablation must be called an ablation.
