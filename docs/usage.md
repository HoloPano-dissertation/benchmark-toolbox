# Usage guide

How to run things: the CLI, the config format, the environment spec, and what a run
leaves behind. The design is in [architecture.md](architecture.md), the metrics in
[metrics.md](metrics.md), and reproducing the paper's numbers in
[REPRODUCING.md](REPRODUCING.md).

The toolbox integrates and evaluates models; it does not carry their source, weights, or
datasets.

## Quick start

Two commands prove the whole path works, with no GPU, no network and no model:

```bash
make setup
make smoke           # metrics over a fixture prediction
make env-demo        # env prepare echo + run echo_local: the full isolated-environment path
cat artifacts/echo/report.md
```

`make env-demo` is the same as:

```bash
benchmark-toolbox env prepare --env configs/environments/echo.yaml
benchmark-toolbox run --config configs/examples/echo_local.yaml
```

A real method follows the same shape, only its environment is heavier: `env prepare`
once per model, then `run` per experiment. Use `benchmark-toolbox env doctor` to see
which backends (conda/venv) and GPU the current host has.

## CLI commands

| Command | Purpose |
|---|---|
| `benchmark-toolbox run --config <cfg>` | run a single experiment |
| `benchmark-toolbox env prepare --env <spec>` | build a model environment from its spec |
| `benchmark-toolbox env prepare --config <cfg>` | same; the environment is taken from the experiment config |
| `benchmark-toolbox env prepare ... --force` | rebuild the environment from scratch |
| `benchmark-toolbox env prepare ... --dry-run` | report what would be built and what needs the network; change nothing |
| `benchmark-toolbox env prepare ... --adopt` | record an environment built by other means as prepared |
| `benchmark-toolbox env prepare ... --verify` | re-check the sha256 of checkpoints already on disk |
| `benchmark-toolbox env doctor` | report available backends, GPU, and the provisioning setup |
| `benchmark-toolbox env submit-script --env <spec>` | generate an sbatch/bash script for a compute cluster |

`env prepare` is idempotent: calling it again with the same spec does not rebuild the
environment.

## Experiment config

YAML or JSON. Paths in `parameters` are resolved relative to the file they are written
in — including a path inherited through `extends`.

A config may inherit a shared **protocol** (`configs/protocols/*.yaml`) so the metric list
is written once for all methods; a method's own config then only says where its
predictions live. Merging is per top-level key: a key set locally replaces the inherited
one outright.

```yaml
extends: ../protocols/igibson_boxes.yaml   # optional: seed + metrics from one place
experiment_name: dpc          # name; defaults to the file name
seed: 42                      # fixed before the run
output_dir: ../../artifacts/dpc

model:
  type: dpc                   # subprocess | dpc | fixture
  parameters:
    environment: ../environments/dpc.yaml   # environment spec (or an inline dict)

dataset:
  type: igibson               # manifest | igibson
  parameters:
    manifest: ../../data/examples/manifest.jsonl

metrics:
  - type: layout_iou_3d
  - type: object_map
    parameters:
      iou_threshold: 0.15      # the comparison-protocol threshold
  - type: object_map_dataset
    parameters:
      iou_threshold: 0.15
  - type: collision_rate
```

Model types: `subprocess` — the adapter that runs a model in its own environment;
`dpc` and `holopano` — aliases of it, so a config reads as the method it runs (which
model actually starts is decided by the environment spec and its runner, not by a Python
class); `fixture` — ready-made predictions from files, the re-scoring path behind every
row of [results.md](results.md). Dataset types: `manifest`, with `igibson` as its alias.

## Environment spec (`configs/environments/*.yaml`)

A declarative description of how to build a model environment.

```yaml
name: Pano3D                  # environment name (conda env / venv directory)
backend: conda                # conda | venv
python: "3.8"
variant: cpu                  # cpu | gpu — set gpu on a GPU node
repo:
  url: https://github.com/chengzhag/DeepPanoContext.git
  # commit: <hash>            # pin a version for reproducibility
env_file: environment.yaml    # conda env file (from the repo or next to the spec)
patches:                      # applied after cloning, before building the environment
  - { file: environment.yaml, drop: ["mirrors.tuna"] }   # clean up / unpin in the env
pip:
  - "detectron2 -f https://.../cu101/torch1.7/index.html"
system: [xvfb, ninja-build]   # system packages (installed separately, need privileges)
checkpoints:
  - url: "https://.../weights?download=1"
    dest: "dpc_pretrained.zip"
    unpack: true              # unpack the archive into the repo directory
    # sha256: <hash>          # integrity check
runner:
  entry: runners/dpc_runner.py
  args: ["--config", "configs/pano3d_igibson.yaml"]
  wrapper: ["xvfb-run", "-a"] # launch wrapper (headless rendering)
```

Environments are cached under `.benchmark_toolbox/` (out
of git):

```text
.benchmark_toolbox/
├── venvs/<name>/      venv environments
├── repos/<name>/      cloned model repositories and weights
└── markers/<name>.json  build marker (for idempotency)
```

### Offline builds (network-restricted hosts)

Compute nodes on an HPC cluster commonly have no route out at all, and the weights for
these models sit behind links that expire or throttle. Four things make a build survive
that.

**1. Plan first.** `env prepare --env <spec> --dry-run` prints what is already in place
and what each remaining step needs, without changing anything:

```text
  [ ok] conda environment 'Pano3D' exists
  [ ok] repository cloned
  [net] pip install detectron2 -f https://dl.fbaipublicfiles.com/...
  [ ok] checkpoint ~/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth
  [net] checkpoint dpc_pretrained.zip
        download https://...  (or stage 'dpc_pretrained.zip' in /home/me/tbx_artifacts)
  [ !!] system packages are NOT installed automatically (need privileges)
```

A build takes tens of minutes and tends to fail late — this is how you find out
beforehand which files to carry over, in one pass instead of by trial and error.

**2. Stage artifacts.** Set `BENCHMARK_TOOLBOX_ARTIFACTS=<dir>`. When `env prepare`
resolves a `checkpoints:` entry it first looks there for a file named after the
checkpoint's `dest` basename and copies it instead of downloading (the `sha256` and
`unpack` handling is unchanged). The same directory can serve as a local pip find-links
for `detectron2` (e.g. `pip: ["detectron2 -f <dir>"]`).

A checkpoint `dest` is normally relative to the model's cloned repository, but an
absolute or `~`-prefixed one is taken literally — which is how files that are not
repository files at all get staged. Torchvision backbones are the usual case: they are
only ever looked up in `~/.cache/torch/hub/checkpoints`, and are otherwise downloaded at
the first forward pass, i.e. minutes into a run on a node that cannot download them.

```yaml
checkpoints:
  - url: "https://download.pytorch.org/models/resnet18-5c106cde.pth"
    dest: "~/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth"
    sha256: "5c106cde386e87d4033832f2996f5493238eda96ccf559d1d62760c4de0613f8"
```

**3. Fail fast instead of hanging.** With `BENCHMARK_TOOLBOX_OFFLINE=1`, any step that
would need the network refuses immediately and names the file to stage and where to put
it — rather than stalling on a socket until it times out, halfway through a build.

**4. Verify what was staged.** `env prepare --verify` re-checks the sha256 of every
checkpoint already on disk. This catches the two classic silent failures: an expired
share link that returned an HTML login page saved under the weights' filename, and a
transfer resumed wrongly into a truncated file. Both look like a normal file until the
model loads them. Note that torchvision itself verifies a backbone's hash only when
downloading it, never when reading one from the cache — a wrongly named copy there is
loaded without complaint.

### Environments built by other means

A model environment often already exists — built from the model's own README, or by an
administrator. `env prepare --adopt` records it as prepared (writing the marker, with
`"adopted": true`) without rebuilding anything. Use it also after editing a spec in a
way that does not change the built environment: the marker stores a fingerprint of the
spec, so the edit alone would otherwise make `is_prepared` report false and refuse to
run.

## Running on a compute cluster

The toolbox runs locally on the machine it is invoked on (a workstation or a cluster
node) and does not make remote connections itself. To run on a cluster, place the
repository on it, then generate a batch script and submit it to the scheduler.

```bash
# on the workstation — generate the script
benchmark-toolbox env submit-script \
  --env configs/environments/dpc.yaml \
  --config configs/examples/dpc.yaml \
  --gpus 1 --out run_dpc.sh

# on the cluster node — submit to the scheduler
sbatch run_dpc.sh        # SLURM; the script runs env prepare and run itself
```

`--scheduler bash` produces a plain bash script without `#SBATCH` directives.

> HPC principle: run `env prepare` (clone + conda + weights) on the **login node**
> (internet and git there), and `run` (inference + metrics) on the **compute node** with
> a GPU. A shared filesystem (home/lustre) is visible from both, so the built
> environment is available to the node without a rebuild.

## Datasets

The `manifest` loader (and its aliases) reads JSONL — one line per scene:

```json
{"sample_id": "scene-001", "input": "panorama.png", "ground_truth": "gt/scene-001.json"}
```

- `input` — path to the image/panorama (relative to the manifest file);
- `ground_truth` — path to ground truth in `SceneOutput` format (or an inline object);
  the field may be omitted when there is no ground truth.

The `igibson` alias behaves identically to `manifest` but makes the config self-documenting.

## Output format (`SceneOutput`)

Every model's result is mapped to one representation, `<layout, objects, relations>`:

```json
{
  "layout": { "min_corner": [0,0,0], "max_corner": [4,3,5] },
  "objects": [
    { "object_id": "chair-1", "label": "chair", "score": 0.96,
      "bbox": { "center": [1.5,0.5,1.5], "size": [1,1,1],
                "basis": [[1,0,0],[0,1,0],[0,0,1]] },
      "attributes": {} }
  ],
  "relations": [],
  "metadata": { "model": "dpc" }
}
```

An object box is ORIENTED (`center`/`size`/`basis`) — the protocol scores a rotated
3D-IoU at threshold 0.15. The axis-aligned form (`min_corner`/`max_corner`) is also
accepted; `parse_box` tells them apart by which keys are present, and layouts are
usually axis-aligned.

The runner side of the contract — the request format, the optional batch mode, and the
helpers in `runners/_common.py` — is documented once, in
[runners/README.md](../runners/README.md).

## Metrics

The comparison protocol is ORIENTED 3D-IoU of objects at threshold **0.15** — not 0.5,
which would make the numbers incomparable with the publications for these methods. The
metric list, every parameter, and how to add your own are in [metrics.md](metrics.md);
the shared protocol files the configs inherit live in `configs/protocols/`.

## Run artifacts (`artifacts/<run>/`)

```text
predictions/<sample>.json   unified per-scene predictions
metrics.jsonl               per-scene metric values (one line each)
summary.json                aggregates (mean / median / ci95)
report.md                   a human-readable table
run.json                    config, time, git revision, and environment info
                            (backend, fingerprint, commit) — for reproducibility
```

## Adding a new model

Two files are enough; the core is unchanged:

1. `configs/environments/<model>.yaml` — the environment spec (backend, repo, weights,
   runner);
2. `runners/<model>_runner.py` — inference and conversion to `SceneOutput` per the
   contract above (use the converter for an existing method as a starting point, e.g.
   `runners/dpc_runner.py`).

Then run `env prepare --env configs/environments/<model>.yaml` and a normal `run`.

A ready-made template is the `holopano` model (`configs/environments/holopano.yaml` +
`runners/holopano_runner.py`): the result converter is already filled in (DPC format);
it remains to point the spec at the repository and weights and to fill in inference in
`run_inference()`. See also [metrics.md](metrics.md#adding-your-own-model).
