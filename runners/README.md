# Model runners

A runner lives in the isolated environment of a specific model and converts the
model's internal result into the shared JSON contract.

The repository ships runners for Total3D, IM3D and DPC. A new model is plugged in
with a separate runner file, without adding its source code to the toolbox.

## Input

```json
{
  "sample_id": "scene-001",
  "input_path": "/absolute/path/to/panorama.png",
  "metadata": {}
}
```

## Output

```json
{
  "layout": {
    "min_corner": [0.0, 0.0, 0.0],
    "max_corner": [4.0, 3.0, 5.0]
  },
  "objects": [
    {
      "object_id": "chair-1",
      "label": "chair",
      "score": 0.96,
      "bbox": {
        "min_corner": [1.0, 0.0, 1.0],
        "max_corner": [2.0, 1.0, 2.0]
      }
    }
  ],
  "relations": [],
  "metadata": {
    "model": "dpc",
    "checkpoint": "..."
  }
}
```

## Batch mode (optional)

Entering a model environment is expensive — for DeepPanoContext, deserializing the
first prediction imports the gibson2 stack (~36 s measured), while every further one in
the same process costs under a millisecond. A runner that can loop internally declares
it in its environment spec:

```yaml
runner:
  entry: runners/dpc_runner.py
  batch: true
```

and then receives ALL samples in one launch:

```json
{ "samples": [ { "sample_id": "scene-001", "input_path": "...", "metadata": {} }, ... ] }
```

answering with one entry per sample, each carrying its own `sample_id`:

```json
{ "outputs": [ { "layout": {...}, "objects": [...], "relations": [],
                 "metadata": { "sample_id": "scene-001" } }, ... ] }
```

Answers are matched by `sample_id`, never by position, and a batch that silently drops
a sample is an error. Use `rc.read_requests()` to read either shape and
`rc.write_outputs()` to answer a batch — `runners/echo_runner.py` is the reference: a
batch-capable runner is a loop around the single-sample case. A model that genuinely
needs a fresh process per sample simply leaves `batch` unset.

## Launch

A runner receives paths through command arguments matching the `{request}` and
`{output}` placeholders. Logs can be written to stdout/stderr: the benchmark
keeps them if the process fails.

The first practical integration step for each model is to extract, from the
existing Colab notebook, the stages that load an already-computed result and
convert it into this JSON. Once the converter is verified, real inference is
wired up to it.
