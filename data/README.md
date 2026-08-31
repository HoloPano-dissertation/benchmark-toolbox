# Data

Only manifests, small fixtures and documentation are stored in git.

- `examples/` is used by the architecture smoke test;
- `raw/` is meant for source datasets and is git-ignored;
- `cache/` is meant for converted data and is git-ignored.

A manifest uses the JSON Lines format:

```json
{"sample_id":"scene-001","input":"panorama.png","ground_truth":"ground_truth.json"}
```

Paths are resolved relative to the manifest file.
