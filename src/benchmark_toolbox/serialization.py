from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def read_mapping_file(path: Path, *, what: str) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError(
                f"{what} in YAML format requires PyYAML. "
                "Install the project with 'pip install -e .'."
            ) from error
        data = yaml.safe_load(text)
    if not isinstance(data, Mapping):
        raise ValueError(f"{what} must be a mapping: {path}")
    return data
