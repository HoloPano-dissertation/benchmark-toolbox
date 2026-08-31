from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmark_toolbox.serialization import read_mapping_file


@dataclass(frozen=True)
class ComponentConfig:
    type: str
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    seed: int
    output_dir: Path
    model: ComponentConfig
    dataset: ComponentConfig
    metrics: tuple[ComponentConfig, ...]
    source_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "output_dir": str(self.output_dir),
            "model": {
                "type": self.model.type,
                "parameters": dict(self.model.parameters),
            },
            "dataset": {
                "type": self.dataset.type,
                "parameters": dict(self.dataset.parameters),
            },
            "metrics": [
                {"type": metric.type, "parameters": dict(metric.parameters)}
                for metric in self.metrics
            ],
            "source_path": str(self.source_path),
        }


def _resolve_parameter_paths(
    parameters: Mapping[str, Any], base_dir: Path
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    path_keys = {"manifest", "checkpoint", "prediction_dir", "environment"}
    for key, value in parameters.items():
        is_path = key in path_keys or key.endswith(("_path", "_dir", "_root"))
        if is_path and isinstance(value, str):
            candidate = Path(value).expanduser()
            result[key] = str(
                candidate if candidate.is_absolute() else (base_dir / candidate).resolve()
            )
        else:
            result[key] = value
    return result


def _component(data: Mapping[str, Any], base_dir: Path) -> ComponentConfig:
    if "type" not in data:
        raise ValueError("Component configuration requires a 'type' field")
    return ComponentConfig(
        type=str(data["type"]),
        parameters=_resolve_parameter_paths(data.get("parameters", {}), base_dir),
    )


def _expand_metrics(
    entries: Sequence[Mapping[str, Any]], base_dir: Path
) -> tuple[ComponentConfig, ...]:
    metrics: list[ComponentConfig] = []
    for entry in entries:
        component = _component(entry, base_dir)
        parameters = dict(component.parameters)
        if not parameters.pop("per_class", False):
            metrics.append(component)
            continue
        classes = parameters.get("classes")
        if not classes:
            raise ValueError("A 'per_class' metric requires a non-empty 'classes' list")
        prefix = str(parameters.pop("name", "ap"))
        for class_name in classes:
            metrics.append(
                ComponentConfig(
                    type=component.type,
                    parameters={
                        **parameters,
                        "classes": [class_name],
                        "name": f"{prefix}_{class_name}",
                    },
                )
            )
    return tuple(metrics)


def _load_layered(
    path: Path, seen: "tuple[Path, ...]" = ()
) -> dict[str, tuple[Any, Path]]:
    if path in seen:
        chain = " -> ".join(str(item) for item in (*seen, path))
        raise ValueError(f"Cyclic 'extends' in the experiment configuration: {chain}")
    data = read_mapping_file(path, what="Experiment configuration")
    layered: dict[str, tuple[Any, Path]] = {}
    parent = data.get("extends")
    if parent:
        parent_path = Path(str(parent)).expanduser()
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        layered.update(_load_layered(parent_path.resolve(), (*seen, path)))
    for key, value in data.items():
        if key != "extends":
            layered[key] = (value, path.parent)
    return layered


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    source_path = Path(path).expanduser().resolve()
    layered = _load_layered(source_path)

    def value(key: str, default: Any = None) -> Any:
        return layered[key][0] if key in layered else default

    def origin(key: str) -> Path:
        return layered[key][1] if key in layered else source_path.parent

    base_dir = source_path.parent
    output_dir = Path(str(value("output_dir", "artifacts")))
    if not output_dir.is_absolute():
        output_dir = (base_dir / output_dir).resolve()
    metrics_data = value("metrics", [])
    if not metrics_data:
        raise ValueError("At least one metric must be configured")
    for required in ("model", "dataset"):
        if required not in layered:
            raise ValueError(f"Experiment configuration requires '{required}'")
    return ExperimentConfig(
        experiment_name=str(value("experiment_name", source_path.stem)),
        seed=int(value("seed", 42)),
        output_dir=output_dir,
        model=_component(value("model"), origin("model")),
        dataset=_component(value("dataset"), origin("dataset")),
        metrics=_expand_metrics(metrics_data, origin("metrics")),
        source_path=source_path,
    )
