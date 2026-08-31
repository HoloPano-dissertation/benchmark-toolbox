from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Sequence

from benchmark_toolbox.domain import SceneOutput, SceneSample
from benchmark_toolbox.models.base import BaseSceneEstimator
from benchmark_toolbox.provenance import git_revision

LOGGER = logging.getLogger("benchmark_toolbox.models")


def _usable_wrapper(wrapper: Sequence[str]) -> tuple[str, ...]:
    wrapper = tuple(str(part) for part in wrapper)
    if not wrapper:
        return ()
    if shutil.which(wrapper[0]) is not None:
        return wrapper
    LOGGER.warning(
        "Launch wrapper '%s' is not on PATH — skipping (headless mode). "
        "If the method needs a display, install it or run on a node with X/EGL.",
        wrapper[0],
    )
    return ()


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "runners").is_dir():
            return parent
    return here.parents[2]


@BaseSceneEstimator.registry.register("subprocess")
class SubprocessSceneEstimator(BaseSceneEstimator):
    def __init__(self, parameters: dict[str, Any]) -> None:
        self.cwd: Path | None = None
        self.run_env: dict[str, str] | None = None
        self.seed = parameters.get("seed")
        environment = parameters.get("environment")

        if environment is not None:
            from benchmark_toolbox.environments import (
                BaseEnvironmentManager,
                load_environment_spec,
            )

            spec = load_environment_spec(environment)
            manager = BaseEnvironmentManager.create(spec.backend)
            if not manager.is_prepared(spec):
                raise RuntimeError(
                    f"Environment '{spec.name}' ({spec.backend}) is not prepared. "
                    f"Run: benchmark-toolbox env prepare --env <spec>"
                )
            handle = manager.resolve(spec)
            entry = self._resolve_entry(spec.runner.entry)
            requested_wrapper = tuple(handle.wrapper)
            wrapper = _usable_wrapper(requested_wrapper)

            extra_args = tuple(
                str(argument) for argument in parameters.get("runner_args", ())
            )
            self.command = (
                wrapper
                + tuple(handle.command_prefix)
                + (entry,)
                + tuple(spec.runner.args)
                + extra_args
                + ("--request", "{request}", "--output", "{output}")
            )
            self.batch = spec.runner.batch
            self.cwd = handle.cwd
            self.run_env = dict(handle.env) if handle.env else None
            self.environment_name = spec.name
            fingerprint_root = manager.repo_dir(spec) if spec.repo else None
            self._provenance: dict[str, Any] = {
                "environment": spec.name,
                "backend": spec.backend,
                "variant": spec.variant,
                "fingerprint": spec.fingerprint(fingerprint_root),
                "runner": spec.runner.entry,
            }
            if extra_args:
                self._provenance["runner_args"] = list(extra_args)

            if wrapper:
                self._provenance["wrapper"] = list(wrapper)
            elif requested_wrapper:
                self._provenance["wrapper_skipped"] = list(requested_wrapper)
            if spec.repo:
                commit = git_revision(manager.repo_dir(spec))
                if commit:
                    self._provenance["repo_commit"] = commit
            if spec.checkpoints:
                self._provenance["checkpoints"] = [
                    {"dest": item.dest, "sha256": item.sha256}
                    for item in spec.checkpoints
                ]
        else:
            command = parameters.get("command")
            if not isinstance(command, Sequence) or isinstance(command, (str, bytes)):
                raise ValueError(
                    "Provide either 'environment' (spec) or 'command' "
                    "(array of process arguments)"
                )
            self.command = tuple(str(argument) for argument in command)
            self.batch = bool(parameters.get("batch", False))
            self.environment_name = str(parameters.get("environment_name", "external"))
            self._provenance = {
                "environment": self.environment_name,
                "command": list(self.command),
            }

        self.timeout_seconds = float(parameters.get("timeout_seconds", 3600))

    @staticmethod
    def _resolve_entry(entry: str) -> str:
        path = Path(entry).expanduser()
        if path.is_absolute():
            return str(path)
        return str((_project_root() / path).resolve())

    def provenance(self) -> dict[str, Any]:
        return dict(self._provenance)

    def _request(self, sample: SceneSample) -> dict[str, Any]:
        return {
            "sample_id": sample.sample_id,
            "input_path": str(sample.input_path.resolve()),
            "metadata": dict(sample.metadata),
            "seed": self.seed,
        }

    def _launch(self, request: dict[str, Any], work_dir: Path) -> Any:
        request_path = work_dir / "request.json"
        output_path = work_dir / "output.json"
        request_path.write_text(
            json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        command = [
            argument.replace("{request}", str(request_path)).replace(
                "{output}", str(output_path)
            )
            for argument in self.command
        ]
        run_env = None
        if self.run_env is not None:
            run_env = {**os.environ, **self.run_env}
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            cwd=str(self.cwd) if self.cwd is not None else None,
            env=run_env,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Model runner failed with code {completed.returncode}.\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        if not output_path.exists():
            raise RuntimeError(
                "Model runner succeeded but did not create the requested output file"
            )
        with output_path.open(encoding="utf-8") as output_file:
            return json.load(output_file)

    def _scene(self, payload: Any) -> SceneOutput:
        output = SceneOutput.from_dict(payload)
        metadata = dict(output.metadata)
        metadata.setdefault("environment", self.environment_name)
        return SceneOutput(
            layout=output.layout,
            objects=output.objects,
            relations=output.relations,
            metadata=metadata,
        )

    @contextmanager
    def _work_dir(self) -> Iterable[Path]:
        with tempfile.TemporaryDirectory(prefix="benchmark-toolbox-") as temporary:
            yield Path(temporary)

    def predict(self, sample: SceneSample) -> SceneOutput:
        with self._work_dir() as work_dir:
            return self._scene(self._launch(self._request(sample), work_dir))

    def predict_many(self, samples: Sequence[SceneSample]) -> Iterable[SceneOutput]:
        if not self.batch or len(samples) <= 1:
            yield from super().predict_many(samples)
            return
        with self._work_dir() as work_dir:
            request = {"samples": [self._request(sample) for sample in samples]}
            payload = self._launch(request, work_dir)
        if not isinstance(payload, dict) or "outputs" not in payload:
            raise RuntimeError(
                "A batch runner must answer with {'outputs': [...]}, one entry per "
                "requested sample (see runners/README.md)"
            )
        by_id: dict[str, Any] = {}
        for entry in payload["outputs"]:
            sample_id = (entry.get("metadata") or {}).get("sample_id") or entry.get(
                "sample_id"
            )
            if sample_id is None:
                raise RuntimeError(
                    "A batch runner's output entry must carry its sample_id "
                    "(in metadata.sample_id, or as a top-level 'sample_id')"
                )
            by_id[str(sample_id)] = entry
        missing = [s.sample_id for s in samples if s.sample_id not in by_id]
        if missing:
            shown = ", ".join(missing[:5])
            more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
            raise RuntimeError(
                f"The batch runner returned no prediction for {len(missing)} of "
                f"{len(samples)} samples: {shown}{more}"
            )
        for sample in samples:
            yield self._scene(by_id[sample.sample_id])

BaseSceneEstimator.registry.alias("subprocess", "dpc", "holopano")
