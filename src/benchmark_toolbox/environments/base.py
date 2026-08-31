from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from benchmark_toolbox.environments.spec import EnvironmentSpec
from benchmark_toolbox.registry import ComponentRegistry

LOGGER = logging.getLogger("benchmark_toolbox.environments")


@dataclass(frozen=True)
class EnvHandle:
    command_prefix: tuple[str, ...] = ()
    cwd: Path | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    wrapper: tuple[str, ...] = ()


class BaseEnvironmentManager(ABC):
    registry: ComponentRegistry[BaseEnvironmentManager] = ComponentRegistry(
        "environment manager"
    )

    def __init__(self, parameters: dict[str, Any] | None = None) -> None:
        self.parameters = dict(parameters or {})

    @abstractmethod
    def prepare(self, spec: EnvironmentSpec) -> EnvHandle:
        """Idempotently build the environment (one-off, may be slow)."""

    @abstractmethod
    def resolve(self, spec: EnvironmentSpec) -> EnvHandle:
        """Return the command prefix / working directory / variables for the runner."""

    @abstractmethod
    def _env_exists(self, spec: EnvironmentSpec) -> bool:
        """Whether the environment physically exists."""

    def repo_dir(self, spec: EnvironmentSpec) -> Path:
        return spec.workspace / "repos" / spec.name

    def marker_path(self, spec: EnvironmentSpec) -> Path:
        return spec.workspace / "markers" / f"{spec.name}.json"

    def _runtime_env(self, spec: EnvironmentSpec) -> dict[str, str]:
        env = dict(spec.env)
        if spec.variant == "cpu":
            env.setdefault("CUDA_VISIBLE_DEVICES", "")
        return env

    def environment_exists(self, spec: EnvironmentSpec) -> bool:
        return self._env_exists(spec)

    def adopt(self, spec: EnvironmentSpec) -> None:
        if not self._env_exists(spec):
            raise RuntimeError(
                f"Cannot adopt '{spec.name}' ({spec.backend}): the environment does not "
                f"exist yet. Run 'benchmark-toolbox env prepare' to build it."
            )
        repo_dir = self.repo_dir(spec) if spec.repo else None
        extra = {"adopted": True}
        if repo_dir is not None:
            extra["repo_dir"] = str(repo_dir)
        self._write_marker(spec, extra, repo_dir=repo_dir)

    def is_prepared(self, spec: EnvironmentSpec) -> bool:
        marker = self.marker_path(spec)
        if not marker.exists():
            return False
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        repo_dir = self.repo_dir(spec) if spec.repo else None
        fingerprint_root = (
            repo_dir if repo_dir is not None and repo_dir.exists() else None
        )
        return (
            data.get("fingerprint") == spec.fingerprint(fingerprint_root)
            and self._env_exists(spec)
        )

    def _write_marker(
        self,
        spec: EnvironmentSpec,
        extra: Mapping[str, Any] | None = None,
        repo_dir: "Path | None" = None,
    ) -> None:
        marker = self.marker_path(spec)
        marker.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "name": spec.name,
            "backend": spec.backend,
            "variant": spec.variant,
            "fingerprint": spec.fingerprint(repo_dir),
            "prepared_at": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            payload.update(extra)
        marker.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def create(
        cls, name: str, parameters: dict[str, Any] | None = None
    ) -> BaseEnvironmentManager:
        return cls.registry.create(name, parameters)
