from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from benchmark_toolbox.serialization import read_mapping_file

DEFAULT_WORKSPACE = ".benchmark_toolbox"


@dataclass(frozen=True)
class RepoSpec:
    url: str
    commit: str | None = None


@dataclass(frozen=True)
class CheckpointSpec:
    url: str
    dest: str
    sha256: str | None = None
    unpack: bool = False


@dataclass(frozen=True)
class RunnerSpec:
    entry: str
    args: tuple[str, ...] = ()
    wrapper: tuple[str, ...] = ()
    batch: bool = False


@dataclass(frozen=True)
class EnvironmentSpec:
    name: str
    backend: str
    runner: RunnerSpec
    base_dir: Path
    workspace: Path
    python: str | None = None
    variant: str = "cpu"
    repo: RepoSpec | None = None
    env_file: str | None = None
    requirements: tuple[str, ...] = ()
    pip: tuple[str, ...] = ()
    system: tuple[str, ...] = ()
    checkpoints: tuple[CheckpointSpec, ...] = ()
    patches: tuple[Mapping[str, Any], ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)

    def fingerprint(self, repo_dir: "Path | None" = None) -> str:
        payload = {
            "backend": self.backend,
            "python": self.python,
            "variant": self.variant,
            "repo": None
            if self.repo is None
            else {"url": self.repo.url, "commit": self.repo.commit},
            "env_file": _read_env_file_text(self.env_file, self.base_dir, repo_dir)
            if self.env_file
            else None,
            "requirements": [
                _read_text_safe(self.base_dir / requirement)
                for requirement in self.requirements
            ],
            "pip": list(self.pip),
            "system": list(self.system),
            "checkpoints": [
                {"url": item.url, "dest": item.dest, "sha256": item.sha256}
                for item in self.checkpoints
            ],
            "patches": [dict(patch) for patch in self.patches],
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _read_text_safe(path: Path) -> "str | None":
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _read_env_file_text(
    name: "str | None", base_dir: Path, repo_dir: "Path | None"
) -> "str | None":
    if not name:
        return None
    if repo_dir is not None:
        text = _read_text_safe(repo_dir / name)
        if text is not None:
            return text
    return _read_text_safe(base_dir / name)


def load_environment_spec(
    source: "str | Path | Mapping[str, Any] | EnvironmentSpec",
    base_dir: "str | Path | None" = None,
) -> EnvironmentSpec:
    if isinstance(source, EnvironmentSpec):
        return source
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser().resolve()
        data = read_mapping_file(path, what="Environment spec")
        resolved_base = path.parent
    elif isinstance(source, Mapping):
        data = source
        resolved_base = (
            Path(base_dir).expanduser().resolve() if base_dir else Path.cwd()
        )
    else:
        raise TypeError(f"Unsupported environment spec source: {type(source)!r}")
    return _build_spec(data, resolved_base)


def _build_spec(data: Mapping[str, Any], base_dir: Path) -> EnvironmentSpec:
    try:
        name = str(data["name"])
        backend = str(data["backend"]).strip().lower()
    except KeyError as error:
        raise ValueError(f"Environment spec requires {error}") from error

    runner_data = data.get("runner") or {}
    if "entry" not in runner_data:
        raise ValueError("Environment spec requires 'runner.entry'")
    runner = RunnerSpec(
        entry=str(runner_data["entry"]),
        args=tuple(str(argument) for argument in runner_data.get("args", ())),
        wrapper=tuple(str(part) for part in runner_data.get("wrapper", ())),
        batch=bool(runner_data.get("batch", False)),
    )

    repo = None
    repo_data = data.get("repo")
    if repo_data:
        repo = RepoSpec(
            url=str(repo_data["url"]),
            commit=str(repo_data["commit"]) if repo_data.get("commit") else None,
        )

    checkpoints = tuple(
        CheckpointSpec(
            url=str(item["url"]),
            dest=str(item["dest"]),
            sha256=str(item["sha256"]) if item.get("sha256") else None,
            unpack=bool(item.get("unpack", False)),
        )
        for item in (data.get("checkpoints") or ())
    )

    workspace = Path(str(data.get("workspace", DEFAULT_WORKSPACE))).expanduser()
    if not workspace.is_absolute():
        workspace = (Path.cwd() / workspace).resolve()

    return EnvironmentSpec(
        name=name,
        backend=backend,
        runner=runner,
        base_dir=base_dir,
        workspace=workspace,
        python=str(data["python"]) if data.get("python") else None,
        variant=str(data.get("variant", "cpu")).strip().lower(),
        repo=repo,
        env_file=str(data["env_file"]) if data.get("env_file") else None,
        requirements=tuple(str(item) for item in (data.get("requirements") or ())),
        pip=tuple(str(item) for item in (data.get("pip") or ())),
        system=tuple(str(item) for item in (data.get("system") or ())),
        checkpoints=checkpoints,
        patches=tuple(dict(patch) for patch in (data.get("patches") or ())),
        env={str(key): str(value) for key, value in (data.get("env") or {}).items()},
    )
