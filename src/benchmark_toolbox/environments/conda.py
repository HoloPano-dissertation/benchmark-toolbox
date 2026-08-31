from __future__ import annotations

import logging
import shlex
import shutil
import subprocess
from pathlib import Path

from benchmark_toolbox.environments.base import BaseEnvironmentManager, EnvHandle
from benchmark_toolbox.environments.patches import apply_patches
from benchmark_toolbox.environments.spec import EnvironmentSpec
from benchmark_toolbox.environments.steps import (
    clone_repo,
    download_checkpoint,
    run_step,
    which_or_raise,
)

LOGGER = logging.getLogger("benchmark_toolbox.environments")

PIP_NET_FLAGS = ["--retries", "10", "--timeout", "120"]


@BaseEnvironmentManager.registry.register("conda")
class CondaEnvironmentManager(BaseEnvironmentManager):
    def _env_file(self, spec: EnvironmentSpec, repo_dir: Path) -> "Path | None":
        if not spec.env_file:
            return None
        committed = spec.base_dir / spec.env_file
        if committed.exists():
            return committed
        return repo_dir / spec.env_file

    def _conda_env_present(self, name: str) -> bool:
        conda = shutil.which("conda")
        if conda is None:
            return False
        try:
            result = subprocess.run(
                [conda, "env", "list"],
                capture_output=True,
                text=True,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return False
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.split()[0] == name:
                return True
        return False

    def _env_exists(self, spec: EnvironmentSpec) -> bool:
        return self._conda_env_present(spec.name)

    def prepare(self, spec: EnvironmentSpec) -> EnvHandle:
        conda = which_or_raise("conda")
        repo_dir = self.repo_dir(spec)
        if spec.repo:
            clone_repo(spec.repo, repo_dir)
        if spec.patches:
            apply_patches(spec.patches, repo_dir)

        if not self._conda_env_present(spec.name):
            env_file = self._env_file(spec, repo_dir)
            if env_file and env_file.exists():
                run_step([conda, "env", "create", "-n", spec.name, "-f", env_file])
            elif spec.python:
                run_step(
                    [conda, "create", "-y", "-n", spec.name, f"python={spec.python}"]
                )
            else:
                raise RuntimeError(
                    f"conda environment '{spec.name}' requires 'env_file' or 'python'"
                )
        else:
            LOGGER.info("conda environment already present: %s", spec.name)

        pip_install = [conda, "run", "-n", spec.name, "python", "-m", "pip", "install"]
        for requirement in spec.requirements:
            run_step(pip_install + PIP_NET_FLAGS + ["-r", spec.base_dir / requirement])
        for item in spec.pip:
            run_step(pip_install + PIP_NET_FLAGS + shlex.split(item))

        for checkpoint in spec.checkpoints:
            download_checkpoint(checkpoint, repo_dir)

        if spec.system:
            LOGGER.warning(
                "System packages are not installed automatically (need privileges): %s",
                ", ".join(spec.system),
            )

        self._write_marker(
            spec,
            {"repo_dir": str(repo_dir) if spec.repo else None},
            repo_dir=repo_dir if spec.repo else None,
        )
        return self.resolve(spec)

    def resolve(self, spec: EnvironmentSpec) -> EnvHandle:
        conda = shutil.which("conda") or "conda"
        prefix = (conda, "run", "--no-capture-output", "-n", spec.name, "python")
        cwd = self.repo_dir(spec) if spec.repo else None
        env = self._runtime_env(spec)
        return EnvHandle(
            command_prefix=prefix,
            cwd=cwd,
            env=env,
            wrapper=tuple(spec.runner.wrapper),
        )
