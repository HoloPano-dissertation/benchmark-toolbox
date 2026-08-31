from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from typing import Sequence

from benchmark_toolbox.environments.spec import CheckpointSpec, RepoSpec

LOGGER = logging.getLogger("benchmark_toolbox.environments")


def which_or_raise(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(
            f"Required executable '{name}' not found in PATH. "
            f"Run 'benchmark-toolbox env doctor' to see what is available."
        )
    return path


def run_step(
    command: Sequence[object],
    *,
    cwd: "Path | None" = None,
    env: "dict[str, str] | None" = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    rendered = [str(part) for part in command]
    LOGGER.info("START: %s", " ".join(rendered))
    completed = subprocess.run(
        rendered,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        check=check,
    )
    LOGGER.info("FINISH: %s", rendered[0])
    return completed


def clone_repo(repo: RepoSpec, dest: Path) -> Path:
    git = which_or_raise("git")
    if (dest / ".git").exists():
        LOGGER.info("Repository already cloned: %s", dest)
    elif offline():
        raise RuntimeError(
            f"Offline mode: cannot clone {repo.url}.\n"
            f"Clone it on a host with network access and copy it to {dest}."
        )
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        run_step([git, "clone", repo.url, dest])
    if repo.commit:
        run_step([git, "-C", dest, "checkout", repo.commit])
    return dest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def offline() -> bool:
    value = os.environ.get("BENCHMARK_TOOLBOX_OFFLINE", "").strip().lower()
    return value not in ("", "0", "false", "no")


def checkpoint_path(checkpoint: CheckpointSpec, repo_dir: Path) -> Path:
    dest = Path(checkpoint.dest).expanduser()
    return dest if dest.is_absolute() else repo_dir / dest


def local_artifacts_dir() -> "Path | None":
    value = os.environ.get("BENCHMARK_TOOLBOX_ARTIFACTS")
    return Path(value).expanduser() if value else None


def _stage_local_artifact(checkpoint: CheckpointSpec, tmp: Path) -> bool:
    base = local_artifacts_dir()
    if base is None:
        return False
    candidate = base / Path(checkpoint.dest).name
    if not candidate.exists():
        return False
    LOGGER.info("Using pre-staged artifact: %s", candidate)
    shutil.copyfile(candidate, tmp)
    return True


def download_checkpoint(checkpoint: CheckpointSpec, repo_dir: Path) -> Path:
    dest = checkpoint_path(checkpoint, repo_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sentinel = dest.with_name(dest.name + ".unpacked")
    if checkpoint.unpack:
        if sentinel.exists():
            LOGGER.info("Archive already downloaded and unpacked: %s", dest)
            return dest
    elif dest.exists() and (
        checkpoint.sha256 is None or sha256_file(dest) == checkpoint.sha256
    ):
        LOGGER.info("Checkpoint present: %s", dest)
        return dest

    tmp = dest.with_name(dest.name + ".part")
    if _stage_local_artifact(checkpoint, tmp):
        LOGGER.info("Staged '%s' from the local artifacts directory", checkpoint.dest)
    elif offline():
        raise RuntimeError(
            f"Offline mode: '{checkpoint.dest}' is missing and cannot be downloaded.\n"
            f"Fetch it on a host with network access:\n  {checkpoint.url}\n"
            f"then put it in BENCHMARK_TOOLBOX_ARTIFACTS as '{Path(checkpoint.dest).name}' "
            f"(or place it directly at {dest})."
        )
    else:
        LOGGER.info("START: download %s -> %s", checkpoint.url, dest)
        request = urllib.request.Request(
            checkpoint.url, headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(request) as response, open(tmp, "wb") as out:
            shutil.copyfileobj(response, out)

    if checkpoint.sha256 and sha256_file(tmp) != checkpoint.sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch for '{checkpoint.dest}'")
    if checkpoint.unpack:
        if not zipfile.is_zipfile(tmp):
            size = tmp.stat().st_size
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"Expected a zip archive, but '{checkpoint.dest}' is not one "
                f"({size} bytes — likely an HTML stub / login page). Provide the "
                f"file manually: unpack it into {dest.parent} and create the "
                f"marker {sentinel.name}, or pre-stage it in "
                f"BENCHMARK_TOOLBOX_ARTIFACTS."
            )
        with zipfile.ZipFile(tmp) as archive:
            archive.extractall(dest.parent)
        tmp.unlink(missing_ok=True)
        sentinel.write_text("", encoding="utf-8")
    else:
        tmp.replace(dest)
    LOGGER.info("FINISH: obtain %s", dest)
    return dest
