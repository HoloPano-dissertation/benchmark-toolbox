from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

LOGGER = logging.getLogger("benchmark_toolbox.environments")


def apply_patches(patches: Sequence[Mapping[str, Any]], repo_dir: Path) -> None:
    for patch in patches:
        if "apply" in patch:
            _git_apply(patch, repo_dir)
        elif "prepend" in patch:
            _prepend(patch, repo_dir)
        elif "after" in patch:
            _insert_after(patch, repo_dir)
        elif "drop" in patch or "relax" in patch:
            _depin(patch, repo_dir)
        else:
            LOGGER.warning("Unknown patch type, skipped: %s", dict(patch))


def _target(patch: Mapping[str, Any], repo_dir: Path) -> Path:
    return repo_dir / str(patch["file"])


def _read_target(
    patch: Mapping[str, Any], repo_dir: Path
) -> "tuple[Path, str] | None":
    path = _target(patch, repo_dir)
    if not path.exists():
        LOGGER.warning("Patch target file not found: %s", path)
        return None
    return path, path.read_text(encoding="utf-8")


def _depin(patch: Mapping[str, Any], repo_dir: Path) -> None:
    target = _read_target(patch, repo_dir)
    if target is None:
        return
    path, text = target
    drops = [str(item) for item in patch.get("drop", ())]
    if drops:
        text = "\n".join(
            line for line in text.splitlines() if not any(d in line for d in drops)
        )
    for package in patch.get("relax", ()):
        text = re.sub(re.escape(str(package)) + r"=[^\s]+", str(package), text)
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8")
    LOGGER.info("de-pin patch applied: %s", path)


def _prepend(patch: Mapping[str, Any], repo_dir: Path) -> None:
    target = _read_target(patch, repo_dir)
    if target is None:
        return
    path, text = target
    snippet = str(patch["prepend"])
    if snippet.strip() and snippet.strip() in text:
        LOGGER.info("prepend patch already applied: %s", path)
        return
    path.write_text(snippet + text, encoding="utf-8")
    LOGGER.info("prepend patch applied: %s", path)


def _insert_after(patch: Mapping[str, Any], repo_dir: Path) -> None:
    target = _read_target(patch, repo_dir)
    if target is None:
        return
    path, text = target
    anchor = str(patch["after"])
    lines = list(patch["lines"])
    if any(line.strip() and line.strip() in text for line in lines):
        LOGGER.info("insert_after patch already applied: %s", path)
        return
    out: list[str] = []
    inserted = False
    for src_line in text.splitlines(keepends=True):
        out.append(src_line)
        if not inserted and anchor in src_line:
            for add in lines:
                out.append(add if add.endswith("\n") else add + "\n")
            inserted = True
    if not inserted:
        LOGGER.warning(
            "Anchor '%s' not found in %s — insert_after skipped", anchor, path
        )
        return
    path.write_text("".join(out), encoding="utf-8")
    LOGGER.info("insert_after patch applied: %s", path)


def _git_apply(patch: Mapping[str, Any], repo_dir: Path) -> None:
    raw = str(patch["apply"])
    diff = Path(raw) if Path(raw).is_absolute() else (repo_dir / raw)
    already = subprocess.run(
        ["git", "-C", str(repo_dir), "apply", "--reverse", "--check", str(diff)],
        capture_output=True,
    )
    if already.returncode == 0:
        LOGGER.info("Patch already applied: %s", diff)
        return
    subprocess.run(["git", "-C", str(repo_dir), "apply", str(diff)], check=True)
    LOGGER.info("git apply patch applied: %s", diff)
