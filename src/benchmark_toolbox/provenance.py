from __future__ import annotations

import subprocess
from pathlib import Path


def git_revision(repo_dir: "Path | None" = None) -> "str | None":
    command = ["git"]
    if repo_dir is not None:
        command += ["-C", str(repo_dir)]
    command += ["rev-parse", "HEAD"]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()
