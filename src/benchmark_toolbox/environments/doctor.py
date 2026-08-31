from __future__ import annotations

import shutil
import subprocess

from benchmark_toolbox.environments.steps import local_artifacts_dir, offline

TOOLS = ["python3", "git", "conda", "mamba"]


def _gpu() -> str:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return "not detected (CPU mode)"
    try:
        result = subprocess.run(
            [executable, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return ", ".join(names) if names else "available, name unknown"
    except (OSError, subprocess.SubprocessError):
        return "query failed"


def doctor_report() -> str:
    lines = ["benchmark-toolbox runtime environment:", ""]
    for tool in TOOLS:
        path = shutil.which(tool)
        marker = "OK" if path else "--"
        lines.append(f"  [{marker}] {tool:<12} {path or 'not found'}")
    lines.append(f"  [..] GPU          {_gpu()}")

    staging = local_artifacts_dir()
    lines += [
        "",
        "Provisioning on a restricted host:",
        f"  network              {'OFFLINE (BENCHMARK_TOOLBOX_OFFLINE)' if offline() else 'allowed'}",
        f"  staged artifacts     {staging or 'not configured (BENCHMARK_TOOLBOX_ARTIFACTS)'}",
        "  plan a build         benchmark-toolbox env prepare --env <spec> --dry-run",
    ]
    lines += [
        "",
        "Environment backend hints:",
        "  conda                -> DPC / IM3D / Total3D (old PyTorch/CUDA)",
        "  venv                 -> lightweight models and local development",
    ]
    return "\n".join(lines)
