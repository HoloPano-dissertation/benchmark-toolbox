from __future__ import annotations

from benchmark_toolbox.environments.spec import EnvironmentSpec


def _env_init_lines(module_load: "str | None") -> list[str]:
    lines: list[str] = []

    if module_load:
        lines += [
            "# Enable conda via environment-modules, if available",
            f'if command -v module >/dev/null 2>&1; then module load {module_load}; fi',
            "",
        ]

    lines += [
        "# Fallback conda lookup (shared cluster lustre directory)",
        "if ! command -v conda >/dev/null 2>&1; then",
        "  for _c in /common/software/miniconda/*/bin/conda \\",
        "            /opt/conda/bin/conda \\",
        "            $HOME/miniconda3/bin/conda \\",
        "            $HOME/anaconda3/bin/conda; do",
        "    if [ -x \"$_c\" ]; then export PATH=\"$(dirname \"$_c\"):$PATH\"; break; fi",
        "  done",
        "fi",
        "",
    ]

    return lines


def generate_submit_script(
    env_path: str,
    config_path: "str | None" = None,
    *,
    spec: EnvironmentSpec,
    scheduler: str = "slurm",
    gpus: int = 0,
    module_load: "str | None" = "anaconda",
    time: str = "04:00:00",
) -> str:
    lines = ["#!/bin/bash"]

    if scheduler == "slurm":
        lines += [
            f"#SBATCH --job-name={spec.name}",
            f"#SBATCH --time={time}",
            "#SBATCH --cpus-per-task=4",
        ]
        if gpus or spec.variant == "gpu":
            lines.append(f"#SBATCH --gres=gpu:{gpus or 1}")
        lines.append("#SBATCH --output=%x-%j.out")

    lines += ["", "set -e", ""]
    lines += _env_init_lines(module_load)

    lines += [
        "# Build the model environment (idempotent) and run the benchmark",
        f"benchmark-toolbox env prepare --env {env_path}",
    ]
    if config_path:
        lines.append(f"benchmark-toolbox run --config {config_path}")

    return "\n".join(lines) + "\n"
