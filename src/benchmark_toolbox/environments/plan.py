from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from benchmark_toolbox.environments.base import BaseEnvironmentManager
from benchmark_toolbox.environments.spec import EnvironmentSpec
from benchmark_toolbox.environments.steps import (
    checkpoint_path,
    local_artifacts_dir,
    offline,
    sha256_file,
)

DONE = "done"
TODO = "todo"
NETWORK = "network"
WARN = "warn"

_MARK = {DONE: "ok", TODO: "..", NETWORK: "net", WARN: "!!"}


@dataclass(frozen=True)
class PlanStep:
    state: str
    summary: str
    detail: str = ""


def build_plan(spec: EnvironmentSpec, manager: BaseEnvironmentManager) -> list[PlanStep]:
    steps: list[PlanStep] = []
    repo_dir = manager.repo_dir(spec)

    if manager.environment_exists(spec):
        steps.append(PlanStep(DONE, f"{spec.backend} environment '{spec.name}' exists"))
    else:
        source = spec.env_file or (f"python={spec.python}" if spec.python else "?")
        steps.append(
            PlanStep(NETWORK, f"create {spec.backend} environment '{spec.name}'", source)
        )

    if spec.repo:
        if (repo_dir / ".git").exists():
            steps.append(PlanStep(DONE, "repository cloned", str(repo_dir)))
        else:
            steps.append(PlanStep(NETWORK, "clone repository", spec.repo.url))
        if spec.repo.commit:
            steps.append(PlanStep(TODO, f"checkout {spec.repo.commit}"))

    if spec.patches:
        steps.append(PlanStep(TODO, f"apply {len(spec.patches)} source patch(es)"))

    for requirement in spec.requirements:
        steps.append(PlanStep(NETWORK, f"pip install -r {requirement}"))
    for item in spec.pip:
        steps.append(PlanStep(NETWORK, f"pip install {item}"))

    staging = local_artifacts_dir()
    for checkpoint in spec.checkpoints:
        target = checkpoint_path(checkpoint, repo_dir)
        sentinel = target.with_name(target.name + ".unpacked")
        if (sentinel if checkpoint.unpack else target).exists():
            state, note = DONE, f"present at {target}"
        elif staging is not None and (staging / Path(checkpoint.dest).name).exists():
            state, note = TODO, f"staged in {staging}"
        else:
            state = NETWORK
            note = f"download {checkpoint.url}"
            if staging is not None:
                note += f"  (or stage '{Path(checkpoint.dest).name}' in {staging})"
        steps.append(PlanStep(state, f"checkpoint {checkpoint.dest}", note))

    if spec.system:
        steps.append(
            PlanStep(
                WARN,
                "system packages are NOT installed automatically (need privileges)",
                ", ".join(spec.system),
            )
        )
    return steps


def verify_checkpoints(spec: EnvironmentSpec, manager: BaseEnvironmentManager) -> list[str]:
    problems: list[str] = []
    repo_dir = manager.repo_dir(spec)
    for checkpoint in spec.checkpoints:
        if not checkpoint.sha256:
            continue
        target = checkpoint_path(checkpoint, repo_dir)
        if not target.exists() or target.is_dir():
            continue
        actual = sha256_file(target)
        if actual != checkpoint.sha256:
            problems.append(
                f"{target}: sha256 {actual} != expected {checkpoint.sha256}"
            )
    return problems


def render_plan(spec: EnvironmentSpec, steps: list[PlanStep]) -> str:
    lines = [
        f"Plan for environment '{spec.name}' ({spec.backend}, variant {spec.variant})",
        f"  workspace: {spec.workspace}",
    ]
    staging = local_artifacts_dir()
    lines.append(f"  pre-staged artifacts: {staging or 'not configured (BENCHMARK_TOOLBOX_ARTIFACTS)'}")
    lines.append(f"  network: {'OFFLINE (BENCHMARK_TOOLBOX_OFFLINE)' if offline() else 'allowed'}")
    lines.append("")
    for step in steps:
        lines.append(f"  [{_MARK[step.state]:>3}] {step.summary}")
        if step.detail:
            lines.append(f"        {step.detail}")
    needs_network = [step for step in steps if step.state == NETWORK]
    lines.append("")
    if not needs_network:
        lines.append("Nothing left to fetch: prepare can run offline.")
    elif offline():
        lines.append(
            f"{len(needs_network)} step(s) need the network, but the host is marked "
            f"offline — stage the files above, or unset BENCHMARK_TOOLBOX_OFFLINE."
        )
    else:
        lines.append(f"{len(needs_network)} step(s) will use the network.")
    return "\n".join(lines)
