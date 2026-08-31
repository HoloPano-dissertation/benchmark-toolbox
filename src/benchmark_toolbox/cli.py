from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from benchmark_toolbox.config import load_experiment_config
from benchmark_toolbox.datasets import BaseDatasetLoader
from benchmark_toolbox.evaluator import Evaluator
from benchmark_toolbox.metrics import BaseMetric
from benchmark_toolbox.models import BaseSceneEstimator


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark-toolbox",
        description="Run reproducible indoor scene understanding benchmarks",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run one experiment")
    run_parser.add_argument("--config", required=True, help="YAML or JSON config")

    env_parser = subparsers.add_parser(
        "env", help="prepare and inspect isolated model environments"
    )
    env_subparsers = env_parser.add_subparsers(dest="env_command", required=True)

    prepare_parser = env_subparsers.add_parser(
        "prepare", help="provision a model environment (one-time, idempotent)"
    )
    prepare_parser.add_argument("--env", help="path to an environment spec")
    prepare_parser.add_argument(
        "--config", help="experiment config; prepares its model environment"
    )
    prepare_parser.add_argument(
        "--force", action="store_true", help="rebuild even if already prepared"
    )
    prepare_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be built and what needs the network; change nothing",
    )
    prepare_parser.add_argument(
        "--adopt",
        action="store_true",
        help="record an environment built by other means as prepared (no rebuild)",
    )
    prepare_parser.add_argument(
        "--verify",
        action="store_true",
        help="re-check the sha256 of checkpoints already on disk",
    )

    env_subparsers.add_parser(
        "doctor", help="report available backends (conda/venv/...) and GPU"
    )

    submit_parser = env_subparsers.add_parser(
        "submit-script", help="generate a cluster submit script (sbatch/bash); no SSH"
    )
    submit_parser.add_argument("--env", required=True, help="environment spec path")
    submit_parser.add_argument("--config", help="experiment config to run on the node")
    submit_parser.add_argument(
        "--scheduler", choices=["slurm", "bash"], default="slurm"
    )
    submit_parser.add_argument("--gpus", type=int, default=0)
    submit_parser.add_argument(
        "--module", default="anaconda", help="conda module to load (slurm)"
    )
    submit_parser.add_argument("--out", help="write to file instead of stdout")
    return parser


def _environment_source(args: argparse.Namespace) -> "str | dict":
    if args.env:
        return args.env
    if args.config:
        config = load_experiment_config(args.config)
        environment = config.model.parameters.get("environment")
        if environment is None:
            raise SystemExit(
                "Config's model has no 'environment'; pass --env explicitly."
            )
        return environment
    raise SystemExit("Provide either --env <spec> or --config <experiment>.")


def _run(config_path: str) -> int:
    config = load_experiment_config(config_path)
    model_parameters = dict(config.model.parameters)
    model_parameters.setdefault("seed", config.seed)
    estimator = BaseSceneEstimator.create(config.model.type, model_parameters)
    dataset = BaseDatasetLoader.create(
        config.dataset.type, dict(config.dataset.parameters)
    )
    metrics = [
        BaseMetric.create(metric.type, dict(metric.parameters))
        for metric in config.metrics
    ]
    result = Evaluator(config, estimator, dataset, metrics).run()
    print(
        f"Evaluated {result.scene_count} scene(s). Artifacts: {result.output_dir}"
    )
    return 0


def _env(args: argparse.Namespace) -> int:
    from pathlib import Path

    from benchmark_toolbox.environments import (
        BaseEnvironmentManager,
        build_plan,
        doctor_report,
        generate_submit_script,
        load_environment_spec,
        render_plan,
        verify_checkpoints,
    )

    if args.env_command == "doctor":
        print(doctor_report())
        return 0

    if args.env_command == "submit-script":
        spec = load_environment_spec(args.env)
        script = generate_submit_script(
            args.env,
            args.config,
            spec=spec,
            scheduler=args.scheduler,
            gpus=args.gpus,
            module_load=args.module if args.scheduler == "slurm" else "",
        )
        if args.out:
            Path(args.out).write_text(script, encoding="utf-8")
            print(f"Submit script written: {args.out}")
        else:
            print(script)
        return 0

    if args.env_command == "prepare":
        spec = load_environment_spec(_environment_source(args))
        manager = BaseEnvironmentManager.create(spec.backend)

        if args.dry_run:
            print(render_plan(spec, build_plan(spec, manager)))
            return 0

        if args.verify:
            problems = verify_checkpoints(spec, manager)
            for problem in problems:
                print(f"CHECKSUM MISMATCH: {problem}")
            if problems:
                return 1
            print("All checkpoints on disk match their recorded sha256.")
            return 0

        if args.adopt:
            manager.adopt(spec)
            print(
                f"Environment '{spec.name}' ({spec.backend}) adopted as prepared "
                f"(built outside the toolbox; nothing was rebuilt)."
            )
            return 0

        if not args.force and manager.is_prepared(spec):
            print(f"Environment '{spec.name}' ({spec.backend}) already prepared.")
            return 0
        print(f"Preparing environment '{spec.name}' ({spec.backend})...")
        manager.prepare(spec)
        print(f"Environment '{spec.name}' is ready under {spec.workspace}")
        return 0
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = _parser().parse_args(argv)
    if args.command == "run":
        return _run(args.config)
    if args.command == "env":
        return _env(args)
    return 2
