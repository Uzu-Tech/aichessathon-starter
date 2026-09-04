"""Command-line and notebook entry points for HalfKP NNUE experiments.

Examples::

    python -m evaluation.pipeline inspect
    python -m evaluation.pipeline baseline
    python -m evaluation.pipeline train --objective wdl --architecture 512x32x1
    python -m evaluation.pipeline train-suite --architecture 512x32x1
    python -m evaluation.pipeline compare evaluation/models/*.pt
    python -m evaluation.pipeline benchmark evaluation/models/model.pt
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import cast

from .baselines import evaluate_baselines, format_metrics
from .training import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    Objective,
    Split,
    TrainingResult,
    benchmark,
    benchmark_accumulator,
    benchmark_game,
    evaluate,
    inspect_data,
    model_info,
    recommended_workers,
    train,
)

OBJECTIVES: tuple[Objective, ...] = ("wdl", "rmse", "mae")


def resolve_workers(value: str | int, data_dir: str | Path) -> int:
    if isinstance(value, int):
        if value < 0:
            raise ValueError("workers cannot be negative")
        return value
    if value.lower() == "auto":
        return recommended_workers(data_dir)
    try:
        workers = int(value)
    except ValueError as error:
        raise ValueError("workers must be a non-negative integer or 'auto'") from error
    if workers < 0:
        raise ValueError("workers cannot be negative")
    return workers


def train_experiment(
    *,
    objective: Objective = "wdl",
    architecture: str = "512x32x1",
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    epochs: int = 50,
    batch_size: int = 8192,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    workers: str | int = "auto",
    device: str = "auto",
    patience: int | None = 5,
    seed: int = 42,
    max_train_batches: int | None = None,
    max_validation_batches: int | None = None,
    test: bool = False,
    run_name: str | None = None,
    resume_from: str | Path | None = None,
    verbose: bool = True,
) -> TrainingResult:
    """Train one model; this is the short notebook-friendly entry point."""
    selected_workers = resolve_workers(workers, data_dir)
    return train(
        data_dir,
        representation="halfkp32",
        architecture=architecture,
        objective=objective,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        workers=selected_workers,
        device=device,
        output_dir=output_dir,
        patience=patience,
        seed=seed,
        max_train_batches=max_train_batches,
        max_validation_batches=max_validation_batches,
        test=test,
        run_name=run_name,
        resume_from=resume_from,
        verbose=verbose,
    )


def train_suite(
    *,
    objectives: Sequence[Objective] = OBJECTIVES,
    architecture: str = "512x32x1",
    run_prefix: str | None = None,
    **kwargs: object,
) -> list[TrainingResult]:
    """Train comparable WDL, RMSE, and MAE runs with identical settings.

    Runs are intentionally sequential on one GPU. Data loading within every
    run is parallel; competing training processes on the same GPU are slower
    and make timing comparisons unreliable.
    """
    prefix = run_prefix or datetime.now().strftime("halfkp_%Y%m%d_%H%M%S_%f")
    results = []
    for objective in objectives:
        if objective not in OBJECTIVES:
            raise ValueError(f"unknown objective {objective!r}")
        name = f"{prefix}_{architecture}_{objective}"
        print(f"\n=== {objective.upper()} model ===", flush=True)
        results.append(
            train_experiment(
                objective=objective,
                architecture=architecture,
                run_name=name,
                **kwargs,
            )
        )
    return results


def _print_training_result(result: TrainingResult) -> None:
    print(
        f"checkpoint={result.model_path}\n"
        f"objective={result.objective} best_epoch={result.best_epoch} "
        f"best_metric={result.best_validation_metric:.9f}\n"
        f"training_seconds={result.training_seconds:.2f} "
        f"parameters={result.parameters:,} size={result.model_size_mib:.2f} MiB"
    )
    if result.test_metrics is not None:
        print(format_metrics("test", result.test_metrics))


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--architecture", default="512x32x1")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--resume-from", type=Path)


def _training_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "workers": args.workers,
        "device": args.device,
        "patience": args.patience,
        "seed": args.seed,
        "max_train_batches": args.max_train_batches,
        "max_validation_batches": args.max_validation_batches,
        "test": args.test,
        "resume_from": args.resume_from,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate HalfKP NNUE models")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="inspect dataset and worker count")
    inspect_parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)

    baseline_parser = subparsers.add_parser("baseline", help="score zero/material baselines")
    baseline_parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    baseline_parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="validation"
    )
    baseline_parser.add_argument("--max-shards", type=int)

    train_parser = subparsers.add_parser("train", help="train one objective")
    _add_training_arguments(train_parser)
    train_parser.add_argument("--objective", choices=OBJECTIVES, default="wdl")
    train_parser.add_argument("--run-name")

    suite_parser = subparsers.add_parser("train-suite", help="train WDL/RMSE/MAE models")
    _add_training_arguments(suite_parser)
    suite_parser.add_argument("--objectives", nargs="+", choices=OBJECTIVES, default=OBJECTIVES)
    suite_parser.add_argument("--run-prefix")

    evaluate_parser = subparsers.add_parser("evaluate", help="evaluate checkpoints")
    evaluate_parser.add_argument("checkpoints", nargs="+", type=Path)
    evaluate_parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    evaluate_parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="validation"
    )
    evaluate_parser.add_argument("--batch-size", type=int, default=8192)
    evaluate_parser.add_argument("--workers", default="auto")
    evaluate_parser.add_argument("--device", default="auto")
    evaluate_parser.add_argument("--max-batches", type=int)

    compare_parser = subparsers.add_parser("compare", help="compare models with fixed baselines")
    compare_parser.add_argument("checkpoints", nargs="+", type=Path)
    compare_parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    compare_parser.add_argument("--split", choices=("validation", "test"), default="validation")
    compare_parser.add_argument("--batch-size", type=int, default=8192)
    compare_parser.add_argument("--workers", default="auto")
    compare_parser.add_argument("--device", default="auto")

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="time full and accumulator inference"
    )
    benchmark_parser.add_argument("checkpoint", type=Path)
    benchmark_parser.add_argument("--device", default="cpu")
    benchmark_parser.add_argument("--iterations", type=int, default=1000)
    benchmark_parser.add_argument("--trials", type=int, default=5)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        print(inspect_data(args.data_dir))
        print(f"recommended loader workers: {recommended_workers(args.data_dir)}")
        return 0

    if args.command == "baseline":
        metrics = evaluate_baselines(
            args.data_dir,
            split=cast(Split, args.split),
            max_shards=args.max_shards,
        )
        for name, values in metrics.items():
            print(format_metrics(name, values))
        return 0

    if args.command == "train":
        result = train_experiment(
            objective=cast(Objective, args.objective),
            architecture=args.architecture,
            run_name=args.run_name,
            **_training_kwargs(args),
        )
        _print_training_result(result)
        return 0


    if args.command == "train-suite":
        results = train_suite(
            objectives=cast(Sequence[Objective], args.objectives),
            architecture=args.architecture,
            run_prefix=args.run_prefix,
            **_training_kwargs(args),
        )
        for result in results:
            _print_training_result(result)
        return 0

    if args.command == "evaluate":
        workers = resolve_workers(args.workers, args.data_dir)
        for checkpoint in args.checkpoints:
            metrics = evaluate(
                checkpoint,
                args.data_dir,
                split=cast(Split, args.split),
                batch_size=args.batch_size,
                workers=workers,
                device=args.device,
                max_batches=args.max_batches,
            )
            print(format_metrics(checkpoint.name, metrics))
        return 0

    if args.command == "compare":
        for name, metrics in evaluate_baselines(
            args.data_dir, split=cast(Split, args.split)
        ).items():
            print(format_metrics(name, metrics))
        workers = resolve_workers(args.workers, args.data_dir)
        for checkpoint in args.checkpoints:
            metrics = evaluate(
                checkpoint,
                args.data_dir,
                split=cast(Split, args.split),
                batch_size=args.batch_size,
                workers=workers,
                device=args.device,
            )
            print(format_metrics(checkpoint.name, metrics))
        return 0

    if args.command == "benchmark":
        info = model_info(args.checkpoint, device="cpu")
        full = benchmark(
            args.checkpoint,
            batch_size=1,
            iterations=args.iterations,
            device=args.device,
        )
        accumulator = benchmark_accumulator(
            args.checkpoint,
            iterations=args.iterations,
            trials=args.trials,
            device=args.device,
        )
        print(json.dumps({
            "model": asdict(info),
            "full_forward": asdict(full),
            "maintained_accumulator": asdict(accumulator),
            "move_and_incremental_inference": asdict(
                benchmark_game(args.checkpoint, device=args.device)
            ),
        }, indent=2))
        return 0

    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
