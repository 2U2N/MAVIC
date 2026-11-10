"""Command-line interface for reusable MAVIC assessment workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .probing import DEFAULT_FOLDS, DEFAULT_SEED, run_probe
from .selection import run_selection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mavic",
        description="Select MAVIC features and probe computational representations.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    select = commands.add_parser(
        "select", help="apply deterministic MAVIC feature selection"
    )
    select.add_argument("--full-candidates", type=Path, required=True)
    select.add_argument("--linguistic-candidates", type=Path, required=True)
    select.add_argument("--config", type=Path, required=True)
    select.add_argument("--output-dir", type=Path, required=True)

    probe = commands.add_parser(
        "probe", help="measure linear coverage of MAVIC features"
    )
    probe.add_argument("--full-features", type=Path, required=True)
    probe.add_argument("--linguistic-features", type=Path, required=True)
    probe.add_argument(
        "--representation",
        action="append",
        required=True,
        metavar="NAME=NPZ_OR_MANIFEST_DIR",
        help="repeat for every representation to compare",
    )
    probe.add_argument("--output-dir", type=Path, required=True)
    probe.add_argument(
        "--vocabulary",
        type=Path,
        help="override the packaged MAVIC v1 vocabulary",
    )
    probe.add_argument("--cv-folds", type=int, default=DEFAULT_FOLDS)
    probe.add_argument("--seed", type=int, default=DEFAULT_SEED)

    plot = commands.add_parser("plot", help="plot aggregate MAVIC results")
    plot_commands = plot.add_subparsers(dest="plot_command", required=True)
    coverage = plot_commands.add_parser(
        "coverage", help="plot long-form representation coverage"
    )
    coverage.add_argument("--scores", type=Path, required=True)
    coverage.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "select":
            outputs = run_selection(
                args.full_candidates,
                args.linguistic_candidates,
                args.config,
                args.output_dir,
            )
            for label, path in outputs.items():
                print(f"{label}: {path}")
        elif args.command == "probe":
            results = run_probe(
                args.full_features,
                args.linguistic_features,
                args.representation,
                args.output_dir,
                vocabulary_path=args.vocabulary,
                n_splits=args.cv_folds,
                seed=args.seed,
            )
            print(
                f"Wrote {len(results):,} scores for "
                f"{results['representation'].nunique()} representation(s)."
            )
        elif args.command == "plot" and args.plot_command == "coverage":
            from .plotting import run_plot_coverage

            outputs = run_plot_coverage(args.scores, args.output_dir)
            for path in outputs:
                print(path)
        else:  # pragma: no cover - argparse prevents this
            raise RuntimeError("unhandled command")
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    return 0
