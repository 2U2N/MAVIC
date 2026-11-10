"""Regenerate all distributed figures from aggregate score tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

try:
    from .build_correlation_appendix_table import run_generation as build_table
    from .common import GENERATED_DIR, PAPER_RESULTS_DIR
    from .plot_audience_relevance import run_plotting as plot_audience
    from .plot_combined_coverage import run_plotting as plot_combined
    from .plot_coverage_overview import run_plotting as plot_overview
except ImportError:  # Allow direct execution from a source checkout.
    from build_correlation_appendix_table import run_generation as build_table  # type: ignore
    from common import GENERATED_DIR, PAPER_RESULTS_DIR  # type: ignore
    from plot_audience_relevance import run_plotting as plot_audience  # type: ignore
    from plot_combined_coverage import run_plotting as plot_combined  # type: ignore
    from plot_coverage_overview import run_plotting as plot_overview  # type: ignore


DEFAULT_OUTPUT_DIR = GENERATED_DIR


def reproduce(output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    """Generate coverage, audience, and appendix artifacts without raw data."""
    output_dir = Path(output_dir)
    coverage_output = output_dir / "coverage"
    audience_output = output_dir / "audience_relevance"
    table_output = output_dir / "tables"
    paths: list[Path] = []

    _, combined_paths = plot_combined(
        PAPER_RESULTS_DIR / "coverage_scores.csv",
        coverage_output,
        feature_layout="mixed",
        sort_by="mean",
    )
    paths.extend(combined_paths)
    _, separate_paths = plot_combined(
        PAPER_RESULTS_DIR / "coverage_scores.csv",
        coverage_output,
        feature_layout="separate",
        sort_by="mean",
    )
    paths.extend(separate_paths)
    paths.extend(
        plot_overview(
            PAPER_RESULTS_DIR / "coverage_scores.csv",
            coverage_output,
        )
    )
    paths.extend(
        plot_audience(
            PAPER_RESULTS_DIR / "audience_relevance_scores.csv",
            audience_output,
            label_modes=("audience", "coverage-gap", "none"),
            figure_layouts=("wide", "article"),
            label_count=5,
            quadrant_guides="fixed",
        )
    )
    _, table_paths = build_table(
        PAPER_RESULTS_DIR / "correlation_audit_full-dataset.csv",
        PAPER_RESULTS_DIR / "correlation_audit_linguistic-dataset.csv",
        table_output,
        threshold=0.8,
    )
    paths.extend(table_paths)
    return paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate paper figures and the appendix table from aggregate data"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        paths = reproduce(args.output_dir)
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    print(f"Generated {len(paths)} artifacts below {args.output_dir}")


if __name__ == "__main__":
    main()
