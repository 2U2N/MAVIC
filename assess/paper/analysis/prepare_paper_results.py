"""Validate and reshape the released paper coverage table."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

try:
    from .common import PAPER_RESULTS_DIR
    from .plot_combined_coverage import validate_coverage_scores
except ImportError:  # Allow direct execution from a source checkout.
    from common import PAPER_RESULTS_DIR  # type: ignore
    from plot_combined_coverage import validate_coverage_scores  # type: ignore


DEFAULT_WIDE = PAPER_RESULTS_DIR / "coverage_scores.csv"
DEFAULT_LONG = PAPER_RESULTS_DIR / "coverage_scores_long.csv"
REPRESENTATION_COLUMNS = {
    "openclip_r2": "openclip",
    "lco_r2": "lco",
    "f2llm_r2": "f2llm",
}


def wide_to_long(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert the verified paper table to the reusable probing schema."""
    validated = validate_coverage_scores(frame)
    metadata = [
        "feature",
        "display_name",
        "feature_type",
        "is_variability",
        "source_dataset",
        "n_videos",
    ]
    long = validated.melt(
        id_vars=metadata,
        value_vars=list(REPRESENTATION_COLUMNS),
        var_name="_score_column",
        value_name="r2",
    )
    long["representation"] = long["_score_column"].map(REPRESENTATION_COLUMNS)
    long = long.rename(columns={"source_dataset": "source_cohort"}).drop(
        columns="_score_column"
    )
    ordered_columns = [
        "feature",
        "display_name",
        "feature_type",
        "is_variability",
        "source_cohort",
        "n_videos",
        "representation",
        "r2",
    ]
    return long.loc[:, ordered_columns].sort_values(
        ["representation", "feature"], kind="mergesort"
    ).reset_index(drop=True)


def run_conversion(
    wide_path: Path = DEFAULT_WIDE, destination: Path = DEFAULT_LONG
) -> pd.DataFrame:
    wide_path = Path(wide_path)
    if not wide_path.is_file():
        raise ValueError(f"wide coverage table not found: {wide_path}")
    long = wide_to_long(pd.read_csv(wide_path))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(destination, index=False)
    return long


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create the long-form reusable paper coverage table"
    )
    parser.add_argument("--wide", type=Path, default=DEFAULT_WIDE)
    parser.add_argument("--output", type=Path, default=DEFAULT_LONG)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        long = run_conversion(args.wide, args.output)
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    print(f"Wrote {len(long):,} long-form scores to {args.output}")


if __name__ == "__main__":
    main()
