"""Regenerate audience-relevance figures from the released aggregate scores."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from .analyze_audience_relevance import (
        FIGURE_LAYOUTS,
        QUADRANT_GUIDE_MODES,
        add_coverage_gap_ranks,
        plot_audience_relevance,
    )
    from .common import GENERATED_DIR, PAPER_RESULTS_DIR
except ImportError:  # Allow direct execution from a source checkout.
    from analyze_audience_relevance import (  # type: ignore
        FIGURE_LAYOUTS,
        QUADRANT_GUIDE_MODES,
        add_coverage_gap_ranks,
        plot_audience_relevance,
    )
    from common import GENERATED_DIR, PAPER_RESULTS_DIR  # type: ignore


DEFAULT_SCORES = PAPER_RESULTS_DIR / "audience_relevance_scores.csv"
DEFAULT_OUTPUT_DIR = GENERATED_DIR / "audience_relevance"
LABEL_MODES = ("audience", "coverage-gap", "none")
REQUIRED_COLUMNS = {
    "feature",
    "display_name",
    "feature_type",
    "is_variability",
    "source_dataset",
    "demographic_n_donors",
    "audience_ols_adjusted_r2",
    "openclip_r2",
    "lco_r2",
    "f2llm_r2",
}


def load_audience_scores(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"audience score table not found: {path}")
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError("audience score table is missing columns: " + ", ".join(missing))
    if frame.empty or frame["feature"].duplicated().any():
        raise ValueError("audience score table must contain unique features")
    if len(frame) != 133:
        raise ValueError(f"expected 133 audience scores, found {len(frame)}")
    numeric = [
        "demographic_n_donors",
        "audience_ols_adjusted_r2",
        "openclip_r2",
        "lco_r2",
        "f2llm_r2",
    ]
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("audience score table contains non-finite values")
    required_rank_columns = {
        f"{model}_coverage_gap_rank" for model in ("openclip", "lco", "f2llm")
    }
    if not required_rank_columns.issubset(frame.columns):
        frame = add_coverage_gap_ranks(frame)
    return frame


def run_plotting(
    scores: Path = DEFAULT_SCORES,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    label_modes: Sequence[str] = LABEL_MODES,
    figure_layouts: Sequence[str] = FIGURE_LAYOUTS,
    label_count: int = 5,
    quadrant_guides: str = "fixed",
) -> list[Path]:
    """Write selected wide/article figures without donor-level inputs."""
    frame = load_audience_scores(scores)
    invalid_modes = sorted(set(label_modes) - set(LABEL_MODES))
    if invalid_modes:
        raise ValueError("invalid label modes: " + ", ".join(invalid_modes))
    invalid_layouts = sorted(set(figure_layouts) - set(FIGURE_LAYOUTS))
    if invalid_layouts:
        raise ValueError("invalid figure layouts: " + ", ".join(invalid_layouts))
    if quadrant_guides not in QUADRANT_GUIDE_MODES:
        raise ValueError(f"invalid quadrant guide mode: {quadrant_guides}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for layout in figure_layouts:
        for label_mode in label_modes:
            suffix = "-article" if layout == "article" else ""
            destination = (
                output_dir
                / f"audience_relevance_scatter-{label_mode}{suffix}.png"
            )
            plot_audience_relevance(
                frame,
                destination=destination,
                label_mode=label_mode,
                label_count=label_count,
                figure_layout=layout,
                quadrant_guides=quadrant_guides,
            )
            paths.append(destination)
            if layout == "article":
                paths.append(destination.with_suffix(".pdf"))
    return paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate audience figures from released aggregate scores"
    )
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--label-mode",
        choices=(*LABEL_MODES, "all"),
        default="all",
    )
    parser.add_argument(
        "--figure-layout",
        choices=(*FIGURE_LAYOUTS, "both"),
        default="both",
    )
    parser.add_argument("--label-count", type=int, default=5)
    parser.add_argument(
        "--quadrant-guides",
        choices=QUADRANT_GUIDE_MODES,
        default="fixed",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    label_modes = LABEL_MODES if args.label_mode == "all" else (args.label_mode,)
    layouts = FIGURE_LAYOUTS if args.figure_layout == "both" else (args.figure_layout,)
    try:
        paths = run_plotting(
            args.scores,
            args.output_dir,
            label_modes=label_modes,
            figure_layouts=layouts,
            label_count=args.label_count,
            quadrant_guides=args.quadrant_guides,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    print("Generated audience figures:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
