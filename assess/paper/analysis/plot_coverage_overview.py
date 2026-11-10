"""Plot compact, independently ranked MAVIC coverage profiles.

This module is deliberately plotting-only.  It reads the score table produced
by :mod:`analyze_embedding_coverage` and never reloads embeddings or refits a
coverage model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

# Keep figure regeneration usable on remote and headless machines.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

try:
    from .common import (
        FEATURE_TYPES,
        GENERATED_DIR,
        PAPER_RESULTS_DIR,
        TYPE_LABELS,
        darken_color,
        feature_color,
        load_palette,
    )
    from .plot_combined_coverage import coverage_x_limits, validate_coverage_scores
except ImportError:  # Allow direct execution from a source checkout.
    from common import (  # type: ignore
        FEATURE_TYPES,
        GENERATED_DIR,
        PAPER_RESULTS_DIR,
        TYPE_LABELS,
        darken_color,
        feature_color,
        load_palette,
    )
    from plot_combined_coverage import (  # type: ignore
        coverage_x_limits,
        validate_coverage_scores,
    )

feature_type_colors, variability_feature_type_colors = load_palette()
DEFAULT_COVERAGE_CSV = PAPER_RESULTS_DIR / "coverage_scores.csv"
DEFAULT_OUTPUT_DIR = GENERATED_DIR / "coverage"


FIGURE_SIZE = (7.2, 3.1)
OUTPUT_STEM = "coverage_profiles_overview"
MODEL_SPECS = (
    ("openclip_r2", "OpenCLIP"),
    ("lco_r2", "LCO"),
    ("f2llm_r2", "F2LLM"),
)


def order_panel_scores(frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    """Return one representation's stable descending feature ranking.

    Raw feature names provide a deterministic ascending tie-break.  The rank is
    deliberately representation-specific: rows at the same vertical position
    in different panels need not describe the same MAVIC feature.
    """
    valid_score_columns = {column for column, _ in MODEL_SPECS}
    if score_column not in valid_score_columns:
        raise ValueError(f"unknown coverage score column: {score_column}")

    ordered = validate_coverage_scores(frame)
    # Two stable passes make the raw feature name the explicit tie-breaker.
    ordered = ordered.sort_values("feature", kind="mergesort")
    ordered = ordered.sort_values(
        score_column, ascending=False, kind="mergesort"
    ).reset_index(drop=True)
    ordered["profile_rank"] = np.arange(1, len(ordered) + 1, dtype=int)
    return ordered


def _row_colors(
    ordered: pd.DataFrame,
    *,
    colors_by_type: Mapping[str, str],
    variability_colors_by_type: Mapping[str, str],
) -> list[str | tuple[float, ...]]:
    return [
        feature_color(
            str(feature_type),
            bool(is_variability),
            colors_by_type=colors_by_type,
            variability_colors_by_type=variability_colors_by_type,
        )
        for feature_type, is_variability in zip(
            ordered["feature_type"], ordered["is_variability"]
        )
    ]


def _legend_handles(
    *,
    colors_by_type: Mapping[str, str],
    variability_colors_by_type: Mapping[str, str],
) -> list[Patch]:
    handles: list[Patch] = []
    for feature_type in FEATURE_TYPES:
        label = TYPE_LABELS[feature_type]
        handles.extend(
            [
                Patch(
                    facecolor=colors_by_type[feature_type],
                    edgecolor=darken_color(
                        colors_by_type[feature_type], retained_fraction=0.62
                    ),
                    linewidth=0.4,
                    label=f"{label} — ordinary",
                ),
                Patch(
                    facecolor=variability_colors_by_type[feature_type],
                    edgecolor=darken_color(
                        colors_by_type[feature_type], retained_fraction=0.62
                    ),
                    linewidth=0.4,
                    label=f"{label} — variability",
                ),
            ]
        )
    return handles


def plot_coverage_overview(
    frame: pd.DataFrame,
    *,
    colors_by_type: Mapping[str, str] = feature_type_colors,
    variability_colors_by_type: Mapping[str, str] = variability_feature_type_colors,
) -> plt.Figure:
    """Build the compact three-panel coverage-profile figure."""
    validated = validate_coverage_scores(frame)
    x_limits = coverage_x_limits(validated)
    feature_count = len(validated)

    figure, axes = plt.subplots(
        1,
        3,
        figsize=FIGURE_SIZE,
        sharex=True,
        sharey=True,
        facecolor="white",
    )

    for axis, (score_column, title) in zip(axes, MODEL_SPECS):
        ordered = order_panel_scores(validated, score_column)
        y_positions = np.arange(feature_count, dtype=float)
        row_colors = _row_colors(
            ordered,
            colors_by_type=colors_by_type,
            variability_colors_by_type=variability_colors_by_type,
        )
        edge_colors = [
            darken_color(
                colors_by_type[str(feature_type)], retained_fraction=0.62
            )
            for feature_type in ordered["feature_type"]
        ]

        axis.barh(
            y_positions,
            ordered[score_column].to_numpy(dtype=float),
            height=0.76,
            color=row_colors,
            edgecolor=edge_colors,
            linewidth=0.20,
            zorder=2,
        )
        axis.set_title(title, fontsize=8.6, fontweight="bold", pad=4)
        axis.set_xlim(*x_limits)
        axis.set_ylim(feature_count - 0.35, -0.65)
        axis.set_yticks([])
        axis.tick_params(axis="y", left=False, labelleft=False)
        axis.tick_params(axis="x", labelsize=6.5, length=2.2, width=0.5)
        axis.xaxis.set_major_locator(
            MaxNLocator(nbins=3, min_n_ticks=3, prune="both")
        )
        axis.axvline(0.0, color="#252525", linewidth=0.65, zorder=3)
        axis.grid(axis="x", color="#d5d5d5", linewidth=0.45, alpha=0.72)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#777777")
        axis.spines["bottom"].set_linewidth(0.55)

    figure.supxlabel("Cross-validated $R^2$", fontsize=7.4, y=0.115)
    figure.legend(
        handles=_legend_handles(
            colors_by_type=colors_by_type,
            variability_colors_by_type=variability_colors_by_type,
        ),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=6,
        frameon=False,
        fontsize=5.7,
        handlelength=1.15,
        handleheight=0.7,
        handletextpad=0.35,
        columnspacing=0.75,
        borderaxespad=0.0,
    )
    figure.subplots_adjust(
        left=0.045,
        right=0.992,
        bottom=0.245,
        top=0.91,
        wspace=0.17,
    )
    return figure


def save_coverage_overview(
    frame: pd.DataFrame,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    colors_by_type: Mapping[str, str] = feature_type_colors,
    variability_colors_by_type: Mapping[str, str] = variability_feature_type_colors,
) -> list[Path]:
    """Write 300-DPI PNG and vector PDF overview artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure = plot_coverage_overview(
        frame,
        colors_by_type=colors_by_type,
        variability_colors_by_type=variability_colors_by_type,
    )
    paths = [
        output_dir / f"{OUTPUT_STEM}.png",
        output_dir / f"{OUTPUT_STEM}.pdf",
    ]
    figure.savefig(paths[0], dpi=300, facecolor="white")
    figure.savefig(paths[1], facecolor="white")
    plt.close(figure)
    return paths


def run_plotting(
    coverage_csv: Path = DEFAULT_COVERAGE_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    """Regenerate the compact overview from the existing coverage score CSV."""
    coverage_csv = Path(coverage_csv)
    if not coverage_csv.is_file():
        raise ValueError(f"coverage score table not found: {coverage_csv}")
    try:
        frame = pd.read_csv(coverage_csv)
    except (OSError, pd.errors.ParserError) as error:
        raise ValueError(
            f"could not read coverage score table {coverage_csv}: {error}"
        ) from error

    paths = save_coverage_overview(frame, output_dir)
    print("Saved compact coverage overview:")
    for path in paths:
        print(f"  {path}")
    return paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot compact, independently ranked OpenCLIP, LCO, and F2LLM "
            "coverage profiles without rerunning the embedding analysis"
        )
    )
    parser.add_argument(
        "--coverage-csv", type=Path, default=DEFAULT_COVERAGE_CSV
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        run_plotting(args.coverage_csv, args.output_dir)
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
