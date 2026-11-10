"""Plot all MAVIC coverage scores across OpenCLIP, LCO, and F2LLM.

This module is deliberately plotting-only: it reads the score table produced by
``analyze_embedding_coverage.py`` and never reloads embeddings or refits models.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence
import textwrap

import matplotlib

# Keep figure regeneration usable on remote and headless machines.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
        is_variability_feature,
        load_palette,
        shared_x_limits,
    )
except ImportError:  # Allow direct execution from a source checkout.
    from common import (  # type: ignore
        FEATURE_TYPES,
        GENERATED_DIR,
        PAPER_RESULTS_DIR,
        TYPE_LABELS,
        darken_color,
        feature_color,
        is_variability_feature,
        load_palette,
        shared_x_limits,
    )

feature_type_colors, variability_feature_type_colors = load_palette()

DEFAULT_COVERAGE_CSV = PAPER_RESULTS_DIR / "coverage_scores.csv"
DEFAULT_OUTPUT_DIR = GENERATED_DIR / "coverage"

MODEL_SPECS = (
    ("openclip", "openclip_r2", "OpenCLIP", "o", -0.22),
    ("lco", "lco_r2", "LCO", "s", 0.0),
    ("f2llm", "f2llm_r2", "F2LLM", "D", 0.22),
)
MODEL_SCORE_COLUMNS = tuple(spec[1] for spec in MODEL_SPECS)
SORT_COLUMNS = {
    "mean": "mean_r2",
    "range": "range_r2",
    "max": "max_r2",
    "openclip": "openclip_r2",
    "lco": "lco_r2",
    "f2llm": "f2llm_r2",
}
SORT_LABELS = {
    "mean": "mean cross-validated $R^2$",
    "range": "cross-model $R^2$ range",
    "max": "best cross-validated $R^2$",
    "openclip": "OpenCLIP cross-validated $R^2$",
    "lco": "LCO cross-validated $R^2$",
    "f2llm": "F2LLM cross-validated $R^2$",
}
REQUIRED_COLUMNS = {
    "feature",
    "display_name",
    "feature_type",
    "is_variability",
    "source_dataset",
    "n_videos",
    *MODEL_SCORE_COLUMNS,
}

LABEL_WRAP_WIDTH = 48
ROW_GAP_UNITS = 0.42
MIXED_FIGURE_SIZE = (8.27, 11.69)  # A4 portrait, in inches.


def _coerce_boolean(value: object) -> bool:
    """Parse a strict CSV-style boolean value."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError("is_variability must contain only true/false values")


def validate_coverage_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a coverage score table for plotting."""
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(
            "coverage score table is missing required columns: " + ", ".join(missing)
        )
    if frame.empty:
        raise ValueError("coverage score table contains no features")

    validated = frame.copy()
    for column in ("feature", "display_name", "feature_type", "source_dataset"):
        values = validated[column]
        if values.isna().any() or any(
            not isinstance(value, str) or not value.strip() for value in values
        ):
            raise ValueError(f"{column} must contain non-empty strings")
        validated[column] = values.str.strip()

    duplicates = validated.loc[
        validated["feature"].duplicated(keep=False), "feature"
    ].unique()
    if len(duplicates):
        raise ValueError(
            "coverage score table contains duplicate features: "
            + ", ".join(sorted(duplicates))
        )

    invalid_types = sorted(set(validated["feature_type"]).difference(FEATURE_TYPES))
    if invalid_types:
        raise ValueError("invalid feature types: " + ", ".join(invalid_types))

    expected_source = validated["feature_type"].map(
        {"audio": "full", "visual": "full", "linguistic": "linguistic"}
    )
    inconsistent_source = validated["source_dataset"] != expected_source
    if inconsistent_source.any():
        features = validated.loc[inconsistent_source, "feature"].tolist()
        raise ValueError(
            "feature types do not match their expected source datasets: "
            + ", ".join(features)
        )

    try:
        validated["is_variability"] = validated["is_variability"].map(
            _coerce_boolean
        )
    except ValueError as error:
        raise ValueError(str(error)) from error
    expected_variability = validated["feature"].map(is_variability_feature)
    inconsistent_variability = validated["is_variability"] != expected_variability
    if inconsistent_variability.any():
        features = validated.loc[inconsistent_variability, "feature"].tolist()
        raise ValueError(
            "is_variability disagrees with MAVIC feature naming conventions: "
            + ", ".join(features)
        )

    for column in (*MODEL_SCORE_COLUMNS, "n_videos"):
        try:
            validated[column] = pd.to_numeric(validated[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"{column} must be numeric") from error
        if not np.isfinite(validated[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain only finite values")

    video_counts = validated["n_videos"].to_numpy(dtype=float)
    if (video_counts <= 0).any() or not np.equal(video_counts, np.floor(video_counts)).all():
        raise ValueError("n_videos must contain positive integers")
    validated["n_videos"] = video_counts.astype(np.int64)
    for source, values in validated.groupby("source_dataset")["n_videos"]:
        if values.nunique() != 1:
            raise ValueError(f"n_videos is inconsistent within the {source} cohort")

    return validated


def load_coverage_scores(path: Path) -> pd.DataFrame:
    """Read and validate the saved coverage analysis score table."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"coverage score table not found: {path}")
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        raise ValueError(f"could not read coverage score table {path}: {error}") from error
    return validate_coverage_scores(frame)


def rank_coverage_scores(frame: pd.DataFrame, *, sort_by: str) -> pd.DataFrame:
    """Add model-neutral summaries and stable global/type-specific ranks."""
    if sort_by not in SORT_COLUMNS:
        raise ValueError(f"unknown coverage sorting rule: {sort_by}")
    ranked = validate_coverage_scores(frame)
    model_scores = ranked.loc[:, MODEL_SCORE_COLUMNS]
    ranked["mean_r2"] = model_scores.mean(axis=1)
    ranked["range_r2"] = model_scores.max(axis=1) - model_scores.min(axis=1)
    ranked["max_r2"] = model_scores.max(axis=1)
    metric_column = SORT_COLUMNS[sort_by]

    # A stable two-pass sort makes the raw feature name the deterministic tie-breaker.
    ranked = ranked.sort_values("feature", kind="mergesort")
    ranked = ranked.sort_values(metric_column, ascending=False, kind="mergesort")
    ranked = ranked.reset_index(drop=True)
    ranked["sort_metric"] = sort_by
    ranked["sort_value"] = ranked[metric_column]
    ranked["global_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
    ranked["feature_type_rank"] = (
        ranked.groupby("feature_type", sort=False).cumcount() + 1
    )
    return ranked


def wrap_display_label(label: str, *, width: int = LABEL_WRAP_WIDTH) -> str:
    """Wrap one readable label without splitting words or hyphenated terms."""
    return textwrap.fill(
        str(label),
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )


def prepare_display_rows(
    ranked: pd.DataFrame, *, rank_column: str, wrap_width: int = LABEL_WRAP_WIDTH
) -> pd.DataFrame:
    """Add wrapped labels and row weights used by the page layout."""
    prepared = ranked.copy()
    prepared["_display_rank"] = prepared[rank_column].astype(int)
    prepared["_wrapped_label"] = prepared["display_name"].map(
        lambda label: wrap_display_label(str(label), width=wrap_width)
    )
    prepared["_label_lines"] = prepared["_wrapped_label"].str.count("\n") + 1
    prepared["_row_weight"] = prepared["_label_lines"] + ROW_GAP_UNITS
    return prepared


def split_contiguous_by_weight(
    prepared: pd.DataFrame, *, columns: int
) -> list[pd.DataFrame]:
    """Split ranked rows into contiguous, label-height-balanced columns."""
    if columns not in {1, 2}:
        raise ValueError("coverage figures support one or two continuation columns")
    if columns == 1 or len(prepared) < 2:
        return [prepared.copy()]

    weights = prepared["_row_weight"].to_numpy(dtype=float)
    cumulative = np.cumsum(weights)
    total = cumulative[-1]
    split_index = min(
        range(1, len(prepared)),
        key=lambda index: (abs(cumulative[index - 1] - total / 2.0), index),
    )
    return [
        prepared.iloc[:split_index].copy(),
        prepared.iloc[split_index:].copy(),
    ]


def coverage_x_limits(frame: pd.DataFrame) -> tuple[float, float]:
    """Return shared limits spanning all three models and zero."""
    return shared_x_limits(
        [frame[column].to_numpy(dtype=float) for column in MODEL_SCORE_COLUMNS]
    )


def cohort_sizes(frame: pd.DataFrame) -> tuple[int, int]:
    """Return the full and linguistic cohort sizes recorded in the score table."""
    full = frame.loc[frame["source_dataset"] == "full", "n_videos"].unique()
    linguistic = frame.loc[
        frame["source_dataset"] == "linguistic", "n_videos"
    ].unique()
    if len(full) != 1 or len(linguistic) != 1:
        raise ValueError("coverage scores must contain one full and one linguistic cohort")
    return int(full[0]), int(linguistic[0])


def _row_centers(chunk: pd.DataFrame) -> tuple[np.ndarray, float]:
    weights = chunk["_row_weight"].to_numpy(dtype=float)
    starts = np.concatenate(([0.0], np.cumsum(weights[:-1])))
    centers = starts + weights / 2.0
    return centers, float(weights.sum())


def _plot_chunk(
    axis: plt.Axes,
    chunk: pd.DataFrame,
    *,
    x_limits: tuple[float, float],
    rank_label: str,
    label_fontsize: float,
    colors_by_type: Mapping[str, str],
    variability_colors_by_type: Mapping[str, str],
) -> None:
    """Draw one contiguous rank block on an existing axis."""
    centers, total_height = _row_centers(chunk)
    values = chunk.loc[:, MODEL_SCORE_COLUMNS].to_numpy(dtype=float)
    row_colors = [
        feature_color(
            str(feature_type),
            bool(is_variability),
            colors_by_type=colors_by_type,
            variability_colors_by_type=variability_colors_by_type,
        )
        for feature_type, is_variability in zip(
            chunk["feature_type"], chunk["is_variability"]
        )
    ]
    edge_colors = [darken_color(color, retained_fraction=0.62) for color in row_colors]

    axis.hlines(
        centers,
        values.min(axis=1),
        values.max(axis=1),
        color="#a7a7a7",
        linewidth=0.65,
        alpha=0.85,
        zorder=1,
    )
    for _, score_column, _, marker, offset in MODEL_SPECS:
        axis.scatter(
            chunk[score_column].to_numpy(dtype=float),
            centers + offset,
            marker=marker,
            s=17,
            c=row_colors,
            edgecolors=edge_colors,
            linewidths=0.45,
            zorder=3,
        )

    tick_labels = [
        f"{int(rank)}. {label}"
        for rank, label in zip(chunk["_display_rank"], chunk["_wrapped_label"])
    ]
    axis.set_yticks(centers, labels=tick_labels)
    for tick_label in axis.get_yticklabels():
        tick_label.set_fontsize(label_fontsize)
        tick_label.set_linespacing(0.92)
    axis.tick_params(axis="y", length=0, pad=2)
    axis.tick_params(axis="x", labelsize=6)
    axis.set_ylim(total_height + 0.1, -0.1)
    axis.set_xlim(*x_limits)
    axis.axvline(0.0, color="#252525", linewidth=0.8, zorder=2)
    axis.grid(axis="x", color="#d5d5d5", linewidth=0.55, alpha=0.75)
    axis.set_axisbelow(True)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
    axis.set_xlabel("Cross-validated $R^2$", fontsize=7, labelpad=3)
    axis.set_title(rank_label, loc="left", fontsize=7.3, pad=5)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#777777")
    axis.spines["bottom"].set_linewidth(0.6)


def _legend_groups(
    *,
    colors_by_type: Mapping[str, str],
) -> tuple[
    tuple[list[Line2D], list[str]],
    tuple[list[Patch], list[str]],
    tuple[list[Patch], list[str]],
]:
    model_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            markerfacecolor="#555555",
            markeredgecolor="#222222",
            markersize=4.5,
        )
        for _, _, _, marker, _ in MODEL_SPECS
    ]
    model_labels = [spec[2] for spec in MODEL_SPECS]
    type_handles = [
        Patch(facecolor=colors_by_type[feature_type], edgecolor="none")
        for feature_type in FEATURE_TYPES
    ]
    type_labels = [TYPE_LABELS[feature_type] for feature_type in FEATURE_TYPES]
    shade_handles = [
        Patch(facecolor="#555555", edgecolor="#333333"),
        Patch(facecolor="#c8c8c8", edgecolor="#777777"),
    ]
    shade_labels = ["Ordinary feature", "SD / variability"]
    return (
        (model_handles, model_labels),
        (type_handles, type_labels),
        (shade_handles, shade_labels),
    )


def _add_page_axes(
    figure: plt.Figure, *, columns: int, bottom: float, top: float
) -> list[plt.Axes]:
    height = top - bottom
    if columns == 1:
        return [figure.add_axes((0.43, bottom, 0.53, height))]
    return [
        figure.add_axes((0.285, bottom, 0.185, height)),
        figure.add_axes((0.785, bottom, 0.185, height)),
    ]


def _save_figure(figure: plt.Figure, *, stem: Path) -> list[Path]:
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    figure.savefig(png, dpi=300, facecolor="white")
    figure.savefig(pdf, facecolor="white")
    plt.close(figure)
    return [png, pdf]


def _draw_figure(
    chunks: Sequence[pd.DataFrame],
    *,
    title: str,
    subtitle: str,
    destination_stem: Path,
    figure_size: tuple[float, float],
    x_limits: tuple[float, float],
    label_fontsize: float,
    colors_by_type: Mapping[str, str],
    variability_colors_by_type: Mapping[str, str],
) -> list[Path]:
    figure = plt.figure(figsize=figure_size, facecolor="white")
    figure_height = figure_size[1]
    title_y = 1.0 - 0.18 / figure_height
    subtitle_y = 1.0 - 0.43 / figure_height
    axes_top = 1.0 - 0.85 / figure_height
    axes_bottom = 0.74 / figure_height
    legend_y = 0.025 / figure_height
    axes = _add_page_axes(
        figure,
        columns=len(chunks),
        bottom=axes_bottom,
        top=axes_top,
    )
    for axis, chunk in zip(axes, chunks):
        first_rank = int(chunk["_display_rank"].iloc[0])
        last_rank = int(chunk["_display_rank"].iloc[-1])
        _plot_chunk(
            axis,
            chunk,
            x_limits=x_limits,
            rank_label=f"Ranks {first_rank}–{last_rank}",
            label_fontsize=label_fontsize,
            colors_by_type=colors_by_type,
            variability_colors_by_type=variability_colors_by_type,
        )

    figure.suptitle(title, fontsize=11, fontweight="bold", y=title_y)
    figure.text(0.5, subtitle_y, subtitle, ha="center", va="center", fontsize=7.2)
    model_group, type_group, shade_group = _legend_groups(
        colors_by_type=colors_by_type
    )
    legend_options = {
        "loc": "lower center",
        "frameon": False,
        "fontsize": 5.7,
        "title_fontsize": 5.9,
        "handlelength": 1.2,
        "handletextpad": 0.35,
        "columnspacing": 0.8,
        "borderaxespad": 0.0,
    }
    figure.legend(
        *model_group,
        title="Representation",
        bbox_to_anchor=(0.20, legend_y),
        ncol=3,
        **legend_options,
    )
    figure.legend(
        *type_group,
        title="Feature modality",
        bbox_to_anchor=(0.57, legend_y),
        ncol=3,
        **legend_options,
    )
    figure.legend(
        *shade_group,
        title="Feature form",
        bbox_to_anchor=(0.85, legend_y),
        ncol=2,
        **legend_options,
    )
    return _save_figure(figure, stem=destination_stem)


def plot_mixed_coverage(
    ranked: pd.DataFrame,
    *,
    sort_by: str,
    output_dir: Path,
    colors_by_type: Mapping[str, str] = feature_type_colors,
    variability_colors_by_type: Mapping[str, str] = variability_feature_type_colors,
) -> list[Path]:
    """Create the two-column, globally ranked all-feature figure."""
    prepared = prepare_display_rows(ranked, rank_column="global_rank")
    chunks = split_contiguous_by_weight(prepared, columns=2)
    full_n, linguistic_n = cohort_sizes(ranked)
    subtitle = (
        f"All {len(ranked):,} features ranked by {SORT_LABELS[sort_by]}  •  "
        f"audio/visual n={full_n:,}  •  linguistic n={linguistic_n:,}"
    )
    return _draw_figure(
        chunks,
        title="MAVIC feature coverage across video representations",
        subtitle=subtitle,
        destination_stem=Path(output_dir)
        / f"combined_coverage-mixed-{sort_by}",
        figure_size=MIXED_FIGURE_SIZE,
        x_limits=coverage_x_limits(ranked),
        label_fontsize=6.15,
        colors_by_type=colors_by_type,
        variability_colors_by_type=variability_colors_by_type,
    )


def _separate_figure_height(chunks: Sequence[pd.DataFrame]) -> float:
    max_weight = max(float(chunk["_row_weight"].sum()) for chunk in chunks)
    return min(MIXED_FIGURE_SIZE[1], max(4.5, 1.7 + 0.13 * max_weight))


def plot_separate_coverage(
    ranked: pd.DataFrame,
    *,
    sort_by: str,
    output_dir: Path,
    colors_by_type: Mapping[str, str] = feature_type_colors,
    variability_colors_by_type: Mapping[str, str] = variability_feature_type_colors,
) -> list[Path]:
    """Create one independently ranked figure per MAVIC feature type."""
    output_paths: list[Path] = []
    x_limits = coverage_x_limits(ranked)
    for feature_type in FEATURE_TYPES:
        selected = ranked.loc[ranked["feature_type"] == feature_type].copy()
        if selected.empty:
            raise ValueError(f"coverage score table contains no {feature_type} features")
        prepared = prepare_display_rows(
            selected, rank_column="feature_type_rank"
        )
        columns = 2 if feature_type == "visual" else 1
        chunks = split_contiguous_by_weight(prepared, columns=columns)
        n_videos = int(selected["n_videos"].iloc[0])
        subtitle = (
            f"All {len(selected):,} {feature_type} features ranked by "
            f"{SORT_LABELS[sort_by]}  •  n={n_videos:,}"
        )
        width = MIXED_FIGURE_SIZE[0] if columns == 2 else 6.6
        output_paths.extend(
            _draw_figure(
                chunks,
                title=(
                    f"{TYPE_LABELS[feature_type]} MAVIC feature coverage "
                    "across video representations"
                ),
                subtitle=subtitle,
                destination_stem=Path(output_dir)
                / f"combined_coverage-{feature_type}-{sort_by}",
                figure_size=(width, _separate_figure_height(chunks)),
                x_limits=x_limits,
                label_fontsize=6.15 if columns == 2 else 7.0,
                colors_by_type=colors_by_type,
                variability_colors_by_type=variability_colors_by_type,
            )
        )
    return output_paths


def run_plotting(
    coverage_csv: Path = DEFAULT_COVERAGE_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    feature_layout: str = "mixed",
    sort_by: str = "mean",
    colors_by_type: Mapping[str, str] = feature_type_colors,
    variability_colors_by_type: Mapping[str, str] = variability_feature_type_colors,
) -> tuple[pd.DataFrame, list[Path]]:
    """Regenerate combined coverage figures from an existing score table."""
    if feature_layout not in {"mixed", "separate"}:
        raise ValueError(f"unknown feature layout: {feature_layout}")
    frame = load_coverage_scores(coverage_csv)
    ranked = rank_coverage_scores(frame, sort_by=sort_by)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = output_dir / f"combined_coverage_ranking-{sort_by}.csv"
    ranked.to_csv(ranking_path, index=False)

    if feature_layout == "mixed":
        figure_paths = plot_mixed_coverage(
            ranked,
            sort_by=sort_by,
            output_dir=output_dir,
            colors_by_type=colors_by_type,
            variability_colors_by_type=variability_colors_by_type,
        )
    else:
        figure_paths = plot_separate_coverage(
            ranked,
            sort_by=sort_by,
            output_dir=output_dir,
            colors_by_type=colors_by_type,
            variability_colors_by_type=variability_colors_by_type,
        )

    all_paths = [ranking_path, *figure_paths]
    print("Saved combined coverage artifacts:")
    for path in all_paths:
        print(f"  {path}")
    return ranked, all_paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot saved OpenCLIP, LCO, and F2LLM MAVIC coverage scores "
            "without rerunning the embedding analysis"
        )
    )
    parser.add_argument(
        "--coverage-csv", type=Path, default=DEFAULT_COVERAGE_CSV
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--feature-layout",
        choices=("mixed", "separate"),
        default="mixed",
        help=(
            "globally mix feature types in one page or write one figure per type "
            "(default: mixed)"
        ),
    )
    parser.add_argument(
        "--sort-by",
        choices=tuple(SORT_COLUMNS),
        default="mean",
        help="feature ranking used in the figure (default: mean)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        run_plotting(
            coverage_csv=args.coverage_csv,
            output_dir=args.output_dir,
            feature_layout=args.feature_layout,
            sort_by=args.sort_by,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
