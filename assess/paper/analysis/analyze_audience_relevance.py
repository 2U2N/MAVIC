"""Relate donor-level MAVIC exposure associations to embedding coverage.

The analysis aggregates sampled videos to one equally weighted profile per
participant, estimates donor-level demographic associations with those profiles,
and combines the resulting effect sizes with the existing embedding-coverage
scores in a three-panel figure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

from matplotlib import patheffects
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

try:
    from .common import (
        GENERATED_DIR,
        PAPER_RESULTS_DIR,
        RESTRICTED_INPUTS_DIR,
        darken_color,
        feature_color,
        load_palette,
    )
except ImportError:  # Allow direct execution from a source checkout.
    from common import (  # type: ignore
        GENERATED_DIR,
        PAPER_RESULTS_DIR,
        RESTRICTED_INPUTS_DIR,
        darken_color,
        feature_color,
        load_palette,
    )

feature_type_colors, variability_feature_type_colors = load_palette()

DEFAULT_META_PARQUET = RESTRICTED_INPUTS_DIR / "donor_metadata.parquet"
DEFAULT_FULL_MAVIC_CSV = RESTRICTED_INPUTS_DIR / "full_features.csv"
DEFAULT_LINGUISTIC_MAVIC_CSV = Path(
    RESTRICTED_INPUTS_DIR / "linguistic_features.csv"
)
DEFAULT_COVERAGE_CSV = PAPER_RESULTS_DIR / "coverage_scores.csv"
DEFAULT_OUTPUT_DIR = GENERATED_DIR / "audience_relevance"

VIDEO_ID_COLUMN = "video_id"
PARTICIPANT_ID_COLUMN = "participant_id"
DEMOGRAPHIC_COLUMNS = ("age", "gender", "education")
REPRESENTATIONS = (
    ("openclip_r2", "OpenCLIP"),
    ("lco_r2", "LCO"),
    ("f2llm_r2", "F2LLM metadata"),
)
FEATURE_TYPES = ("audio", "visual", "linguistic")
TYPE_LABELS = {"audio": "Audio", "visual": "Visual", "linguistic": "Linguistic"}
FIGURE_LAYOUTS = ("wide", "article")
QUADRANT_GUIDE_MODES = ("fixed", "none")
QUADRANT_GUIDE_X = 0.40
QUADRANT_GUIDE_Y = 0.075

DEFAULT_MIN_VIDEOS = 5
DEFAULT_LABEL_COUNT = 5


def _require_file(path: Path, description: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{description} not found: {path}")
    return path


def load_metadata(path: Path) -> pd.DataFrame:
    """Load watch-event metadata and validate donor identifiers/demographics."""
    path = _require_file(path, "metadata parquet")
    try:
        metadata = pd.read_parquet(path)
    except (ImportError, OSError, ValueError) as error:
        raise ValueError(f"could not read metadata parquet {path}: {error}") from error

    if VIDEO_ID_COLUMN not in metadata.columns:
        if metadata.index.name != VIDEO_ID_COLUMN:
            raise ValueError(
                f"{path} must contain {VIDEO_ID_COLUMN!r} as a column or index"
            )
        metadata = metadata.reset_index()

    required = {VIDEO_ID_COLUMN, PARTICIPANT_ID_COLUMN, *DEMOGRAPHIC_COLUMNS}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError("metadata is missing required columns: " + ", ".join(missing))

    events = metadata.loc[:, list(required)].copy()
    events[VIDEO_ID_COLUMN] = events[VIDEO_ID_COLUMN].astype("string").str.strip()
    events[PARTICIPANT_ID_COLUMN] = (
        events[PARTICIPANT_ID_COLUMN].astype("string").str.strip()
    )
    for column in (VIDEO_ID_COLUMN, PARTICIPANT_ID_COLUMN):
        if events[column].isna().any() or events[column].eq("").any():
            raise ValueError(f"metadata {column} values must be non-empty")

    for column in ("gender", "education"):
        events[column] = events[column].astype("string").str.strip()
        events.loc[events[column].eq(""), column] = pd.NA
    events["age"] = pd.to_numeric(events["age"], errors="coerce")

    duplicates = events.duplicated(
        subset=[PARTICIPANT_ID_COLUMN, VIDEO_ID_COLUMN], keep=False
    )
    if duplicates.any():
        sample = (
            events.loc[duplicates, [PARTICIPANT_ID_COLUMN, VIDEO_ID_COLUMN]]
            .drop_duplicates()
            .head(5)
        )
        pairs = [f"{row.participant_id}/{row.video_id}" for row in sample.itertuples()]
        raise ValueError(
            "metadata contains duplicate participant-video pairs: " + ", ".join(pairs)
        )

    for column in DEMOGRAPHIC_COLUMNS:
        counts = events.groupby(PARTICIPANT_ID_COLUMN, observed=True)[column].nunique(
            dropna=True
        )
        inconsistent = counts[counts > 1]
        if not inconsistent.empty:
            raise ValueError(
                f"metadata contains inconsistent {column} values for participants: "
                + ", ".join(inconsistent.index.astype(str).tolist()[:5])
            )

    return events.sort_values(
        [PARTICIPANT_ID_COLUMN, VIDEO_ID_COLUMN], kind="mergesort"
    ).reset_index(drop=True)


def load_feature_table(path: Path) -> pd.DataFrame:
    """Load and validate a unique, finite MAVIC feature table."""
    path = _require_file(path, "MAVIC feature CSV")
    try:
        frame = pd.read_csv(path, dtype={VIDEO_ID_COLUMN: "string"})
    except (OSError, pd.errors.ParserError) as error:
        raise ValueError(f"could not read MAVIC CSV {path}: {error}") from error
    if VIDEO_ID_COLUMN not in frame.columns:
        raise ValueError(f"{path} must contain {VIDEO_ID_COLUMN!r}")
    frame[VIDEO_ID_COLUMN] = frame[VIDEO_ID_COLUMN].astype("string").str.strip()
    if frame[VIDEO_ID_COLUMN].isna().any() or frame[VIDEO_ID_COLUMN].eq("").any():
        raise ValueError(f"{path}: video IDs must be non-empty")
    if frame[VIDEO_ID_COLUMN].duplicated().any():
        raise ValueError(f"{path}: video IDs must be unique")
    features = [column for column in frame.columns if column != VIDEO_ID_COLUMN]
    if not features:
        raise ValueError(f"{path}: no feature columns found")
    try:
        values = frame[features].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: all feature values must be numeric") from error
    if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{path}: feature values must be finite")
    frame.loc[:, features] = values
    return frame


def load_coverage_scores(path: Path) -> pd.DataFrame:
    """Load the central coverage table and validate its public schema."""
    path = _require_file(path, "coverage score CSV")
    coverage = pd.read_csv(path)
    required = {
        "feature",
        "display_name",
        "feature_type",
        "is_variability",
        "source_dataset",
        *(column for column, _ in REPRESENTATIONS),
    }
    missing = sorted(required - set(coverage.columns))
    if missing:
        raise ValueError("coverage scores are missing columns: " + ", ".join(missing))
    if coverage["feature"].duplicated().any():
        raise ValueError("coverage scores contain duplicate features")
    invalid_types = sorted(set(coverage["feature_type"]) - set(FEATURE_TYPES))
    if invalid_types:
        raise ValueError("coverage scores contain invalid feature types: " + ", ".join(invalid_types))
    representation_columns = [column for column, _ in REPRESENTATIONS]
    if not np.isfinite(coverage[representation_columns].to_numpy(dtype=float)).all():
        raise ValueError("embedding coverage scores must be finite")
    return coverage


def _demographic_table(events: pd.DataFrame) -> pd.DataFrame:
    return (
        events.groupby(PARTICIPANT_ID_COLUMN, observed=True)[list(DEMOGRAPHIC_COLUMNS)]
        .first()
        .reset_index()
    )


def build_donor_profiles(
    events: pd.DataFrame,
    feature_frame: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    cohort_name: str,
    min_videos: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Create one equal-weight mean feature profile per eligible donor."""
    feature_names = list(feature_names)
    missing = [name for name in feature_names if name not in feature_frame.columns]
    if missing:
        raise ValueError(
            f"{cohort_name} feature table is missing selected features: "
            + ", ".join(missing)
        )
    if min_videos < 1:
        raise ValueError("min_videos must be positive")

    demographics = _demographic_table(events)
    complete_demographics = demographics.dropna(subset=list(DEMOGRAPHIC_COLUMNS)).copy()
    matched = events[[PARTICIPANT_ID_COLUMN, VIDEO_ID_COLUMN]].merge(
        feature_frame[[VIDEO_ID_COLUMN, *feature_names]],
        on=VIDEO_ID_COLUMN,
        how="inner",
        validate="many_to_one",
    )
    counts = matched.groupby(PARTICIPANT_ID_COLUMN, observed=True).size().rename(
        "n_usable_videos"
    )
    profiles = (
        matched.groupby(PARTICIPANT_ID_COLUMN, observed=True)[feature_names]
        .mean()
        .join(counts)
        .reset_index()
        .merge(
            complete_demographics,
            on=PARTICIPANT_ID_COLUMN,
            how="inner",
            validate="one_to_one",
        )
    )
    profiles = profiles.loc[profiles["n_usable_videos"] >= min_videos].copy()
    profiles = profiles.sort_values(PARTICIPANT_ID_COLUMN, kind="mergesort").reset_index(
        drop=True
    )
    if len(profiles) < 2:
        raise ValueError(f"{cohort_name} cohort contains fewer than two eligible donors")
    if not np.isfinite(profiles[feature_names].to_numpy(dtype=float)).all():
        raise ValueError(f"{cohort_name} donor profiles contain non-finite values")

    eligible_counts = profiles["n_usable_videos"]
    audit: dict[str, object] = {
        "cohort": cohort_name,
        "selected_features": len(feature_names),
        "metadata_events": int(len(events)),
        "matched_events": int(len(matched)),
        "metadata_donors": int(demographics[PARTICIPANT_ID_COLUMN].nunique()),
        "complete_demographic_donors": int(
            complete_demographics[PARTICIPANT_ID_COLUMN].nunique()
        ),
        "eligible_donors": int(len(profiles)),
        "eligible_events": int(eligible_counts.sum()),
        "min_videos_required": int(min_videos),
        "usable_videos_per_donor": {
            "min": int(eligible_counts.min()),
            "median": float(eligible_counts.median()),
            "mean": float(eligible_counts.mean()),
            "max": int(eligible_counts.max()),
        },
        "age": {
            "min": float(profiles["age"].min()),
            "mean": float(profiles["age"].mean()),
            "max": float(profiles["age"].max()),
        },
        "gender_counts": {
            str(key): int(value)
            for key, value in profiles["gender"].value_counts(dropna=False).items()
        },
        "education_counts": {
            str(key): int(value)
            for key, value in profiles["education"].value_counts(dropna=False).items()
        },
    }
    return profiles, audit


def _ols_design_matrix(
    demographics: pd.DataFrame,
) -> tuple[np.ndarray, list[str], dict[str, str]]:
    """Build one full-cohort treatment-coded OLS design matrix."""
    missing = sorted(set(DEMOGRAPHIC_COLUMNS) - set(demographics.columns))
    if missing:
        raise ValueError(
            "demographics are missing required columns: " + ", ".join(missing)
        )
    if demographics[list(DEMOGRAPHIC_COLUMNS)].isna().any().any():
        raise ValueError("demographics must be complete before OLS fitting")

    age = demographics["age"].to_numpy(dtype=np.float64)
    if not np.isfinite(age).all():
        raise ValueError("age values must be finite")
    if float(age.max()) == float(age.min()):
        raise ValueError("age has zero variance in the donor cohort")

    parts = [
        np.ones((len(demographics), 1), dtype=np.float64),
        age[:, None],
    ]
    column_names = ["intercept", "age"]
    reference_categories: dict[str, str] = {}
    for column in ("gender", "education"):
        values = demographics[column].astype(str)
        categories = sorted(values.unique().tolist())
        if not categories:
            raise ValueError(f"{column} has no observed categories")
        reference_categories[column] = categories[0]
        for category in categories[1:]:
            parts.append(values.eq(category).to_numpy(dtype=float)[:, None])
            column_names.append(f"{column}[{category}]")

    design = np.column_stack(parts)
    if not np.isfinite(design).all():
        raise ValueError("demographic OLS design matrix must be finite")
    return design, column_names, reference_categories


def demographic_ols_adjusted_r2(
    demographics: pd.DataFrame,
    targets: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit multivariate donor-level OLS and return adjusted R-squared per target."""
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim != 2:
        raise ValueError("OLS targets must be a two-dimensional matrix")
    if len(demographics) != targets.shape[0]:
        raise ValueError("demographic rows and OLS target rows must match")
    if not np.isfinite(targets).all():
        raise ValueError("OLS targets must be finite")

    design, column_names, reference_categories = _ols_design_matrix(demographics)
    design_rank = int(np.linalg.matrix_rank(design))
    residual_degrees_of_freedom = len(design) - design_rank
    if residual_degrees_of_freedom <= 0:
        raise ValueError("OLS design leaves no residual degrees of freedom")

    coefficients = np.linalg.lstsq(design, targets, rcond=None)[0]
    predictions = design @ coefficients
    residual_sse = np.square(targets - predictions).sum(axis=0)
    total_sse = np.square(targets - targets.mean(axis=0, keepdims=True)).sum(axis=0)
    if np.any(total_sse <= 0.0):
        raise ValueError("a donor-level feature has zero total variance")

    adjusted_r2 = 1.0 - (
        (residual_sse / total_sse)
        * (len(design) - 1)
        / residual_degrees_of_freedom
    )
    if not np.isfinite(adjusted_r2).all():
        raise ValueError("demographic OLS produced non-finite adjusted R-squared")

    audit: dict[str, object] = {
        "model": "OLS",
        "score": "adjusted_r2",
        "n_observations": int(len(design)),
        "n_design_columns": int(design.shape[1]),
        "design_rank": design_rank,
        "residual_degrees_of_freedom": int(residual_degrees_of_freedom),
        "design_columns": column_names,
        "reference_categories": reference_categories,
        "age_coding": "raw continuous age",
        "categorical_coding": "treatment coding with lexicographically first reference",
    }
    return adjusted_r2, audit


def add_coverage_gap_ranks(results: pd.DataFrame) -> pd.DataFrame:
    """Add representation-specific high-audience/low-coverage percentile ranks."""
    results = results.copy()
    audience_percentile = results["audience_ols_adjusted_r2"].rank(
        method="average", pct=True
    )
    for score_column, _ in REPRESENTATIONS:
        prefix = score_column.removesuffix("_r2")
        coverage_percentile = results[score_column].rank(method="average", pct=True)
        gap_score = audience_percentile + (1.0 - coverage_percentile)
        results[f"{prefix}_coverage_gap_score"] = gap_score
        results[f"{prefix}_coverage_gap_rank"] = (
            gap_score.rank(method="min", ascending=False).astype(int)
        )
    return results


def _axis_limits(values: np.ndarray, *, include_zero: bool = True) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if include_zero:
        values = np.concatenate([values, [0.0]])
    lower = float(values.min())
    upper = float(values.max())
    span = max(upper - lower, 0.05)
    return lower - 0.05 * span, upper + 0.05 * span


def _label_rows(
    results: pd.DataFrame,
    *,
    score_column: str,
    label_mode: str,
    label_count: int,
) -> pd.DataFrame:
    if label_mode == "none" or label_count == 0:
        return results.iloc[0:0]
    if label_mode == "audience":
        return results.nlargest(label_count, "audience_ols_adjusted_r2")
    if label_mode == "coverage-gap":
        prefix = score_column.removesuffix("_r2")
        return results.nsmallest(label_count, f"{prefix}_coverage_gap_rank")
    raise ValueError(f"invalid label mode: {label_mode}")


def _label_selection_note(label_mode: str, label_count: int) -> str:
    """Return the concise footer explanation for the active label mode."""
    if label_mode == "none" or label_count == 0:
        return "No feature labels."
    if label_mode == "audience":
        noun = "association" if label_count == 1 else "associations"
        return f"Labels: {label_count} largest exposure {noun}."
    if label_mode == "coverage-gap":
        noun = "feature" if label_count == 1 else "features"
        return (
            f"Labels: {label_count} high-association/low-coverage {noun} "
            "per panel."
        )
    raise ValueError(f"invalid label mode: {label_mode}")


def _place_nonoverlapping_labels(
    axis: plt.Axes,
    labels: pd.DataFrame,
    *,
    score_column: str,
    minimum_vertical_points: float = 11.0,
) -> list[plt.Annotation]:
    """Annotate points with deterministic, vertically separated labels."""
    if labels.empty:
        return []
    if minimum_vertical_points <= 0:
        raise ValueError("minimum_vertical_points must be positive")

    x_values = labels[score_column].to_numpy(dtype=float)
    y_values = labels["audience_ols_adjusted_r2"].to_numpy(dtype=float)
    display_points = axis.transData.transform(np.column_stack([x_values, y_values]))
    desired_y = display_points[:, 1]
    order = np.argsort(-desired_y, kind="mergesort")

    pixels_per_point = axis.figure.dpi / 72.0
    minimum_separation = minimum_vertical_points * pixels_per_point
    vertical_margin = 5.0 * pixels_per_point
    top = float(axis.bbox.y1 - vertical_margin)
    bottom = float(axis.bbox.y0 + vertical_margin)
    placed_y = np.empty(len(labels), dtype=float)
    previous = top + minimum_separation
    for index in order:
        placed_y[index] = min(float(desired_y[index]), previous - minimum_separation)
        previous = placed_y[index]

    # Five labels comfortably fit in every supported figure. If the greedy
    # pass crosses the lower margin, shift the complete block upward while
    # retaining its ordering and exact separation.
    lowest = float(placed_y[order[-1]])
    if lowest < bottom:
        placed_y += bottom - lowest
    highest = float(placed_y[order[0]])
    if highest > top:
        placed_y -= highest - top

    midpoint = sum(axis.get_xlim()) / 2.0
    horizontal_offset = 5.0 * pixels_per_point
    annotations: list[plt.Annotation] = []
    for row_index, row in enumerate(labels.itertuples(index=False)):
        point_x = float(getattr(row, score_column))
        point_y = float(row.audience_ols_adjusted_r2)
        point_display_x = float(display_points[row_index, 0])
        place_right = point_x <= midpoint
        text_display_x = point_display_x + (
            horizontal_offset if place_right else -horizontal_offset
        )
        text_x, text_y = axis.transData.inverted().transform(
            (text_display_x, placed_y[row_index])
        )
        annotations.append(
            axis.annotate(
                str(row.display_name),
                (point_x, point_y),
                xytext=(text_x, text_y),
                textcoords="data",
                ha="left" if place_right else "right",
                va="center",
                fontsize=8,
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#666666",
                    "lw": 0.5,
                    "shrinkA": 2,
                    "shrinkB": 2,
                },
                zorder=4,
            )
        )
    return annotations


def _place_numbered_callouts(
    axis: plt.Axes,
    labels: pd.DataFrame,
    *,
    score_column: str,
    minimum_vertical_points: float = 11.0,
) -> list[plt.Annotation]:
    """Place plain, collision-aware numbered callouts for an article panel."""
    if labels.empty:
        return []

    x_values = labels[score_column].to_numpy(dtype=float)
    y_values = labels["audience_ols_adjusted_r2"].to_numpy(dtype=float)
    display_points = axis.transData.transform(np.column_stack([x_values, y_values]))
    desired_y = display_points[:, 1]
    order = np.argsort(-desired_y, kind="mergesort")
    pixels_per_point = axis.figure.dpi / 72.0
    minimum_separation = minimum_vertical_points * pixels_per_point
    vertical_margin = 4.0 * pixels_per_point
    top = float(axis.bbox.y1 - vertical_margin)
    bottom = float(axis.bbox.y0 + vertical_margin)
    placed_y = np.empty(len(labels), dtype=float)
    previous = top + minimum_separation
    for index in order:
        placed_y[index] = min(float(desired_y[index]), previous - minimum_separation)
        previous = placed_y[index]
    if float(placed_y[order[-1]]) < bottom:
        placed_y += bottom - float(placed_y[order[-1]])
    if float(placed_y[order[0]]) > top:
        placed_y -= float(placed_y[order[0]]) - top

    midpoint = sum(axis.get_xlim()) / 2.0
    horizontal_offset = 7.0 * pixels_per_point
    annotations: list[plt.Annotation] = []
    for number, row in enumerate(labels.itertuples(index=False), start=1):
        row_index = number - 1
        point_x = float(getattr(row, score_column))
        point_y = float(row.audience_ols_adjusted_r2)
        point_display_x = float(display_points[row_index, 0])
        place_right = point_x <= midpoint
        text_display_x = point_display_x + (
            horizontal_offset if place_right else -horizontal_offset
        )
        text_x, text_y = axis.transData.inverted().transform(
            (text_display_x, placed_y[row_index])
        )
        annotation = axis.annotate(
            str(number),
            (point_x, point_y),
            xytext=(text_x, text_y),
            textcoords="data",
            ha="center",
            va="center",
            fontsize=6.5,
            fontweight="bold",
            color="#111111",
            arrowprops={
                "arrowstyle": "-",
                "color": "#666666",
                "lw": 0.45,
                "shrinkA": 1,
                "shrinkB": 2,
            },
            zorder=5,
        )
        annotation.set_path_effects(
            [patheffects.withStroke(linewidth=1.8, foreground="white")]
        )
        annotation.set_gid(f"article-callout-{number}")
        annotations.append(annotation)
    return annotations


def _populate_article_label_key(
    axis: plt.Axes,
    labels: pd.DataFrame,
    *,
    colors_by_type: Mapping[str, str],
    variability_colors_by_type: Mapping[str, str],
) -> None:
    """List article callout numbers and wrapped readable feature names."""
    axis.set_gid("audience-label-key")
    axis.set_axis_off()
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.text(
        0.02,
        0.98,
        "Feature key",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
    ).set_gid("article-key-heading")

    wrapped_names = [
        textwrap.wrap(
            str(row.display_name),
            width=25,
            break_long_words=False,
            break_on_hyphens=False,
        )
        or [str(row.display_name)]
        for row in labels.itertuples(index=False)
    ]
    block_units = [len(lines) + 0.55 for lines in wrapped_names]
    unit_height = 0.87 / max(sum(block_units), 1.0)
    cursor = 0.89
    for number, (row, lines, units) in enumerate(
        zip(labels.itertuples(index=False), wrapped_names, block_units), start=1
    ):
        modality = str(row.feature_type)
        facecolor = feature_color(
            modality,
            bool(row.is_variability),
            colors_by_type=colors_by_type,
            variability_colors_by_type=variability_colors_by_type,
        )
        number_text = axis.text(
            0.08,
            cursor - 0.01,
            str(number),
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=6.5,
            fontweight="bold",
            color="#111111",
            bbox={
                "boxstyle": "circle,pad=0.16",
                "facecolor": facecolor,
                "edgecolor": darken_color(colors_by_type[modality]),
                "linewidth": 0.7,
            },
        )
        number_text.set_gid(f"article-key-number-{number}")
        name_text = axis.text(
            0.19,
            cursor,
            "\n".join(lines),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
            linespacing=1.1,
            color="#202020",
        )
        name_text.set_gid(f"article-key-label-{number}")
        cursor -= units * unit_height


def plot_audience_relevance(
    results: pd.DataFrame,
    *,
    destination: Path,
    label_mode: str,
    label_count: int,
    figure_layout: str = "wide",
    quadrant_guides: str = "fixed",
    colors_by_type: Mapping[str, str] = feature_type_colors,
    variability_colors_by_type: Mapping[str, str] = variability_feature_type_colors,
) -> None:
    """Render a wide or article-layout audience-relevance figure."""
    if label_count < 0:
        raise ValueError("label_count cannot be negative")
    if figure_layout not in FIGURE_LAYOUTS:
        raise ValueError(
            f"invalid figure layout {figure_layout!r}; "
            f"choose from {', '.join(FIGURE_LAYOUTS)}"
        )
    if quadrant_guides not in QUADRANT_GUIDE_MODES:
        raise ValueError(
            f"invalid quadrant guide mode {quadrant_guides!r}; "
            f"choose from {', '.join(QUADRANT_GUIDE_MODES)}"
        )
    full_n = int(
        results.loc[results["source_dataset"].eq("full"), "demographic_n_donors"].iloc[0]
    )
    linguistic_n = int(
        results.loc[
            results["source_dataset"].eq("linguistic"), "demographic_n_donors"
        ].iloc[0]
    )
    x_values = results[[column for column, _ in REPRESENTATIONS]].to_numpy().ravel()
    y_values = results["audience_ols_adjusted_r2"].to_numpy()
    x_limits = _axis_limits(x_values)
    y_limits = _axis_limits(y_values)
    labels_active = label_mode != "none" and label_count > 0

    if figure_layout == "wide":
        figure, axes = plt.subplots(1, 3, figsize=(21, 7.5), sharex=True, sharey=True)
        key_axes: list[plt.Axes] = []
    elif labels_active:
        figure = plt.figure(figsize=(7.2, 9.0))
        grid = figure.add_gridspec(
            3,
            2,
            width_ratios=(4.8, 1.7),
            hspace=0.25,
            wspace=0.04,
        )
        axes_list: list[plt.Axes] = []
        key_axes = []
        for row_index in range(3):
            shared_axis = axes_list[0] if axes_list else None
            axes_list.append(
                figure.add_subplot(
                    grid[row_index, 0],
                    sharex=shared_axis,
                    sharey=shared_axis,
                )
            )
            key_axes.append(figure.add_subplot(grid[row_index, 1]))
        axes = np.asarray(axes_list, dtype=object)
    else:
        figure, axes = plt.subplots(3, 1, figsize=(7.2, 9.0), sharex=True, sharey=True)
        key_axes = []
    axes = np.atleast_1d(axes)
    for axis in axes:
        axis.set_gid("audience-scatter-panel")
    panel_labels: list[tuple[plt.Axes, str, pd.DataFrame]] = []
    for panel_index, (axis, (score_column, representation_name)) in enumerate(
        zip(axes, REPRESENTATIONS)
    ):
        for row in results.itertuples(index=False):
            facecolor = feature_color(
                str(row.feature_type),
                bool(row.is_variability),
                colors_by_type=colors_by_type,
                variability_colors_by_type=variability_colors_by_type,
            )
            edgecolor = darken_color(colors_by_type[str(row.feature_type)])
            axis.scatter(
                getattr(row, score_column),
                row.audience_ols_adjusted_r2,
                s=42,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=0.65,
                alpha=0.82,
                zorder=2,
            )

        labels = _label_rows(
            results,
            score_column=score_column,
            label_mode=label_mode,
            label_count=label_count,
        )
        panel_labels.append((axis, score_column, labels))
        if key_axes:
            _populate_article_label_key(
                key_axes[panel_index],
                labels,
                colors_by_type=colors_by_type,
                variability_colors_by_type=variability_colors_by_type,
            )

        if quadrant_guides == "fixed":
            vertical_guide = axis.axvline(
                QUADRANT_GUIDE_X,
                color="#8f8f8f",
                linewidth=0.75,
                linestyle=(0, (4, 3)),
                alpha=0.72,
                zorder=0.8,
            )
            vertical_guide.set_gid("audience-quadrant-guide-x")
            horizontal_guide = axis.axhline(
                QUADRANT_GUIDE_Y,
                color="#8f8f8f",
                linewidth=0.75,
                linestyle=(0, (4, 3)),
                alpha=0.72,
                zorder=0.8,
            )
            horizontal_guide.set_gid("audience-quadrant-guide-y")
        axis.axhline(0.0, color="#555555", linewidth=0.8, zorder=1)
        axis.axvline(0.0, color="#555555", linewidth=0.8, zorder=1)
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_title(representation_name)
        if figure_layout == "wide":
            axis.set_xlabel("Cross-validated $R^2$: embedding predicts MAVIC feature")
        else:
            axis.tick_params(axis="both", labelsize=8.5)
            if panel_index < len(REPRESENTATIONS) - 1:
                axis.tick_params(labelbottom=False)
        axis.grid(color="#d8d8d8", linewidth=0.7, alpha=0.65)
        axis.set_axisbelow(True)

    type_handles = [
        Patch(
            facecolor=colors_by_type[feature_type],
            edgecolor=darken_color(colors_by_type[feature_type]),
            label=TYPE_LABELS[feature_type],
        )
        for feature_type in FEATURE_TYPES
    ]
    style_handles = [
        Patch(facecolor="#777777", edgecolor="#555555", label="Ordinary feature"),
        Patch(facecolor="#eeeeee", edgecolor="#777777", label="SD / variability"),
    ]
    title = "Social exposure patterning and representation coverage of video features"
    footer = (
        f"{_label_selection_note(label_mode, label_count)} "
        f"Audio/visual donor cohort: n={full_n:,}; "
        f"linguistic donor cohort: n={linguistic_n:,}."
    )
    if figure_layout == "wide":
        axes[0].set_ylabel(
            "OLS adjusted $R^2$:\n"
            "demographic patterning of donor mean feature exposure"
        )
        figure.suptitle(title, fontsize=15, y=0.985)
        figure.legend(
            handles=[*type_handles, *style_handles],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.935),
            ncol=5,
            frameon=False,
        )
        figure.text(0.5, 0.018, footer, ha="center", fontsize=9)
        figure.tight_layout(rect=(0.02, 0.06, 1.0, 0.90))
    else:
        figure.suptitle(title, fontsize=12, y=0.988)
        figure.legend(
            handles=[*type_handles, *style_handles],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=5,
            frameon=False,
            fontsize=8,
            handlelength=1.5,
            columnspacing=1.0,
        )
        figure.supxlabel(
            "Cross-validated $R^2$: embedding predicts MAVIC feature",
            fontsize=9.5,
            y=0.035,
        )
        figure.supylabel(
            "OLS adjusted $R^2$: demographic patterning of donor mean feature exposure",
            fontsize=9.5,
            x=0.015,
        )
        figure.text(
            0.5,
            0.008,
            footer,
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
        figure.subplots_adjust(
            left=0.10,
            right=0.985,
            bottom=0.08,
            top=0.90,
            hspace=0.32,
            wspace=0.06,
        )
    # Text placement needs final axis dimensions. Draw once after tight_layout,
    # then position labels in display space with a guaranteed vertical gap.
    figure.canvas.draw()
    for axis, score_column, labels in panel_labels:
        if figure_layout == "article" and labels_active:
            _place_numbered_callouts(
                axis,
                labels,
                score_column=score_column,
            )
        else:
            _place_nonoverlapping_labels(
                axis,
                labels,
                score_column=score_column,
            )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if figure_layout == "article":
        figure.savefig(destination, dpi=300)
        figure.savefig(destination.with_suffix(".pdf"))
    else:
        figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)


def run_analysis(
    meta_parquet: Path = DEFAULT_META_PARQUET,
    full_mavic_csv: Path = DEFAULT_FULL_MAVIC_CSV,
    linguistic_mavic_csv: Path = DEFAULT_LINGUISTIC_MAVIC_CSV,
    coverage_csv: Path = DEFAULT_COVERAGE_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    min_videos: int = DEFAULT_MIN_VIDEOS,
    label_mode: str = "audience",
    label_count: int = DEFAULT_LABEL_COUNT,
    figure_layout: str = "wide",
    quadrant_guides: str = "fixed",
) -> pd.DataFrame:
    """Run donor aggregation, demographic scoring, and figure generation."""
    if figure_layout not in FIGURE_LAYOUTS:
        raise ValueError(
            f"invalid figure layout {figure_layout!r}; "
            f"choose from {', '.join(FIGURE_LAYOUTS)}"
        )
    if quadrant_guides not in QUADRANT_GUIDE_MODES:
        raise ValueError(
            f"invalid quadrant guide mode {quadrant_guides!r}; "
            f"choose from {', '.join(QUADRANT_GUIDE_MODES)}"
        )
    events = load_metadata(meta_parquet)
    full_features = load_feature_table(full_mavic_csv)
    linguistic_features = load_feature_table(linguistic_mavic_csv)
    coverage = load_coverage_scores(coverage_csv)

    metadata_video_ids = set(events[VIDEO_ID_COLUMN].tolist())
    missing_full_events = sorted(
        set(full_features[VIDEO_ID_COLUMN].tolist()) - metadata_video_ids
    )
    if missing_full_events:
        raise ValueError(
            "metadata parquet does not cover the full MAVIC cohort: "
            f"{len(missing_full_events):,} full-dataset videos have no donor event. "
            "Export meta_df.parquet from the transcript_clustering=False run; "
            "sample missing IDs: "
            + ", ".join(missing_full_events[:5])
        )

    full_names = coverage.loc[
        coverage["source_dataset"].eq("full"), "feature"
    ].tolist()
    linguistic_names = coverage.loc[
        coverage["source_dataset"].eq("linguistic"), "feature"
    ].tolist()
    if not full_names or not linguistic_names:
        raise ValueError("coverage table must contain full and linguistic features")
    if set(full_names) != set(full_features.columns) - {VIDEO_ID_COLUMN}:
        raise ValueError("full MAVIC features do not match the coverage score table")
    if not set(linguistic_names).issubset(linguistic_features.columns):
        raise ValueError("linguistic MAVIC features do not match the coverage score table")

    cohort_specs = (
        ("full", full_features, full_names),
        ("linguistic", linguistic_features, linguistic_names),
    )
    result_parts = []
    audits: dict[str, object] = {
        "settings": {
            "min_videos": min_videos,
            "model": "OLS",
            "score": "adjusted_r2",
            "demographics": list(DEMOGRAPHIC_COLUMNS),
            "age_coding": "raw continuous age",
            "categorical_coding": (
                "treatment coding with lexicographically first reference"
            ),
        },
        "cohorts": {},
    }

    for cohort_name, feature_frame, feature_names in cohort_specs:
        profiles, cohort_audit = build_donor_profiles(
            events,
            feature_frame,
            feature_names,
            cohort_name=cohort_name,
            min_videos=min_videos,
        )
        audits["cohorts"][cohort_name] = cohort_audit
        demographics = profiles[list(DEMOGRAPHIC_COLUMNS)].reset_index(drop=True)
        targets = profiles[feature_names].to_numpy(dtype=np.float64)

        print(
            f"Scoring {cohort_name} donor cohort: {len(profiles):,} donors, "
            f"{len(feature_names)} features...",
            flush=True,
        )
        adjusted_r2, model_audit = demographic_ols_adjusted_r2(
            demographics, targets
        )
        cohort_audit["demographic_model"] = model_audit
        result_parts.append(
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "demographic_n_donors": len(profiles),
                    "demographic_n_events": int(cohort_audit["eligible_events"]),
                    "mean_usable_videos_per_donor": float(
                        cohort_audit["usable_videos_per_donor"]["mean"]
                    ),
                    "audience_ols_adjusted_r2": adjusted_r2,
                }
            )
        )

    demographic_results = pd.concat(result_parts, ignore_index=True)
    results = coverage.merge(
        demographic_results, on="feature", how="left", validate="one_to_one"
    )
    if results["audience_ols_adjusted_r2"].isna().any():
        raise ValueError("some coverage features are missing demographic results")
    results = add_coverage_gap_ranks(results)

    ordered_columns = [
        "feature",
        "display_name",
        "feature_type",
        "is_variability",
        "source_dataset",
        "demographic_n_donors",
        "demographic_n_events",
        "mean_usable_videos_per_donor",
        "audience_ols_adjusted_r2",
        "openclip_r2",
        "lco_r2",
        "f2llm_r2",
        "openclip_coverage_gap_score",
        "openclip_coverage_gap_rank",
        "lco_coverage_gap_score",
        "lco_coverage_gap_rank",
        "f2llm_coverage_gap_score",
        "f2llm_coverage_gap_rank",
    ]
    results = results.loc[:, ordered_columns]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "audience_relevance_scores.csv", index=False)
    (output_dir / "donor_profile_audit.json").write_text(
        json.dumps(audits, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if figure_layout == "article":
        destination = (
            output_dir / f"audience_relevance_scatter-{label_mode}-article.png"
        )
    else:
        destination = output_dir / f"audience_relevance_scatter-{label_mode}.png"
    plot_audience_relevance(
        results,
        destination=destination,
        label_mode=label_mode,
        label_count=label_count,
        figure_layout=figure_layout,
        quadrant_guides=quadrant_guides,
        variability_colors_by_type=variability_feature_type_colors,
    )
    print(f"Saved audience-relevance results to {output_dir}", flush=True)
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Relate donor-level demographic exposure associations to MAVIC "
            "embedding coverage"
        )
    )
    parser.add_argument("--meta-parquet", type=Path, default=DEFAULT_META_PARQUET)
    parser.add_argument("--full-mavic-csv", type=Path, default=DEFAULT_FULL_MAVIC_CSV)
    parser.add_argument(
        "--linguistic-mavic-csv",
        type=Path,
        default=DEFAULT_LINGUISTIC_MAVIC_CSV,
    )
    parser.add_argument("--coverage-csv", type=Path, default=DEFAULT_COVERAGE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-videos", type=int, default=DEFAULT_MIN_VIDEOS)
    parser.add_argument(
        "--label-mode",
        choices=("audience", "coverage-gap", "none"),
        default="audience",
    )
    parser.add_argument("--label-count", type=int, default=DEFAULT_LABEL_COUNT)
    parser.add_argument(
        "--figure-layout",
        choices=FIGURE_LAYOUTS,
        default="wide",
    )
    parser.add_argument(
        "--quadrant-guides",
        choices=QUADRANT_GUIDE_MODES,
        default="fixed",
        help=(
            "draw shared orientation guides at x=.40 and y=.075 or omit them "
            "(default: fixed)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        run_analysis(
            meta_parquet=args.meta_parquet,
            full_mavic_csv=args.full_mavic_csv,
            linguistic_mavic_csv=args.linguistic_mavic_csv,
            coverage_csv=args.coverage_csv,
            output_dir=args.output_dir,
            min_videos=args.min_videos,
            label_mode=args.label_mode,
            label_count=args.label_count,
            figure_layout=args.figure_layout,
            quadrant_guides=args.quadrant_guides,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
