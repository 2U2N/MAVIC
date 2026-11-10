"""Generic plotting for long-form MAVIC representation-coverage scores."""

from __future__ import annotations

from importlib.resources import files
import json
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from .probing import SCORE_COLUMNS
from .io import validate_representation_name


def load_palette() -> dict[str, dict[str, str]]:
    """Load the release's color-blind-aware modality palette."""
    resource = files("mavic").joinpath("data", "palette.json")
    try:
        palette = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load MAVIC palette: {error}") from error
    for shade in ("ordinary", "variability"):
        if not isinstance(palette.get(shade), dict):
            raise ValueError(f"palette is missing {shade!r} colors")
        if set(palette[shade]) != {"audio", "visual", "linguistic"}:
            raise ValueError(f"palette {shade!r} colors must cover all feature types")
    return palette


def validate_long_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the generic long-form probing result."""
    missing = [column for column in SCORE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            "coverage score table is missing columns: " + ", ".join(missing)
        )
    result = frame.loc[:, SCORE_COLUMNS].copy()
    if result.empty:
        raise ValueError("coverage score table is empty")
    for column in (
        "feature",
        "display_name",
        "feature_type",
        "source_cohort",
        "representation",
    ):
        if result[column].isna().any():
            raise ValueError(f"coverage score column {column!r} contains missing values")
        result[column] = result[column].astype(str)
        if result[column].str.strip().eq("").any():
            raise ValueError(f"coverage score column {column!r} contains empty values")
    if result.duplicated(["representation", "feature"]).any():
        raise ValueError("each representation-feature pair must be unique")
    for representation in result["representation"].unique():
        validate_representation_name(representation)
    if not set(result["feature_type"]).issubset({"audio", "visual", "linguistic"}):
        raise ValueError("coverage scores contain an invalid feature type")
    if not set(result["source_cohort"]).issubset({"full", "linguistic"}):
        raise ValueError("coverage scores contain an invalid source cohort")
    boolean = result["is_variability"]
    if not pd.api.types.is_bool_dtype(boolean):
        normalized = boolean.astype(str).str.lower()
        if not set(normalized).issubset({"true", "false"}):
            raise ValueError("is_variability must contain booleans")
        result["is_variability"] = normalized.map({"true": True, "false": False})
    try:
        result["r2"] = pd.to_numeric(result["r2"], errors="raise")
        result["n_videos"] = pd.to_numeric(
            result["n_videos"], errors="raise", downcast="integer"
        )
    except (TypeError, ValueError) as error:
        raise ValueError("r2 and n_videos must be numeric") from error
    if not np.isfinite(result["r2"].to_numpy(float)).all():
        raise ValueError("coverage scores must be finite")
    if (result["n_videos"] <= 0).any():
        raise ValueError("n_videos must be positive")
    if np.any(result["n_videos"].to_numpy(float) % 1):
        raise ValueError("n_videos must contain integers")

    metadata_columns = [
        "display_name",
        "feature_type",
        "is_variability",
        "source_cohort",
        "n_videos",
    ]
    inconsistent = [
        feature
        for feature, group in result.groupby("feature", sort=False)
        if any(group[column].nunique(dropna=False) != 1 for column in metadata_columns)
    ]
    if inconsistent:
        raise ValueError(
            "feature metadata differs across representations: "
            + ", ".join(sorted(inconsistent))
        )
    expected = set(result["feature"])
    for representation, group in result.groupby("representation", sort=False):
        if set(group["feature"]) != expected:
            raise ValueError(
                f"representation {representation!r} does not contain every feature"
            )
    return result


def shared_score_limits(scores: pd.Series) -> tuple[float, float]:
    values = np.concatenate([scores.to_numpy(dtype=float), np.asarray([0.0])])
    lower, upper = float(values.min()), float(values.max())
    span = max(upper - lower, 0.1)
    return lower - 0.03 * span, upper + 0.03 * span


def plot_coverage_scores(
    scores: pd.DataFrame,
    output_dir: Path | str,
) -> list[Path]:
    """Create one readable PNG and PDF chart per representation."""
    scores = validate_long_scores(scores)
    palette = load_palette()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    x_limits = shared_score_limits(scores["r2"])
    outputs: list[Path] = []
    feature_types = ("audio", "visual", "linguistic")
    for representation, subset in scores.groupby("representation", sort=False):
        ordered = subset.sort_values(
            ["r2", "feature"], ascending=[False, True], kind="mergesort"
        ).reset_index(drop=True)
        colors = [
            palette["variability" if variability else "ordinary"][feature_type]
            for feature_type, variability in zip(
                ordered["feature_type"], ordered["is_variability"], strict=True
            )
        ]
        labels = [
            textwrap.fill(label, 47, break_long_words=False)
            for label in ordered["display_name"]
        ]
        height = max(5.0, 0.28 * len(ordered))
        figure, axis = plt.subplots(figsize=(11.5, height))
        y = np.arange(len(ordered))
        axis.barh(y, ordered["r2"], color=colors, height=0.76)
        axis.set_yticks(y, labels=labels, fontsize=7.5)
        axis.invert_yaxis()
        axis.set_xlim(*x_limits)
        axis.axvline(0.0, color="#303030", linewidth=0.9)
        axis.grid(axis="x", color="#d5d5d5", linewidth=0.65, alpha=0.7)
        axis.set_axisbelow(True)
        axis.set_xlabel("Five-fold out-of-fold $R^2$")
        cohort_sizes = (
            ordered.groupby("source_cohort", sort=False)["n_videos"].first().to_dict()
        )
        subtitle = " • ".join(
            f"{cohort}: n={int(count):,}" for cohort, count in cohort_sizes.items()
        )
        axis.set_title(f"{representation} feature coverage\n{subtitle}", pad=12)
        modality_handles = [
            Patch(
                facecolor=palette["ordinary"][feature_type],
                label=feature_type.capitalize(),
            )
            for feature_type in feature_types
        ]
        shade_handles = [
            Patch(facecolor="#555555", label="Ordinary"),
            Patch(facecolor="#bbbbbb", label="SD / variability"),
        ]
        first = axis.legend(
            handles=modality_handles,
            title="Modality",
            loc="lower left",
            bbox_to_anchor=(1.01, 0.06),
        )
        axis.add_artist(first)
        axis.legend(
            handles=shade_handles,
            title="Feature form",
            loc="lower left",
            bbox_to_anchor=(1.01, 0.0),
        )
        figure.tight_layout()
        for suffix in (".png", ".pdf"):
            destination = output / f"coverage-{representation}{suffix}"
            figure.savefig(
                destination,
                dpi=300 if suffix == ".png" else None,
                bbox_inches="tight",
            )
            outputs.append(destination)
        plt.close(figure)
    return outputs


def run_plot_coverage(
    score_path: Path | str,
    output_dir: Path | str,
) -> list[Path]:
    source = Path(score_path)
    if not source.is_file():
        raise ValueError(f"coverage score CSV not found: {source}")
    try:
        scores = pd.read_csv(source)
    except (OSError, pd.errors.ParserError) as error:
        raise ValueError(f"could not read coverage score CSV {source}: {error}") from error
    return plot_coverage_scores(scores, output_dir)
