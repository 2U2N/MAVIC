"""Shared vocabulary and palette helpers.

The paper layer deliberately reads the public CSV/JSON resources rather than
importing private project dictionaries.  This keeps the analysis scripts
auditable and runnable both as modules and as files from a source checkout.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Mapping

from matplotlib.colors import to_rgb
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
VOCABULARY_PATH = REPOSITORY_ROOT / "src" / "mavic" / "data" / "mavic_v1.csv"
PALETTE_PATH = REPOSITORY_ROOT / "src" / "mavic" / "data" / "palette.json"
PAPER_RESULTS_DIR = REPOSITORY_ROOT / "paper" / "results"
PAPER_FIGURES_DIR = REPOSITORY_ROOT / "paper" / "figures"
GENERATED_DIR = REPOSITORY_ROOT / "paper" / "generated"
RESTRICTED_INPUTS_DIR = REPOSITORY_ROOT / "restricted_inputs"

FEATURE_TYPES = ("audio", "visual", "linguistic")
TYPE_LABELS = {
    "audio": "Audio",
    "visual": "Visual",
    "linguistic": "Linguistic",
}


@lru_cache(maxsize=1)
def load_vocabulary(path: Path = VOCABULARY_PATH) -> pd.DataFrame:
    """Load and validate the public 133-probe vocabulary."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"MAVIC vocabulary not found: {path}")
    frame = pd.read_csv(path)
    required = {
        "feature",
        "display_name",
        "feature_type",
        "is_variability",
        "source_cohort",
        "vocabulary_version",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("MAVIC vocabulary is missing columns: " + ", ".join(missing))
    if len(frame) != 133:
        raise ValueError(f"MAVIC v1 must contain 133 probes, found {len(frame)}")
    if frame["feature"].duplicated().any():
        raise ValueError("MAVIC vocabulary contains duplicate feature names")
    if frame["display_name"].isna().any() or frame["display_name"].astype(str).str.strip().eq("").any():
        raise ValueError("MAVIC display names must be non-empty")
    invalid_types = sorted(set(frame["feature_type"]) - set(FEATURE_TYPES))
    if invalid_types:
        raise ValueError("invalid MAVIC feature types: " + ", ".join(invalid_types))
    expected_counts = {"visual": 86, "audio": 13, "linguistic": 34}
    if frame["feature_type"].value_counts().to_dict() != expected_counts:
        raise ValueError(
            "MAVIC v1 modality counts must be 86 visual, 13 audio, and "
            "34 linguistic"
        )
    expected_cohort = frame["feature_type"].map(
        {"audio": "full", "visual": "full", "linguistic": "linguistic"}
    )
    if not frame["source_cohort"].eq(expected_cohort).all():
        raise ValueError("MAVIC source cohorts disagree with feature types")
    return frame.copy()


@lru_cache(maxsize=1)
def load_palette(path: Path = PALETTE_PATH) -> tuple[dict[str, str], dict[str, str]]:
    """Load ordinary and variability colors from the public palette."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"MAVIC palette not found: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("MAVIC palette must be a JSON object")
    ordinary = payload.get("ordinary", payload.get("feature_type_colors"))
    variability = payload.get(
        "variability", payload.get("variability_feature_type_colors")
    )
    for label, colors in (("ordinary", ordinary), ("variability", variability)):
        if not isinstance(colors, dict) or set(colors) != set(FEATURE_TYPES):
            raise ValueError(
                f"palette {label} colors must define audio, visual, and linguistic"
            )
        for feature_type, color in colors.items():
            try:
                to_rgb(color)
            except ValueError as error:
                raise ValueError(
                    f"invalid {label} color for {feature_type}: {color!r}"
                ) from error
    return dict(ordinary), dict(variability)


def display_name_mapping() -> dict[str, str]:
    vocabulary = load_vocabulary()
    return dict(zip(vocabulary["feature"], vocabulary["display_name"]))


def feature_type_mapping() -> dict[str, str]:
    vocabulary = load_vocabulary()
    return dict(zip(vocabulary["feature"], vocabulary["feature_type"]))


def darken_color(
    color: str, retained_fraction: float = 0.70
) -> tuple[float, float, float]:
    """Blend a color toward black by retaining a fraction of each channel."""
    if not 0.0 <= retained_fraction <= 1.0:
        raise ValueError("retained_fraction must be between zero and one")
    red, green, blue = to_rgb(color)
    return (
        red * retained_fraction,
        green * retained_fraction,
        blue * retained_fraction,
    )


def feature_color(
    feature_type: str,
    is_variability: bool,
    *,
    colors_by_type: Mapping[str, str] | None = None,
    variability_colors_by_type: Mapping[str, str] | None = None,
) -> str:
    """Return the configured modality/form color for one probe."""
    if colors_by_type is None or variability_colors_by_type is None:
        default_colors, default_variability = load_palette()
        colors_by_type = colors_by_type or default_colors
        variability_colors_by_type = (
            variability_colors_by_type or default_variability
        )
    palette = variability_colors_by_type if is_variability else colors_by_type
    try:
        return palette[feature_type]
    except KeyError as error:
        raise ValueError(f"unknown feature type: {feature_type}") from error


def is_variability_feature(feature: str) -> bool:
    """Recognize the variability naming conventions used by MAVIC."""
    return (
        feature.startswith(("std_", "sd_"))
        or feature.endswith(("_std", "_sd"))
    )


def shared_x_limits(score_arrays: list[object] | tuple[object, ...]) -> tuple[float, float]:
    """Return padded limits spanning every supplied score array and zero."""
    arrays = [np.asarray(scores, dtype=float).ravel() for scores in score_arrays]
    combined = np.concatenate([*arrays, np.asarray([0.0])])
    if not np.isfinite(combined).all():
        raise ValueError("coverage scores must be finite")
    lower = float(combined.min())
    upper = float(combined.max())
    span = max(upper - lower, 0.1)
    return lower - 0.03 * span, upper + 0.03 * span
