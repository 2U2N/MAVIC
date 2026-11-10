"""Deterministic feature selection for full and linguistic MAVIC cohorts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


VIDEO_ID_COLUMN = "video_id"
DEFAULT_CONFIG: dict[str, Any] = {
    "duration_column": "duration_seconds",
    "minimum_duration_seconds": 30.0,
    "minimum_distinct_values": 10,
    "maximum_absolute_correlation": 0.8,
    "exclude_features": [],
    "exclude_name_contains": [],
}


def is_variability_feature(feature: str) -> bool:
    """Return whether a feature follows a MAVIC SD naming convention."""
    return feature.endswith(("_sd", "_std")) or feature.startswith(("std_", "sd_"))


def sd_counterpart(feature: str) -> str | None:
    """Return the SD counterpart of a recognized volatility feature."""
    if feature.startswith("volatility_"):
        return "std_" + feature[len("volatility_") :]
    if feature.startswith("vol_"):
        return "sd_" + feature[len("vol_") :]
    if feature.endswith("_volatility"):
        return feature[: -len("_volatility")] + "_std"
    if feature.endswith("_vol"):
        return feature[: -len("_vol")] + "_sd"
    return None


def correlation_order(columns: Sequence[str]) -> list[str]:
    """Prioritize duration, then variability, then ordinary features."""
    names = list(columns)
    if len(names) != len(set(names)):
        raise ValueError("feature columns must be unique")
    duration = [name for name in names if name == "duration_seconds"]
    variability = [
        name
        for name in names
        if name != "duration_seconds" and is_variability_feature(name)
    ]
    ordinary = [
        name
        for name in names
        if name != "duration_seconds" and not is_variability_feature(name)
    ]
    return duration + variability + ordinary


def load_selection_config(path: Path | str) -> dict[str, Any]:
    """Load and normalize the YAML selection configuration."""
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"selection configuration not found: {source}")
    try:
        with source.open(encoding="utf-8") as handle:
            supplied = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"could not read selection configuration {source}: {error}") from error
    if not isinstance(supplied, dict):
        raise ValueError(f"{source}: selection configuration must be an object")
    if "correlation_threshold" in supplied:
        if "maximum_absolute_correlation" in supplied:
            raise ValueError(
                f"{source}: specify only one of correlation_threshold and "
                "maximum_absolute_correlation"
            )
        supplied["maximum_absolute_correlation"] = supplied.pop(
            "correlation_threshold"
        )
    unknown = sorted(
        set(supplied) - (set(DEFAULT_CONFIG) | {"linguistic_features", "schema_version"})
    )
    if unknown:
        raise ValueError(f"{source}: unknown settings: {', '.join(unknown)}")
    supplied.pop("schema_version", None)
    config = {**DEFAULT_CONFIG, **supplied}

    duration = config["minimum_duration_seconds"]
    distinct = config["minimum_distinct_values"]
    threshold = config["maximum_absolute_correlation"]
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise ValueError("minimum_duration_seconds must be numeric")
    if isinstance(distinct, bool) or not isinstance(distinct, int) or distinct < 2:
        raise ValueError("minimum_distinct_values must be an integer of at least 2")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0.0 < float(threshold) < 1.0
    ):
        raise ValueError("maximum_absolute_correlation must be between zero and one")
    for field in ("exclude_features", "exclude_name_contains"):
        value = config[field]
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item for item in value
        ):
            raise ValueError(f"{field} must be a list of non-empty strings")
    if "linguistic_features" in config:
        value = config["linguistic_features"]
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item for item in value
        ):
            raise ValueError("linguistic_features must be a list of non-empty strings")
    if not isinstance(config["duration_column"], str) or not config["duration_column"]:
        raise ValueError("duration_column must be a non-empty string")
    return config


def _read_candidates(path: Path | str) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"candidate feature CSV not found: {source}")
    try:
        frame = pd.read_csv(source, dtype={VIDEO_ID_COLUMN: "string"})
    except (OSError, pd.errors.ParserError) as error:
        raise ValueError(f"could not read candidate feature CSV {source}: {error}") from error
    if VIDEO_ID_COLUMN not in frame.columns:
        raise ValueError(f"{source}: missing {VIDEO_ID_COLUMN!r} column")
    if frame.columns.duplicated().any():
        raise ValueError(f"{source}: duplicate column names")
    if frame[VIDEO_ID_COLUMN].isna().any():
        raise ValueError(f"{source}: video IDs may not be missing")
    frame[VIDEO_ID_COLUMN] = frame[VIDEO_ID_COLUMN].astype(str)
    if frame[VIDEO_ID_COLUMN].eq("").any():
        raise ValueError(f"{source}: video IDs must be non-empty")
    if frame[VIDEO_ID_COLUMN].duplicated().any():
        raise ValueError(f"{source}: duplicate video IDs")
    return frame


def _duration_filter(
    frame: pd.DataFrame, *, duration_column: str, minimum_duration: float
) -> pd.DataFrame:
    if duration_column not in frame.columns:
        raise ValueError(f"candidate table is missing duration column {duration_column!r}")
    try:
        duration = pd.to_numeric(frame[duration_column], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{duration_column} must be numeric") from error
    if duration.isna().any() or not np.isfinite(duration.to_numpy(dtype=float)).all():
        raise ValueError(f"{duration_column} must be complete and finite")
    return frame.loc[duration > minimum_duration].reset_index(drop=True)


def quality_filter(
    frame: pd.DataFrame,
    *,
    cohort: str,
    minimum_distinct_values: int,
    excluded_features: Sequence[str] = (),
    excluded_name_fragments: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep numeric, complete, finite, sufficiently variable feature columns."""
    rows: list[dict[str, object]] = []
    kept: dict[str, pd.Series] = {}
    excluded = set(excluded_features)
    for feature in (name for name in frame.columns if name != VIDEO_ID_COLUMN):
        reason: str | None = None
        numeric: pd.Series | None = None
        if feature in excluded or any(
            fragment in feature for fragment in excluded_name_fragments
        ):
            reason = "configured_exclusion"
        else:
            try:
                numeric = pd.to_numeric(frame[feature], errors="raise")
            except (TypeError, ValueError):
                reason = "non_numeric"
        missing = int(frame[feature].isna().sum())
        nonfinite = 0
        unique = int(frame[feature].nunique(dropna=True))
        if numeric is not None:
            values = numeric.to_numpy(dtype=float)
            nonfinite = int((~np.isfinite(values) & ~pd.isna(values)).sum())
            unique = int(numeric.nunique(dropna=True))
            if missing:
                reason = "missing_values"
            elif nonfinite:
                reason = "nonfinite_values"
            elif unique < minimum_distinct_values:
                reason = "low_uniqueness"
        if reason is None:
            assert numeric is not None
            kept[feature] = numeric.astype(float)
        rows.append(
            {
                "cohort": cohort,
                "feature": feature,
                "kept": reason is None,
                "reason": "retained" if reason is None else reason,
                "n_rows": len(frame),
                "n_missing": missing,
                "n_nonfinite": nonfinite,
                "n_distinct": unique,
            }
        )
    if not kept:
        raise ValueError(f"{cohort}: no features survived data-quality filtering")
    values = pd.DataFrame(kept, index=frame.index)
    return values, pd.DataFrame(rows)


def apply_sd_preference(
    frame: pd.DataFrame, *, selection_stage: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove volatility measures whenever the paired SD measure is available."""
    pairs = [
        (feature, counterpart)
        for feature in frame.columns
        if (counterpart := sd_counterpart(feature)) is not None
        and counterpart in frame.columns
    ]
    rows = []
    for volatility, sd_feature in pairs:
        correlation = abs(float(frame[[sd_feature, volatility]].corr().iloc[0, 1]))
        rows.append(
            {
                "selection_stage": selection_stage,
                "n_rows": len(frame),
                "preferred_feature": sd_feature,
                "removed_feature": volatility,
                "absolute_correlation": correlation,
            }
        )
    return (
        frame.drop(columns=[volatility for volatility, _ in pairs]).copy(),
        pd.DataFrame(
            rows,
            columns=[
                "selection_stage",
                "n_rows",
                "preferred_feature",
                "removed_feature",
                "absolute_correlation",
            ],
        ),
    )


def reject_correlated(
    frame: pd.DataFrame,
    *,
    threshold: float,
    selection_stage: str,
    protected_features: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Greedily reject candidates correlated with an earlier retained feature."""
    protected = list(protected_features)
    missing = [feature for feature in protected if feature not in frame.columns]
    if missing:
        raise ValueError(
            "protected canonical features are unavailable after quality filtering: "
            + ", ".join(missing)
        )
    order = correlation_order(frame.columns)
    correlation = frame.loc[:, order].corr().abs()
    kept = list(protected)
    dropped: list[str] = []
    for candidate in order:
        if candidate in protected:
            continue
        if any(float(correlation.loc[retained, candidate]) > threshold for retained in kept):
            dropped.append(candidate)
        else:
            kept.append(candidate)

    rows = []
    for retained in kept:
        for rejected in dropped:
            coefficient = float(correlation.loc[retained, rejected])
            if coefficient > threshold:
                rows.append(
                    {
                        "selection_stage": selection_stage,
                        "n_rows": len(frame),
                        "kept_feature": retained,
                        "dropped_feature": rejected,
                        "absolute_correlation": coefficient,
                    }
                )
    audit = pd.DataFrame(
        rows,
        columns=[
            "selection_stage",
            "n_rows",
            "kept_feature",
            "dropped_feature",
            "absolute_correlation",
        ],
    )
    return frame.loc[:, kept].copy(), audit


def _assert_no_paired_volatility(
    selected: Sequence[str], available_before_preference: Sequence[str]
) -> None:
    available = set(available_before_preference)
    invalid = [
        feature
        for feature in selected
        if (counterpart := sd_counterpart(feature)) is not None
        and counterpart in available
    ]
    if invalid:
        raise RuntimeError(
            "SD preference invariant failed for: " + ", ".join(sorted(invalid))
        )


def _write_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)


def run_selection(
    full_candidates: Path | str,
    linguistic_candidates: Path | str,
    config_path: Path | str,
    output_dir: Path | str,
) -> dict[str, Path]:
    """Run the two-stage MAVIC selection workflow and write auditable outputs."""
    config = load_selection_config(config_path)
    full_raw = _duration_filter(
        _read_candidates(full_candidates),
        duration_column=config["duration_column"],
        minimum_duration=float(config["minimum_duration_seconds"]),
    )
    linguistic_raw = _duration_filter(
        _read_candidates(linguistic_candidates),
        duration_column=config["duration_column"],
        minimum_duration=float(config["minimum_duration_seconds"]),
    )
    if not set(linguistic_raw[VIDEO_ID_COLUMN]).issubset(set(full_raw[VIDEO_ID_COLUMN])):
        raise ValueError(
            "duration-filtered linguistic video IDs must be a subset of full video IDs"
        )

    quality_kwargs = {
        "minimum_distinct_values": config["minimum_distinct_values"],
        "excluded_features": config["exclude_features"],
        "excluded_name_fragments": config["exclude_name_contains"],
    }
    full_quality, full_quality_audit = quality_filter(
        full_raw, cohort="full", **quality_kwargs
    )
    full_available = list(full_quality.columns)
    full_preferred, full_preference_audit = apply_sd_preference(
        full_quality, selection_stage="audio_visual"
    )
    full_selected, full_correlation_audit = reject_correlated(
        full_preferred,
        threshold=float(config["maximum_absolute_correlation"]),
        selection_stage="audio_visual",
    )
    canonical = list(full_selected.columns)
    _assert_no_paired_volatility(canonical, full_available)

    linguistic_quality, linguistic_quality_audit = quality_filter(
        linguistic_raw, cohort="linguistic", **quality_kwargs
    )
    unavailable = [feature for feature in canonical if feature not in linguistic_quality]
    if unavailable:
        raise ValueError(
            "canonical full-cohort features failed linguistic-cohort quality checks: "
            + ", ".join(unavailable)
        )

    if "linguistic_features" in config:
        requested = list(config["linguistic_features"])
        missing_requested = [
            feature for feature in requested if feature not in linguistic_quality.columns
        ]
        if missing_requested:
            raise ValueError(
                "configured linguistic features failed quality checks: "
                + ", ".join(missing_requested)
            )
        additions = [feature for feature in requested if feature not in canonical]
    else:
        # Features unavailable after full-cohort quality filtering but complete in
        # the transcript-eligible cohort are the linguistic additions.
        additions = [
            feature
            for feature in linguistic_quality.columns
            if feature not in set(full_available)
        ]
    if not additions:
        raise ValueError("no linguistic feature additions survived quality filtering")

    additions_preferred, linguistic_preference_audit = apply_sd_preference(
        linguistic_quality.loc[:, additions], selection_stage="linguistic_additions"
    )
    linguistic_input = pd.concat(
        [
            linguistic_quality.loc[:, canonical],
            additions_preferred,
        ],
        axis=1,
    )
    linguistic_selected, linguistic_correlation_audit = reject_correlated(
        linguistic_input,
        threshold=float(config["maximum_absolute_correlation"]),
        selection_stage="linguistic_additions",
        protected_features=canonical,
    )
    _assert_no_paired_volatility(
        list(linguistic_selected.columns), list(linguistic_input.columns)
    )

    # Ensure shared measurements agree exactly up to numerical serialization.
    full_lookup = full_raw.set_index(VIDEO_ID_COLUMN)
    linguistic_lookup = linguistic_raw.set_index(VIDEO_ID_COLUMN)
    shared_ids = linguistic_lookup.index
    for feature in canonical:
        left = pd.to_numeric(full_lookup.loc[shared_ids, feature]).to_numpy(float)
        right = pd.to_numeric(linguistic_lookup.loc[shared_ids, feature]).to_numpy(float)
        if not np.allclose(left, right, rtol=1e-10, atol=1e-12):
            raise ValueError(
                f"canonical feature {feature!r} differs between candidate tables"
            )

    output = Path(output_dir)
    destinations = {
        "full_features": output / "decorrelated_df_full-dataset.csv",
        "linguistic_features": output / "decorrelated_df_linguistic-dataset.csv",
        "quality_audit": output / "quality_filter_audit.csv",
        "preference_audit": output / "sd_preference_audit.csv",
        "correlation_audit": output / "correlation_audit.csv",
        "canonical_features": output / "canonical_audio_visual_features.csv",
    }
    _write_csv(
        pd.concat(
            [
                full_raw[[VIDEO_ID_COLUMN]].reset_index(drop=True),
                full_selected.reset_index(drop=True),
            ],
            axis=1,
        ),
        destinations["full_features"],
    )
    _write_csv(
        pd.concat(
            [
                linguistic_raw[[VIDEO_ID_COLUMN]].reset_index(drop=True),
                linguistic_selected.reset_index(drop=True),
            ],
            axis=1,
        ),
        destinations["linguistic_features"],
    )
    _write_csv(
        pd.concat([full_quality_audit, linguistic_quality_audit], ignore_index=True),
        destinations["quality_audit"],
    )
    _write_csv(
        pd.concat(
            [full_preference_audit, linguistic_preference_audit], ignore_index=True
        ),
        destinations["preference_audit"],
    )
    _write_csv(
        pd.concat(
            [full_correlation_audit, linguistic_correlation_audit], ignore_index=True
        ),
        destinations["correlation_audit"],
    )
    _write_csv(
        pd.DataFrame({"feature": canonical}), destinations["canonical_features"]
    )
    return destinations
