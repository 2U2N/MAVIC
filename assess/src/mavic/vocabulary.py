"""Loading and validation for the authoritative MAVIC v1 vocabulary."""

from __future__ import annotations

import os
from importlib.resources import files
from pathlib import Path

import pandas as pd


VOCABULARY_COLUMNS = (
    "feature",
    "display_name",
    "feature_type",
    "is_variability",
    "source_cohort",
    "vocabulary_version",
)
FEATURE_TYPES = ("audio", "visual", "linguistic")
SOURCE_COHORTS = ("full", "linguistic")
MAVIC_V1_COUNTS = {"visual": 86, "audio": 13, "linguistic": 34}
MAVIC_V1_TOTAL = sum(MAVIC_V1_COUNTS.values())


def is_variability_name(feature: str) -> bool:
    """Return whether a feature follows a MAVIC SD naming convention."""
    return feature.endswith(("_sd", "_std")) or feature.startswith(("std_", "sd_"))


def default_vocabulary_path() -> Path:
    """Locate the package-data copy of the authoritative vocabulary."""
    override = os.environ.get("MAVIC_VOCABULARY")
    if override:
        return Path(override)
    resource = files("mavic").joinpath("data", "mavic_v1.csv")
    return Path(str(resource))


def _parse_boolean(series: pd.Series, *, source: Path) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    accepted = {"true": True, "false": False}
    invalid = sorted(set(normalized) - set(accepted))
    if invalid:
        raise ValueError(
            f"{source}: is_variability must contain only true/false values; "
            f"found {', '.join(invalid)}"
        )
    return normalized.map(accepted).astype(bool)


def validate_vocabulary(
    frame: pd.DataFrame,
    *,
    source: Path | str = "<vocabulary>",
    require_mavic_v1: bool = True,
) -> pd.DataFrame:
    """Validate and return a normalized vocabulary table.

    ``require_mavic_v1=False`` is useful for extensions and small synthetic
    tests. The authoritative MAVIC v1 table is always loaded in strict mode.
    """
    source = Path(source)
    missing = [column for column in VOCABULARY_COLUMNS if column not in frame.columns]
    extra = [column for column in frame.columns if column not in VOCABULARY_COLUMNS]
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise ValueError(f"{source}: invalid vocabulary schema ({'; '.join(details)})")

    result = frame.loc[:, VOCABULARY_COLUMNS].copy()
    for column in ("feature", "display_name", "feature_type", "source_cohort", "vocabulary_version"):
        if result[column].isna().any():
            raise ValueError(f"{source}: {column} contains missing values")
        result[column] = result[column].astype(str)
        if result[column].str.strip().ne(result[column]).any():
            raise ValueError(f"{source}: {column} values may not have surrounding whitespace")
        if result[column].eq("").any():
            raise ValueError(f"{source}: {column} values must be non-empty")

    if result["feature"].duplicated().any():
        duplicates = sorted(result.loc[result["feature"].duplicated(False), "feature"].unique())
        raise ValueError(f"{source}: duplicate feature names: {', '.join(duplicates)}")
    if result["display_name"].duplicated().any():
        duplicates = sorted(
            result.loc[result["display_name"].duplicated(False), "display_name"].unique()
        )
        raise ValueError(f"{source}: duplicate display names: {', '.join(duplicates)}")

    invalid_types = sorted(set(result["feature_type"]) - set(FEATURE_TYPES))
    if invalid_types:
        raise ValueError(f"{source}: invalid feature types: {', '.join(invalid_types)}")
    invalid_cohorts = sorted(set(result["source_cohort"]) - set(SOURCE_COHORTS))
    if invalid_cohorts:
        raise ValueError(f"{source}: invalid source cohorts: {', '.join(invalid_cohorts)}")

    result["is_variability"] = _parse_boolean(result["is_variability"], source=source)
    inconsistent_variability = result[
        result["is_variability"] != result["feature"].map(is_variability_name)
    ]
    if not inconsistent_variability.empty:
        raise ValueError(
            f"{source}: is_variability disagrees with feature naming for: "
            + ", ".join(inconsistent_variability["feature"])
        )

    bad_source_types = result[
        ((result["source_cohort"] == "full") & (result["feature_type"] == "linguistic"))
        | (
            (result["source_cohort"] == "linguistic")
            & (result["feature_type"] != "linguistic")
        )
    ]
    if not bad_source_types.empty:
        raise ValueError(
            f"{source}: full features must be audio/visual and linguistic features "
            "must use the linguistic cohort"
        )

    if require_mavic_v1:
        versions = set(result["vocabulary_version"])
        if versions != {"mavic-v1"}:
            raise ValueError(
                f"{source}: authoritative vocabulary_version must be mavic-v1"
            )
        counts = result["feature_type"].value_counts().to_dict()
        if len(result) != MAVIC_V1_TOTAL or any(
            counts.get(feature_type, 0) != expected
            for feature_type, expected in MAVIC_V1_COUNTS.items()
        ):
            found = ", ".join(
                f"{feature_type}={counts.get(feature_type, 0)}"
                for feature_type in FEATURE_TYPES
            )
            raise ValueError(
                f"{source}: expected {MAVIC_V1_TOTAL} probes "
                f"({', '.join(f'{key}={value}' for key, value in MAVIC_V1_COUNTS.items())}); "
                f"found {len(result)} ({found})"
            )

    return result.reset_index(drop=True)


def load_vocabulary(
    path: Path | str | None = None, *, require_mavic_v1: bool = True
) -> pd.DataFrame:
    """Read the vocabulary CSV and apply strict schema validation."""
    if path is None and not os.environ.get("MAVIC_VOCABULARY"):
        resource = files("mavic").joinpath("data", "mavic_v1.csv")
        source: Path | str = "mavic:data/mavic_v1.csv"
        try:
            with resource.open("rb") as handle:
                frame = pd.read_csv(handle)
        except (OSError, pd.errors.ParserError) as error:
            raise ValueError(f"could not read vocabulary CSV {source}: {error}") from error
    else:
        source = Path(path) if path is not None else default_vocabulary_path()
        if not Path(source).is_file():
            raise ValueError(f"vocabulary CSV not found: {source}")
        try:
            frame = pd.read_csv(source)
        except (OSError, pd.errors.ParserError) as error:
            raise ValueError(f"could not read vocabulary CSV {source}: {error}") from error
    return validate_vocabulary(
        frame, source=source, require_mavic_v1=require_mavic_v1
    )
