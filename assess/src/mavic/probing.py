"""Linear probing of arbitrary computational representations with MAVIC."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .io import (
    FeatureTable,
    Representation,
    load_feature_table,
    load_representation,
    parse_representation_specs,
)
from .vocabulary import load_vocabulary, validate_vocabulary


DEFAULT_ALPHAS = np.asarray(
    [0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0],
    dtype=np.float64,
)
DEFAULT_FOLDS = 5
DEFAULT_SEED = 42
SCORE_COLUMNS = (
    "feature",
    "display_name",
    "feature_type",
    "is_variability",
    "source_cohort",
    "n_videos",
    "representation",
    "r2",
)


def _positions(source_ids: np.ndarray, ordered_ids: np.ndarray) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(source_ids.tolist())}
    return np.asarray([lookup[value] for value in ordered_ids], dtype=np.int64)


def validate_feature_cohorts(
    full: FeatureTable,
    linguistic: FeatureTable,
    vocabulary: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Validate selected tables and return full/linguistic feature lists."""
    full_expected = vocabulary.loc[
        vocabulary["source_cohort"] == "full", "feature"
    ].tolist()
    linguistic_expected = vocabulary.loc[
        vocabulary["source_cohort"] == "linguistic", "feature"
    ].tolist()
    expected_full_set = set(full_expected)
    expected_linguistic_set = expected_full_set | set(linguistic_expected)
    if set(full.feature_names) != expected_full_set:
        missing = sorted(expected_full_set - set(full.feature_names))
        extra = sorted(set(full.feature_names) - expected_full_set)
        raise ValueError(
            "full feature table does not match the vocabulary"
            + (f"; missing: {', '.join(missing)}" if missing else "")
            + (f"; unexpected: {', '.join(extra)}" if extra else "")
        )
    if set(linguistic.feature_names) != expected_linguistic_set:
        missing = sorted(expected_linguistic_set - set(linguistic.feature_names))
        extra = sorted(set(linguistic.feature_names) - expected_linguistic_set)
        raise ValueError(
            "linguistic feature table does not match the vocabulary"
            + (f"; missing: {', '.join(missing)}" if missing else "")
            + (f"; unexpected: {', '.join(extra)}" if extra else "")
        )
    full_ids = set(full.video_ids.tolist())
    if not set(linguistic.video_ids.tolist()).issubset(full_ids):
        raise ValueError("linguistic video IDs must be a subset of full video IDs")

    full_columns = {name: index for index, name in enumerate(full.feature_names)}
    linguistic_columns = {
        name: index for index, name in enumerate(linguistic.feature_names)
    }
    row_positions = _positions(full.video_ids, linguistic.video_ids)
    left = full.values[row_positions][
        :, np.asarray([full_columns[name] for name in full_expected])
    ]
    right = linguistic.values[
        :, np.asarray([linguistic_columns[name] for name in full_expected])
    ]
    if not np.allclose(left, right, rtol=1e-10, atol=1e-12):
        difference = np.abs(left - right)
        row, column = np.unravel_index(int(np.argmax(difference)), difference.shape)
        raise ValueError(
            "shared feature values differ between cohorts for "
            f"video {linguistic.video_ids[row]!r}, feature {full_expected[column]!r}"
        )
    return full_expected, linguistic_expected


def align_probe_inputs(
    full: FeatureTable,
    linguistic: FeatureTable,
    representations: Sequence[Representation],
    *,
    n_splits: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """Align both cohorts and every representation to sorted shared IDs."""
    if not representations:
        raise ValueError("at least one representation is required")
    shared = set(full.video_ids.tolist())
    for representation in representations:
        shared.intersection_update(representation.video_ids.tolist())
    full_ids = np.asarray(sorted(shared), dtype=np.str_)
    if len(full_ids) < n_splits:
        raise ValueError(
            f"at least {n_splits} shared full-cohort videos are required; "
            f"found {len(full_ids)}"
        )
    linguistic_ids = np.asarray(
        sorted(set(linguistic.video_ids.tolist()).intersection(full_ids.tolist())),
        dtype=np.str_,
    )
    if len(linguistic_ids) < n_splits:
        raise ValueError(
            f"at least {n_splits} shared linguistic-cohort videos are required; "
            f"found {len(linguistic_ids)}"
        )

    full_values = full.values[_positions(full.video_ids, full_ids)]
    linguistic_values = linguistic.values[
        _positions(linguistic.video_ids, linguistic_ids)
    ]
    full_embeddings = {
        item.name: item.embeddings[_positions(item.video_ids, full_ids)]
        for item in representations
    }
    linguistic_embeddings = {
        item.name: item.embeddings[_positions(item.video_ids, linguistic_ids)]
        for item in representations
    }
    return (
        full_ids,
        linguistic_ids,
        {"targets": full_values, **full_embeddings},
        {"targets": linguistic_values, **linguistic_embeddings},
    )


def build_inherited_folds(
    full_ids: np.ndarray,
    linguistic_ids: np.ndarray,
    *,
    n_splits: int = DEFAULT_FOLDS,
    seed: int = DEFAULT_SEED,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    """Build full folds and inherit each video's assignment for linguistics."""
    if len(set(full_ids.tolist())) != len(full_ids):
        raise ValueError("full cohort contains duplicate IDs")
    if not set(linguistic_ids.tolist()).issubset(full_ids.tolist()):
        raise ValueError("linguistic IDs must be a subset of full IDs")
    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    full_folds = list(
        KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(full_ids)
    )
    assignment: dict[str, int] = {}
    for fold, (_, test_indices) in enumerate(full_folds):
        for index in test_indices:
            assignment[str(full_ids[index])] = fold
    linguistic_assignment = np.asarray(
        [assignment[str(video_id)] for video_id in linguistic_ids], dtype=int
    )
    all_indices = np.arange(len(linguistic_ids), dtype=int)
    linguistic_folds = []
    for fold in range(n_splits):
        test = all_indices[linguistic_assignment == fold]
        train = all_indices[linguistic_assignment != fold]
        if not len(test) or not len(train):
            raise ValueError(
                f"linguistic cohort has an empty train/test partition in fold {fold}"
            )
        linguistic_folds.append((train, test))
    return full_folds, linguistic_folds


def score_targets(
    embeddings: np.ndarray,
    targets: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    alphas: np.ndarray = DEFAULT_ALPHAS,
) -> np.ndarray:
    """Return target-specific out-of-fold R² values."""
    embeddings = np.asarray(embeddings, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if embeddings.ndim != 2 or targets.ndim != 2:
        raise ValueError("embeddings and targets must both be two-dimensional")
    if len(embeddings) != len(targets):
        raise ValueError("embeddings and targets must have the same rows")
    if not np.isfinite(embeddings).all() or not np.isfinite(targets).all():
        raise ValueError("probing inputs must be finite")
    if np.any(np.ptp(targets, axis=0) == 0):
        raise ValueError("probing targets must vary in the aligned cohort")
    alphas = np.asarray(alphas, dtype=np.float64)
    if alphas.ndim != 1 or not len(alphas) or np.any(alphas <= 0):
        raise ValueError("alphas must be a non-empty vector of positive values")

    predictions = np.empty_like(targets)
    predicted = np.zeros(len(targets), dtype=bool)
    for train, test in folds:
        train = np.asarray(train, dtype=int)
        test = np.asarray(test, dtype=int)
        if not len(train) or not len(test):
            raise ValueError("folds must have non-empty train and test partitions")
        if predicted[test].any():
            raise ValueError("test folds overlap")
        model = make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=alphas, alpha_per_target=True),
        )
        model.fit(embeddings[train], targets[train])
        fold_predictions = np.asarray(model.predict(embeddings[test]))
        if fold_predictions.ndim == 1:
            fold_predictions = fold_predictions[:, None]
        predictions[test] = fold_predictions
        predicted[test] = True
    if not predicted.all():
        raise ValueError("folds do not predict every row exactly once")
    scores = np.atleast_1d(
        np.asarray(
            r2_score(targets, predictions, multioutput="raw_values"),
            dtype=np.float64,
        )
    )
    if not np.isfinite(scores).all():
        raise ValueError("probing produced non-finite R-squared values")
    return scores


def _select_columns(
    table: FeatureTable, names: Sequence[str], row_values: np.ndarray
) -> np.ndarray:
    positions = {name: index for index, name in enumerate(table.feature_names)}
    return row_values[:, np.asarray([positions[name] for name in names], dtype=int)]


def probe_representations(
    full: FeatureTable,
    linguistic: FeatureTable,
    representations: Sequence[Representation],
    vocabulary: pd.DataFrame,
    *,
    n_splits: int = DEFAULT_FOLDS,
    seed: int = DEFAULT_SEED,
    alphas: np.ndarray = DEFAULT_ALPHAS,
) -> pd.DataFrame:
    """Probe one or more representations and return the generic long table."""
    vocabulary = validate_vocabulary(
        vocabulary,
        source="<provided vocabulary>",
        require_mavic_v1=False,
    )
    full_names, linguistic_names = validate_feature_cohorts(
        full, linguistic, vocabulary
    )
    names = [item.name for item in representations]
    if len(names) != len(set(names)):
        raise ValueError("representation names must be unique")
    if "targets" in names:
        raise ValueError("'targets' is reserved and cannot be a representation name")

    full_ids, linguistic_ids, full_data, linguistic_data = align_probe_inputs(
        full, linguistic, representations, n_splits=n_splits
    )
    full_folds, linguistic_folds = build_inherited_folds(
        full_ids, linguistic_ids, n_splits=n_splits, seed=seed
    )
    full_targets = _select_columns(full, full_names, full_data["targets"])
    linguistic_targets = _select_columns(
        linguistic, linguistic_names, linguistic_data["targets"]
    )
    metadata = vocabulary.set_index("feature")
    rows: list[dict[str, object]] = []
    for representation in representations:
        full_scores = score_targets(
            full_data[representation.name], full_targets, full_folds, alphas=alphas
        )
        linguistic_scores = score_targets(
            linguistic_data[representation.name],
            linguistic_targets,
            linguistic_folds,
            alphas=alphas,
        )
        for cohort, feature_names, scores, count in (
            ("full", full_names, full_scores, len(full_ids)),
            ("linguistic", linguistic_names, linguistic_scores, len(linguistic_ids)),
        ):
            for feature, score in zip(feature_names, scores, strict=True):
                row = metadata.loc[feature]
                rows.append(
                    {
                        "feature": feature,
                        "display_name": row["display_name"],
                        "feature_type": row["feature_type"],
                        "is_variability": bool(row["is_variability"]),
                        "source_cohort": cohort,
                        "n_videos": count,
                        "representation": representation.name,
                        "r2": float(score),
                    }
                )
    return pd.DataFrame(rows, columns=SCORE_COLUMNS)


def run_probe(
    full_features: Path | str,
    linguistic_features: Path | str,
    representation_specs: Sequence[str],
    output_dir: Path | str,
    *,
    vocabulary_path: Path | str | None = None,
    n_splits: int = DEFAULT_FOLDS,
    seed: int = DEFAULT_SEED,
    alphas: np.ndarray = DEFAULT_ALPHAS,
) -> pd.DataFrame:
    """Load, probe, and write ``coverage_scores_long.csv``."""
    full = load_feature_table(full_features)
    linguistic = load_feature_table(linguistic_features)
    specifications = parse_representation_specs(representation_specs)
    representations = [
        load_representation(name, path) for name, path in specifications
    ]
    vocabulary = load_vocabulary(vocabulary_path)
    results = probe_representations(
        full,
        linguistic,
        representations,
        vocabulary,
        n_splits=n_splits,
        seed=seed,
        alphas=alphas,
    )
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / "coverage_scores_long.csv"
    temporary = destination.with_suffix(".csv.tmp")
    results.to_csv(temporary, index=False)
    temporary.replace(destination)

    counts = (
        results.groupby("source_cohort", sort=False)["n_videos"].first().to_dict()
    )
    settings = {
        "schema_version": 1,
        "representations": [item.name for item in representations],
        "n_splits": n_splits,
        "seed": seed,
        "alphas": [float(value) for value in alphas],
        "cohort_sizes": {key: int(value) for key, value in counts.items()},
        "score": "out-of-fold R2",
        "preprocessing": "training-fold StandardScaler",
        "estimator": "target-specific RidgeCV",
    }
    settings_path = destination_dir / "probe_settings.json"
    temporary_settings = settings_path.with_suffix(".json.tmp")
    temporary_settings.write_text(
        json.dumps(settings, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_settings.replace(settings_path)
    return results
