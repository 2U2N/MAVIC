"""Compare MAVIC feature coverage across three video representations.

Audio and visual targets are evaluated on the full MAVIC cohort. Linguistic
targets are evaluated only on videos with usable linguistic measurements. All
three representations use the same videos and cross-validation fold assignment
within each cohort.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

# Use a non-interactive backend so chart creation also works on remote machines.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .common import (
        FEATURE_TYPES,
        GENERATED_DIR,
        RESTRICTED_INPUTS_DIR,
        TYPE_LABELS,
        darken_color,
        display_name_mapping,
        feature_color,
        feature_type_mapping,
        is_variability_feature,
        load_palette,
    )
except ImportError:  # Allow ``python paper/analysis/<script>.py``.
    from common import (  # type: ignore
        FEATURE_TYPES,
        GENERATED_DIR,
        RESTRICTED_INPUTS_DIR,
        TYPE_LABELS,
        darken_color,
        display_name_mapping,
        feature_color,
        feature_type_mapping,
        is_variability_feature,
        load_palette,
    )

feature_display_names = display_name_mapping()
feature_type_dict = feature_type_mapping()
feature_type_colors, variability_feature_type_colors = load_palette()


DEFAULT_FULL_MAVIC_CSV = RESTRICTED_INPUTS_DIR / "full_features.csv"
DEFAULT_LINGUISTIC_MAVIC_CSV = Path(
    RESTRICTED_INPUTS_DIR / "linguistic_features.csv"
)
DEFAULT_OPENCLIP_DIR = RESTRICTED_INPUTS_DIR / "representations" / "openclip"
DEFAULT_LCO_DIR = RESTRICTED_INPUTS_DIR / "representations" / "lco"
DEFAULT_F2LLM_DIR = RESTRICTED_INPUTS_DIR / "representations" / "f2llm"
DEFAULT_OUTPUT_DIR = GENERATED_DIR / "coverage"

VIDEO_ID_COLUMN = "video_id"
ALPHAS = np.asarray([0.1, 1, 10, 100, 1_000, 10_000, 100_000], dtype=float)
N_SPLITS = 5
RANDOM_STATE = 42


def load_embedding_shards(output_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate all embedding shards listed in a run manifest."""
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"embedding manifest not found: {manifest_path}")

    try:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read embedding manifest {manifest_path}: {error}") from error

    shards = manifest.get("shards") if isinstance(manifest, dict) else None
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"embedding manifest contains no shards: {manifest_path}")

    all_ids: list[np.ndarray] = []
    all_embeddings: list[np.ndarray] = []
    seen_ids: set[str] = set()

    try:
        ordered_shards = sorted(shards, key=lambda item: int(item["index"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid shard inventory in {manifest_path}") from error

    for shard in ordered_shards:
        filename = shard.get("file") if isinstance(shard, dict) else None
        expected_count = shard.get("count") if isinstance(shard, dict) else None
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"invalid shard filename in {manifest_path}")

        shard_path = output_dir / filename
        if not shard_path.is_file():
            raise ValueError(f"manifest references missing shard: {shard_path}")
        try:
            with np.load(shard_path, allow_pickle=False) as data:
                if "video_ids" not in data.files or "embeddings" not in data.files:
                    raise ValueError(
                        f"{shard_path} must contain video_ids and embeddings arrays"
                    )
                video_ids = np.asarray(data["video_ids"], dtype=np.str_)
                embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        except (OSError, KeyError, ValueError) as error:
            if isinstance(error, ValueError) and str(error).startswith(str(shard_path)):
                raise
            raise ValueError(f"could not read embedding shard {shard_path}: {error}") from error

        if video_ids.ndim != 1:
            raise ValueError(f"{shard_path}: video_ids must be one-dimensional")
        if embeddings.ndim != 2 or len(embeddings) != len(video_ids):
            raise ValueError(
                f"{shard_path}: embeddings must be a matrix aligned with video_ids"
            )
        if len(video_ids) == 0:
            raise ValueError(f"{shard_path}: empty shards are not supported")
        if isinstance(expected_count, bool) or not isinstance(expected_count, int):
            raise ValueError(f"invalid shard count for {shard_path}")
        if expected_count != len(video_ids):
            raise ValueError(
                f"manifest count does not match {filename}: "
                f"{expected_count} != {len(video_ids)}"
            )
        if any(not value for value in video_ids.tolist()):
            raise ValueError(f"{shard_path}: video IDs must be non-empty")
        if not np.isfinite(embeddings).all():
            raise ValueError(f"{shard_path}: embeddings contain non-finite values")

        shard_ids = set(video_ids.tolist())
        if len(shard_ids) != len(video_ids):
            raise ValueError(f"{shard_path}: duplicate video IDs")
        overlap = seen_ids.intersection(shard_ids)
        if overlap:
            raise ValueError(
                "duplicate video IDs across shards: " + ", ".join(sorted(overlap))
            )
        seen_ids.update(shard_ids)
        all_ids.append(video_ids)
        all_embeddings.append(embeddings)

    dimensions = {matrix.shape[1] for matrix in all_embeddings}
    if len(dimensions) != 1:
        raise ValueError(f"embedding dimensions differ across shards in {output_dir}")

    return np.concatenate(all_ids), np.concatenate(all_embeddings, axis=0)


def load_mavic_features(
    csv_path: Path, *, expected_feature_count: int | None = None
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Load a MAVIC CSV and return IDs, feature names, and numeric values."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise ValueError(f"MAVIC feature CSV not found: {csv_path}")

    try:
        frame = pd.read_csv(csv_path, dtype={VIDEO_ID_COLUMN: "string"})
    except (OSError, pd.errors.ParserError) as error:
        raise ValueError(f"could not read MAVIC feature CSV {csv_path}: {error}") from error

    if VIDEO_ID_COLUMN not in frame.columns:
        raise ValueError(f"{csv_path} must contain a {VIDEO_ID_COLUMN!r} column")
    video_ids = frame[VIDEO_ID_COLUMN].astype(str).to_numpy()
    if any(not value for value in video_ids.tolist()):
        raise ValueError("MAVIC video IDs must be non-empty")
    if len(set(video_ids.tolist())) != len(video_ids):
        raise ValueError("MAVIC CSV contains duplicate video IDs")

    feature_names = [name for name in frame.columns if name != VIDEO_ID_COLUMN]
    if expected_feature_count is not None and len(feature_names) != expected_feature_count:
        raise ValueError(
            f"expected {expected_feature_count} MAVIC features, found {len(feature_names)}"
        )
    if not feature_names:
        raise ValueError("MAVIC CSV contains no feature columns")

    try:
        numeric = frame[feature_names].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("all MAVIC feature values must be numeric") from error
    features = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(features).all():
        raise ValueError("MAVIC feature values must all be finite")

    return video_ids, feature_names, features


def validate_feature_tables(
    full_ids: np.ndarray,
    full_names: Sequence[str],
    full_values: np.ndarray,
    linguistic_ids: np.ndarray,
    linguistic_names: Sequence[str],
    linguistic_values: np.ndarray,
    *,
    display_names: Mapping[str, str],
    type_by_feature: Mapping[str, str],
    colors_by_type: Mapping[str, str],
) -> list[str]:
    """Validate cross-table invariants and return linguistic-only features."""
    full_set = set(full_names)
    linguistic_set = set(linguistic_names)
    shared_names = [name for name in full_names if name in linguistic_set]
    if not shared_names:
        raise ValueError("the two MAVIC CSVs contain no shared feature columns")

    full_id_set = set(full_ids.tolist())
    extra_linguistic_ids = sorted(set(linguistic_ids.tolist()) - full_id_set)
    if extra_linguistic_ids:
        raise ValueError(
            "linguistic MAVIC video IDs must be a subset of full MAVIC video IDs"
        )

    linguistic_only = [name for name in linguistic_names if name not in full_set]
    if not linguistic_only:
        raise ValueError("linguistic MAVIC CSV contains no linguistic-only features")

    selected_names = list(full_names) + linguistic_only
    missing_display_names = [name for name in selected_names if name not in display_names]
    if missing_display_names:
        raise ValueError(
            "features are missing readable names in feature_display_names: "
            + ", ".join(missing_display_names)
        )
    missing_types = [name for name in selected_names if name not in type_by_feature]
    if missing_types:
        raise ValueError(
            "features are missing types in feature_type_dict: "
            + ", ".join(missing_types)
        )

    invalid_full_types = [
        name for name in full_names if type_by_feature[name] not in {"audio", "visual"}
    ]
    if invalid_full_types:
        raise ValueError(
            "full MAVIC features must be audio or visual: "
            + ", ".join(invalid_full_types)
        )
    invalid_linguistic_types = [
        name for name in linguistic_only if type_by_feature[name] != "linguistic"
    ]
    if invalid_linguistic_types:
        raise ValueError(
            "linguistic-only MAVIC features must be typed linguistic: "
            + ", ".join(invalid_linguistic_types)
        )
    selected_types = {type_by_feature[name] for name in selected_names}
    missing_colors = sorted(selected_types - set(colors_by_type))
    if missing_colors:
        raise ValueError(
            "feature types are missing colors in feature_type_colors: "
            + ", ".join(missing_colors)
        )

    full_id_positions = {video_id: index for index, video_id in enumerate(full_ids)}
    full_column_positions = {name: index for index, name in enumerate(full_names)}
    linguistic_column_positions = {
        name: index for index, name in enumerate(linguistic_names)
    }
    aligned_full = full_values[
        np.asarray([full_id_positions[video_id] for video_id in linguistic_ids])
    ][:, np.asarray([full_column_positions[name] for name in shared_names])]
    aligned_linguistic = linguistic_values[
        :, np.asarray([linguistic_column_positions[name] for name in shared_names])
    ]
    if not np.allclose(
        aligned_full,
        aligned_linguistic,
        rtol=1e-10,
        atol=1e-12,
        equal_nan=False,
    ):
        differences = np.abs(aligned_full - aligned_linguistic)
        row_index, column_index = np.unravel_index(
            int(np.argmax(differences)), differences.shape
        )
        raise ValueError(
            "shared MAVIC feature values differ between datasets; first maximum "
            f"difference is for video {linguistic_ids[row_index]!r}, feature "
            f"{shared_names[column_index]!r}"
        )

    return linguistic_only


def align_cohort(
    mavic_ids: np.ndarray,
    targets: np.ndarray,
    embeddings: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    n_splits: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Align one feature cohort and every embedding matrix to sorted shared IDs."""
    shared = set(mavic_ids.tolist())
    for embedding_ids, _ in embeddings.values():
        shared.intersection_update(embedding_ids.tolist())
    shared_ids = np.asarray(sorted(shared), dtype=np.str_)
    if len(shared_ids) < n_splits:
        raise ValueError(
            f"at least {n_splits} shared videos are required, found {len(shared_ids)}"
        )

    def positions(ids: np.ndarray) -> np.ndarray:
        lookup = {video_id: index for index, video_id in enumerate(ids.tolist())}
        return np.asarray([lookup[video_id] for video_id in shared_ids], dtype=np.int64)

    aligned_embeddings = {
        name: matrix[positions(ids)] for name, (ids, matrix) in embeddings.items()
    }
    return shared_ids, targets[positions(mavic_ids)], aligned_embeddings


def build_nested_folds(
    full_ids: np.ndarray,
    linguistic_ids: np.ndarray,
    *,
    n_splits: int,
    random_state: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    """Build full folds and inherit their test-fold assignment for linguistic IDs."""
    if len(set(full_ids.tolist())) != len(full_ids):
        raise ValueError("full cohort contains duplicate video IDs")
    if not set(linguistic_ids.tolist()).issubset(full_ids.tolist()):
        raise ValueError("linguistic cohort must be a subset of the full cohort")

    full_folds = list(
        KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(full_ids)
    )
    test_fold_by_id: dict[str, int] = {}
    for fold_index, (_, test_indices) in enumerate(full_folds):
        for index in test_indices:
            test_fold_by_id[str(full_ids[index])] = fold_index

    linguistic_assignments = np.asarray(
        [test_fold_by_id[str(video_id)] for video_id in linguistic_ids], dtype=np.int64
    )
    linguistic_folds: list[tuple[np.ndarray, np.ndarray]] = []
    all_indices = np.arange(len(linguistic_ids), dtype=np.int64)
    for fold_index in range(n_splits):
        test_indices = all_indices[linguistic_assignments == fold_index]
        train_indices = all_indices[linguistic_assignments != fold_index]
        if len(test_indices) == 0 or len(train_indices) == 0:
            raise ValueError(
                f"linguistic cohort has an empty train or test split in fold {fold_index}"
            )
        linguistic_folds.append((train_indices, test_indices))

    return full_folds, linguistic_folds


def score_representation(
    embeddings: np.ndarray,
    targets: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    alphas: np.ndarray = ALPHAS,
) -> np.ndarray:
    """Return one out-of-fold R-squared score per MAVIC feature."""
    model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=alphas, alpha_per_target=True),
    )
    predictions = cross_val_predict(model, embeddings, targets, cv=folds, n_jobs=1)
    scores = np.atleast_1d(
        np.asarray(
            r2_score(targets, predictions, multioutput="raw_values"),
            dtype=np.float64,
        )
    )
    if not np.isfinite(scores).all():
        raise ValueError("coverage analysis produced non-finite R-squared scores")
    return scores


def is_variability_feature(feature: str) -> bool:
    """Return whether a name follows one of the project's SD conventions."""
    return feature.endswith(("_sd", "_std")) or feature.startswith(("std_", "sd_"))


def darken_color(color: str, *, retained_fraction: float = 0.70) -> tuple[float, ...]:
    """Blend a Matplotlib color toward black by retaining its RGB fraction."""
    if not 0.0 <= retained_fraction <= 1.0:
        raise ValueError("retained_fraction must be between zero and one")
    return tuple(channel * retained_fraction for channel in to_rgb(color))


def feature_color(
    feature_type: str,
    is_variability: bool,
    *,
    colors_by_type: Mapping[str, str],
    variability_colors_by_type: Mapping[str, str] = variability_feature_type_colors,
) -> str | tuple[float, ...]:
    """Return the explicit ordinary or SD/variability modality color."""
    palette = variability_colors_by_type if is_variability else colors_by_type
    return palette[feature_type]


def shared_x_limits(score_arrays: Sequence[np.ndarray]) -> tuple[float, float]:
    """Return padded limits spanning every score and zero."""
    combined = np.concatenate([*(np.asarray(scores) for scores in score_arrays), [0.0]])
    lower = float(combined.min())
    upper = float(combined.max())
    span = max(upper - lower, 0.1)
    return lower - 0.03 * span, upper + 0.03 * span


def plot_scores(
    results: pd.DataFrame,
    *,
    score_column: str,
    title: str,
    destination: Path,
    x_limits: tuple[float, float],
    full_n: int,
    linguistic_n: int,
    colors_by_type: Mapping[str, str],
    variability_colors_by_type: Mapping[str, str] = variability_feature_type_colors,
) -> None:
    """Save one globally feature-ranked horizontal R-squared chart."""
    ordered = results.sort_values(score_column, kind="mergesort")
    labels = [
        textwrap.fill(str(label), width=48, break_long_words=False)
        for label in ordered["display_name"]
    ]
    colors = [
        feature_color(
            str(feature_type),
            bool(variability),
            colors_by_type=colors_by_type,
            variability_colors_by_type=variability_colors_by_type,
        )
        for feature_type, variability in zip(
            ordered["feature_type"], ordered["is_variability"]
        )
    ]
    edges = [
        darken_color(colors_by_type[str(feature_type)])
        for feature_type in ordered["feature_type"]
    ]

    figure_height = max(10.0, 0.31 * len(ordered))
    figure, axis = plt.subplots(figsize=(16, figure_height))
    y_positions = np.arange(len(ordered))
    axis.barh(
        y_positions,
        ordered[score_column].to_numpy(),
        color=colors,
        edgecolor=edges,
        linewidth=0.45,
    )
    axis.set_yticks(y_positions, labels=labels, fontsize=8)
    axis.axvline(0.0, color="#202020", linewidth=1.0, zorder=3)
    axis.set_xlim(*x_limits)
    axis.set_xlabel("Cross-validated $R^2$")
    axis.set_title(
        f"{title}\n"
        f"Audio/visual features: n={full_n:,}  •  "
        f"Linguistic features: n={linguistic_n:,}",
        pad=16,
    )
    axis.grid(axis="x", color="#d0d0d0", linewidth=0.7, alpha=0.65)
    axis.set_axisbelow(True)

    type_handles = [
        Patch(
            facecolor=colors_by_type[feature_type],
            edgecolor=darken_color(colors_by_type[feature_type]),
            label=TYPE_LABELS[feature_type],
        )
        for feature_type in FEATURE_TYPES
    ]
    type_legend = axis.legend(
        handles=type_handles,
        title="Feature type",
        loc="lower left",
        bbox_to_anchor=(1.01, 0.07),
        frameon=True,
    )
    axis.add_artist(type_legend)
    shade_handles = [
        Patch(facecolor="#707070", edgecolor="#505050", label="Ordinary feature"),
        Patch(facecolor="#eeeeee", edgecolor="#707070", label="SD / variability"),
    ]
    axis.legend(
        handles=shade_handles,
        title="Shade",
        loc="lower left",
        bbox_to_anchor=(1.01, 0.0),
        frameon=True,
    )

    figure.tight_layout()
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)


def run_analysis(
    full_mavic_csv: Path = DEFAULT_FULL_MAVIC_CSV,
    linguistic_mavic_csv: Path = DEFAULT_LINGUISTIC_MAVIC_CSV,
    openclip_dir: Path = DEFAULT_OPENCLIP_DIR,
    lco_dir: Path = DEFAULT_LCO_DIR,
    f2llm_dir: Path = DEFAULT_F2LLM_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
    alphas: np.ndarray = ALPHAS,
    display_names: Mapping[str, str] = feature_display_names,
    type_by_feature: Mapping[str, str] = feature_type_dict,
    colors_by_type: Mapping[str, str] = feature_type_colors,
    variability_colors_by_type: Mapping[str, str] = variability_feature_type_colors,
) -> pd.DataFrame:
    """Run the complete two-cohort coverage analysis and save its artifacts."""
    full_ids, full_names, full_targets = load_mavic_features(full_mavic_csv)
    linguistic_ids, linguistic_names, linguistic_targets = load_mavic_features(
        linguistic_mavic_csv
    )
    linguistic_only = validate_feature_tables(
        full_ids,
        full_names,
        full_targets,
        linguistic_ids,
        linguistic_names,
        linguistic_targets,
        display_names=display_names,
        type_by_feature=type_by_feature,
        colors_by_type=colors_by_type,
    )

    embeddings = {
        "openclip": load_embedding_shards(openclip_dir),
        "lco": load_embedding_shards(lco_dir),
        "f2llm": load_embedding_shards(f2llm_dir),
    }

    full_shared_ids, full_targets, full_embeddings = align_cohort(
        full_ids, full_targets, embeddings, n_splits=n_splits
    )
    linguistic_positions = {
        name: index for index, name in enumerate(linguistic_names)
    }
    linguistic_only_targets = linguistic_targets[
        :, np.asarray([linguistic_positions[name] for name in linguistic_only])
    ]
    linguistic_shared_ids, linguistic_only_targets, linguistic_embeddings = align_cohort(
        linguistic_ids,
        linguistic_only_targets,
        embeddings,
        n_splits=n_splits,
    )
    if not set(linguistic_shared_ids.tolist()).issubset(full_shared_ids.tolist()):
        raise ValueError("matched linguistic cohort is not a subset of the full cohort")

    print(
        f"Matched {len(full_shared_ids):,} full-cohort videos and "
        f"{len(linguistic_shared_ids):,} linguistic-cohort videos across all models."
    )
    full_folds, linguistic_folds = build_nested_folds(
        full_shared_ids,
        linguistic_shared_ids,
        n_splits=n_splits,
        random_state=random_state,
    )

    score_columns: dict[str, np.ndarray] = {}
    for model_name, label in (
        ("openclip", "OpenCLIP"),
        ("lco", "LCO"),
        ("f2llm", "F2LLM metadata"),
    ):
        print(f"Scoring {label} embeddings on the full cohort...")
        full_scores = score_representation(
            full_embeddings[model_name], full_targets, full_folds, alphas=alphas
        )
        print(f"Scoring {label} embeddings on the linguistic cohort...")
        linguistic_scores = score_representation(
            linguistic_embeddings[model_name],
            linguistic_only_targets,
            linguistic_folds,
            alphas=alphas,
        )
        score_columns[f"{model_name}_r2"] = np.concatenate(
            [full_scores, linguistic_scores]
        )

    all_names = list(full_names) + linguistic_only
    results = pd.DataFrame(
        {
            "feature": all_names,
            "display_name": [display_names[name] for name in all_names],
            "feature_type": [type_by_feature[name] for name in all_names],
            "is_variability": [is_variability_feature(name) for name in all_names],
            "source_dataset": ["full"] * len(full_names)
            + ["linguistic"] * len(linguistic_only),
            "n_videos": [len(full_shared_ids)] * len(full_names)
            + [len(linguistic_shared_ids)] * len(linguistic_only),
            **score_columns,
        }
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "coverage_scores.csv", index=False)

    x_limits = shared_x_limits(list(score_columns.values()))
    plot_specs = (
        ("openclip_r2", "OpenCLIP MAVIC feature coverage", "openclip_r2_coverage.png"),
        ("lco_r2", "LCO MAVIC feature coverage", "lco_r2_coverage.png"),
        ("f2llm_r2", "F2LLM metadata MAVIC feature coverage", "f2llm_r2_coverage.png"),
    )
    for score_column, title, filename in plot_specs:
        plot_scores(
            results,
            score_column=score_column,
            title=title,
            destination=output_dir / filename,
            x_limits=x_limits,
            full_n=len(full_shared_ids),
            linguistic_n=len(linguistic_shared_ids),
            colors_by_type=colors_by_type,
            variability_colors_by_type=variability_colors_by_type,
        )

    print(f"Saved coverage results to {output_dir}")
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare full-cohort and linguistic MAVIC feature coverage in "
            "OpenCLIP, LCO, and F2LLM embeddings"
        )
    )
    parser.add_argument(
        "--full-mavic-csv", type=Path, default=DEFAULT_FULL_MAVIC_CSV
    )
    parser.add_argument(
        "--linguistic-mavic-csv",
        type=Path,
        default=DEFAULT_LINGUISTIC_MAVIC_CSV,
    )
    parser.add_argument("--openclip-dir", type=Path, default=DEFAULT_OPENCLIP_DIR)
    parser.add_argument("--lco-dir", type=Path, default=DEFAULT_LCO_DIR)
    parser.add_argument("--f2llm-dir", type=Path, default=DEFAULT_F2LLM_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        run_analysis(
            full_mavic_csv=args.full_mavic_csv,
            linguistic_mavic_csv=args.linguistic_mavic_csv,
            openclip_dir=args.openclip_dir,
            lco_dir=args.lco_dir,
            f2llm_dir=args.f2llm_dir,
            output_dir=args.output_dir,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
