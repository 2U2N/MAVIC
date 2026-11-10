from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mavic.io import FeatureTable, Representation
from mavic.probing import (
    build_inherited_folds,
    probe_representations,
    run_probe,
    score_targets,
)
from mavic.vocabulary import load_vocabulary


def _vocabulary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature": "visual_mean",
                "display_name": "Visual mean",
                "feature_type": "visual",
                "is_variability": False,
                "source_cohort": "full",
                "vocabulary_version": "test",
            },
            {
                "feature": "linguistic_sd",
                "display_name": "Linguistic SD",
                "feature_type": "linguistic",
                "is_variability": True,
                "source_cohort": "linguistic",
                "vocabulary_version": "test",
            },
        ]
    )


def _inputs() -> tuple[FeatureTable, FeatureTable, list[Representation]]:
    rng = np.random.default_rng(12)
    ids = np.asarray([f"v{i:02d}" for i in range(30)])
    embeddings = rng.normal(size=(30, 4)).astype(np.float32)
    visual = embeddings[:, 0] - 0.5 * embeddings[:, 1]
    linguistic_indices = np.arange(0, 30, 2)
    linguistic_ids = ids[linguistic_indices]
    linguistic = embeddings[linguistic_indices, 2] + rng.normal(
        scale=0.05, size=len(linguistic_indices)
    )
    full = FeatureTable(ids, ("visual_mean",), visual[:, None])
    ling = FeatureTable(
        linguistic_ids,
        ("visual_mean", "linguistic_sd"),
        np.column_stack([visual[linguistic_indices], linguistic]),
    )
    shuffled = rng.permutation(len(ids))
    representations = [
        Representation("one", ids[shuffled], embeddings[shuffled], __file__),
        Representation("two", ids, (embeddings * 0.9).astype(np.float32), __file__),
    ]
    return full, ling, representations


def test_probe_multiple_representations_and_nested_cohorts() -> None:
    full, linguistic, representations = _inputs()
    scores = probe_representations(
        full,
        linguistic,
        representations,
        _vocabulary(),
        n_splits=3,
        seed=42,
        alphas=np.asarray([0.1, 1.0]),
    )
    assert len(scores) == 4
    assert scores.groupby("representation")["feature"].nunique().eq(2).all()
    assert set(scores.loc[scores.source_cohort == "full", "n_videos"]) == {30}
    assert set(scores.loc[scores.source_cohort == "linguistic", "n_videos"]) == {15}
    assert np.isfinite(scores["r2"]).all()
    assert (scores["r2"] > 0.7).all()


@pytest.mark.parametrize("representation_count", [1, 2, 3, 4])
def test_probe_accepts_arbitrary_representation_counts(
    representation_count: int,
) -> None:
    full, linguistic, base = _inputs()
    template = base[0]
    representations = [
        Representation(
            f"representation-{index + 1}",
            template.video_ids,
            template.embeddings * (1.0 + index * 0.01),
            template.source,
        )
        for index in range(representation_count)
    ]
    scores = probe_representations(
        full,
        linguistic,
        representations,
        _vocabulary(),
        n_splits=3,
        seed=42,
        alphas=np.asarray([1.0]),
    )
    assert scores["representation"].nunique() == representation_count
    assert len(scores) == representation_count * 2


def test_folds_are_inherited_by_video_id() -> None:
    full = np.asarray([f"v{i}" for i in range(20)])
    linguistic = full[[1, 2, 7, 9, 11, 13, 16, 18]]
    full_folds, linguistic_folds = build_inherited_folds(
        full, linguistic, n_splits=4, seed=8
    )
    full_assignment = {
        full[index]: fold
        for fold, (_, test) in enumerate(full_folds)
        for index in test
    }
    linguistic_assignment = {
        linguistic[index]: fold
        for fold, (_, test) in enumerate(linguistic_folds)
        for index in test
    }
    assert linguistic_assignment == {
        video_id: full_assignment[video_id] for video_id in linguistic
    }


def test_negative_out_of_fold_r2_is_preserved() -> None:
    targets = np.arange(20, dtype=float)[:, None]
    embeddings = np.zeros((20, 2), dtype=float)
    folds = list(
        build_inherited_folds(
            np.asarray([f"v{i}" for i in range(20)]),
            np.asarray([f"v{i}" for i in range(10)]),
            n_splits=2,
            seed=1,
        )[0]
    )
    scores = score_targets(
        embeddings, targets, folds, alphas=np.asarray([1.0, 10.0])
    )
    assert scores.shape == (1,)
    assert scores[0] < 0


def test_probe_rejects_changed_shared_values() -> None:
    full, linguistic, representations = _inputs()
    changed = linguistic.values.copy()
    changed[0, 0] += 1
    invalid = FeatureTable(
        linguistic.video_ids, linguistic.feature_names, changed
    )
    with pytest.raises(ValueError, match="shared feature values differ"):
        probe_representations(
            full,
            invalid,
            representations,
            _vocabulary(),
            n_splits=3,
            alphas=np.asarray([1.0]),
        )


def test_v1_vocabulary_runs_from_generated_tables_and_npz(tmp_path) -> None:
    """Exercise the complete loader/alignment/scoring/writer path without data."""
    rng = np.random.default_rng(120)
    vocabulary = load_vocabulary()
    full_names = vocabulary.loc[
        vocabulary["source_cohort"] == "full", "feature"
    ].tolist()
    linguistic_names = vocabulary.loc[
        vocabulary["source_cohort"] == "linguistic", "feature"
    ].tolist()
    ids = np.asarray([f"synthetic-{index:02d}" for index in range(15)])
    full_values = rng.normal(size=(len(ids), len(full_names)))
    linguistic_indices = np.arange(10)
    linguistic_values = rng.normal(size=(10, len(linguistic_names)))
    full_frame = pd.DataFrame(full_values, columns=full_names)
    full_frame.insert(0, "video_id", ids)
    linguistic_frame = pd.DataFrame(
        np.column_stack(
            [full_values[linguistic_indices], linguistic_values]
        ),
        columns=full_names + linguistic_names,
    )
    linguistic_frame.insert(0, "video_id", ids[linguistic_indices])
    full_path = tmp_path / "full.csv"
    linguistic_path = tmp_path / "linguistic.csv"
    full_frame.to_csv(full_path, index=False)
    linguistic_frame.to_csv(linguistic_path, index=False)
    representation_path = tmp_path / "representation.npz"
    np.savez(
        representation_path,
        video_ids=ids[::-1],
        embeddings=rng.normal(size=(len(ids), 3)).astype(np.float32),
    )

    scores = run_probe(
        full_path,
        linguistic_path,
        [f"synthetic={representation_path}"],
        tmp_path / "output",
        n_splits=3,
        alphas=np.asarray([1.0]),
    )
    assert len(scores) == 133
    assert scores["feature"].is_unique
    assert set(scores["n_videos"]) == {15, 10}
    assert (tmp_path / "output" / "coverage_scores_long.csv").is_file()
    assert (tmp_path / "output" / "probe_settings.json").is_file()
