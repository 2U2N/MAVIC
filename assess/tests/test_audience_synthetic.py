from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper.analysis.analyze_audience_relevance import (
    add_coverage_gap_ranks,
    build_donor_profiles,
    demographic_ols_adjusted_r2,
)
from paper.analysis.plot_audience_relevance import run_plotting


def _events() -> pd.DataFrame:
    rows = []
    demographics = {
        "p1": (21, "woman", "higher"),
        "p2": (44, "man", "vocational"),
        "p3": (67, "other", "higher"),
    }
    for participant, (age, gender, education) in demographics.items():
        for video_index in range(3):
            rows.append(
                {
                    "participant_id": participant,
                    "video_id": f"{participant}-v{video_index}",
                    "age": age,
                    "gender": gender,
                    "education": education,
                }
            )
    return pd.DataFrame(rows)


def _features(events: pd.DataFrame) -> pd.DataFrame:
    values = {
        "p1": [1.0, 3.0, 20.0],
        "p2": [10.0, 14.0, 30.0],
        "p3": [4.0, 8.0, 40.0],
    }
    rows = []
    for event in events.itertuples(index=False):
        index = int(str(event.video_id).rsplit("v", maxsplit=1)[1])
        rows.append(
            {
                "video_id": event.video_id,
                "feature_a": values[event.participant_id][index],
                "feature_b": float(index),
            }
        )
    return pd.DataFrame(rows)


def test_donor_profiles_use_equal_video_weights_and_donor_threshold() -> None:
    events = _events()
    profiles, audit = build_donor_profiles(
        events,
        _features(events),
        ["feature_a", "feature_b"],
        cohort_name="synthetic",
        min_videos=3,
    )
    assert profiles["participant_id"].tolist() == ["p1", "p2", "p3"]
    assert profiles["feature_a"].tolist() == pytest.approx(
        [8.0, 18.0, 52.0 / 3.0]
    )
    assert profiles["n_usable_videos"].tolist() == [3, 3, 3]
    assert audit["eligible_donors"] == 3
    assert audit["eligible_events"] == 9

    reduced_features = _features(events).loc[
        lambda frame: ~frame["video_id"].isin(["p3-v1", "p3-v2"])
    ]
    reduced, _ = build_donor_profiles(
        events,
        reduced_features,
        ["feature_a"],
        cohort_name="synthetic",
        min_videos=3,
    )
    assert reduced["participant_id"].tolist() == ["p1", "p2"]


def test_ols_adjusted_r2_matches_formula_and_can_be_negative() -> None:
    random = np.random.default_rng(4)
    n = 90
    age = np.linspace(18, 75, n)
    demographics = pd.DataFrame(
        {
            "age": age,
            "gender": np.where(np.arange(n) % 2, "woman", "man"),
            "education": np.where(np.arange(n) % 3, "higher", "vocational"),
        }
    )
    design = np.column_stack(
        [
            np.ones(n),
            age,
            demographics["gender"].eq("woman").to_numpy(dtype=float),
            demographics["education"].eq("vocational").to_numpy(dtype=float),
        ]
    )
    random_target = random.normal(size=n)
    orthogonal = random_target - design @ np.linalg.lstsq(
        design, random_target, rcond=None
    )[0]
    scores, audit = demographic_ols_adjusted_r2(
        demographics, np.column_stack([age, orthogonal])
    )
    expected_null = 1.0 - (n - 1) / (n - int(audit["design_rank"]))
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(expected_null)
    assert scores[1] < 0.0


def test_aggregate_audience_plotting_needs_no_donor_rows(tmp_path: Path) -> None:
    rows = []
    types = ("visual", "audio", "linguistic")
    for index in range(133):
        feature_type = types[index % 3]
        source = "linguistic" if feature_type == "linguistic" else "full"
        rows.append(
            {
                "feature": f"feature_{index:03d}",
                "display_name": f"Readable feature {index}",
                "feature_type": feature_type,
                "is_variability": index % 2 == 0,
                "source_dataset": source,
                "demographic_n_donors": 25 if source == "linguistic" else 40,
                "audience_ols_adjusted_r2": -0.02 + index / 1000,
                "openclip_r2": index / 200,
                "lco_r2": index / 250,
                "f2llm_r2": index / 300,
            }
        )
    frame = add_coverage_gap_ranks(pd.DataFrame(rows))
    path = tmp_path / "scores.csv"
    frame.to_csv(path, index=False)
    outputs = run_plotting(
        path,
        tmp_path / "figures",
        label_modes=("none",),
        figure_layouts=("article",),
        label_count=5,
    )
    assert {output.suffix for output in outputs} == {".png", ".pdf"}
    assert all(output.stat().st_size > 1000 for output in outputs)
