from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mavic.selection import (
    correlation_order,
    reject_correlated,
    run_selection,
    load_selection_config,
    sd_counterpart,
)


def test_shipped_selection_configuration_loads() -> None:
    config = load_selection_config(
        Path(__file__).parents[1] / "config" / "mavic-v1-selection.yaml"
    )
    assert config["minimum_duration_seconds"] == 30
    assert config["minimum_distinct_values"] == 10
    assert config["maximum_absolute_correlation"] == 0.8


def test_sd_counterparts_and_priority_order() -> None:
    assert sd_counterpart("volatility_energy") == "std_energy"
    assert sd_counterpart("motion_vol") == "motion_sd"
    assert correlation_order(
        ["ordinary", "std_audio", "duration_seconds", "visual_sd"]
    ) == ["duration_seconds", "std_audio", "visual_sd", "ordinary"]


def test_rejection_is_greedy_and_respects_protected_features() -> None:
    x = np.arange(20, dtype=float)
    frame = pd.DataFrame(
        {
            "canonical": x,
            "addition_copy": x,
            "addition_other": np.asarray(
                [0, 1, 0, 2, 1, 3, 0, 4, 1, 2, 4, 0, 3, 1, 4, 2, 0, 3, 2, 4],
                dtype=float,
            ),
        }
    )
    selected, audit = reject_correlated(
        frame,
        threshold=0.8,
        selection_stage="linguistic_additions",
        protected_features=["canonical"],
    )
    assert list(selected) == ["canonical", "addition_other"]
    assert set(audit["dropped_feature"]) == {"addition_copy"}


def test_complete_two_cohort_selection(tmp_path) -> None:
    rng = np.random.default_rng(72)
    count = 24
    ids = np.asarray([f"v{index:02d}" for index in range(count)])
    duration = 35 + rng.permutation(np.arange(count))
    duration[0] = 20
    sd_signal = rng.normal(size=count)
    ordinary = rng.normal(size=count)
    linguistic_values = rng.normal(size=count)
    full = pd.DataFrame(
        {
            "video_id": ids,
            "duration_seconds": duration,
            "std_energy": sd_signal,
            "volatility_energy": sd_signal,
            "ordinary": ordinary,
            "ordinary_copy": ordinary,
            "transcript_measure": np.nan,
            "constant": 1,
        }
    )
    linguistic_rows = np.arange(1, 20)
    linguistic = full.iloc[linguistic_rows].copy()
    linguistic["transcript_measure"] = linguistic_values[linguistic_rows]
    full_path = tmp_path / "full.csv"
    linguistic_path = tmp_path / "ling.csv"
    full.to_csv(full_path, index=False)
    linguistic.to_csv(linguistic_path, index=False)
    config = {
        "duration_column": "duration_seconds",
        "minimum_duration_seconds": 30,
        "minimum_distinct_values": 3,
        "maximum_absolute_correlation": 0.8,
        "exclude_features": [],
        "exclude_name_contains": [],
    }
    config_path = tmp_path / "selection.yaml"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    outputs = run_selection(full_path, linguistic_path, config_path, tmp_path / "out")
    full_selected = pd.read_csv(outputs["full_features"])
    linguistic_selected = pd.read_csv(outputs["linguistic_features"])
    preference = pd.read_csv(outputs["preference_audit"])
    quality = pd.read_csv(outputs["quality_audit"])

    assert "std_energy" in full_selected
    assert "volatility_energy" not in full_selected
    assert "ordinary_copy" not in full_selected
    assert "transcript_measure" not in full_selected
    assert list(linguistic_selected.columns[: len(full_selected.columns)]) == list(
        full_selected.columns
    )
    assert "transcript_measure" in linguistic_selected
    assert set(preference["removed_feature"]) == {"volatility_energy"}
    constant = quality.loc[
        (quality["cohort"] == "full") & (quality["feature"] == "constant")
    ].iloc[0]
    assert constant["reason"] == "low_uniqueness"
    assert len(full_selected) == count - 1


def test_selection_rejects_changed_shared_measurements(tmp_path) -> None:
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(
        {
            "video_id": [f"v{i}" for i in range(15)],
            "duration_seconds": np.arange(40, 55),
            "visual": rng.normal(size=15),
            "ling": np.nan,
        }
    )
    linguistic = frame.iloc[:12].copy()
    linguistic["ling"] = rng.normal(size=12)
    linguistic.loc[0, "visual"] += 10
    full_path = tmp_path / "full.csv"
    ling_path = tmp_path / "ling.csv"
    frame.to_csv(full_path, index=False)
    linguistic.to_csv(ling_path, index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        json.dumps(
            {
                "minimum_distinct_values": 3,
                "maximum_absolute_correlation": 0.99,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="differs"):
        run_selection(full_path, ling_path, config_path, tmp_path / "out")
