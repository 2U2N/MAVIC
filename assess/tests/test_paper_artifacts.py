import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper.analysis.plot_coverage_overview import run_plotting as plot_overview
from paper.analysis.prepare_paper_results import wide_to_long


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "paper" / "results"
FIGURES = ROOT / "paper" / "figures"


def test_released_coverage_tables_are_complete_and_equivalent() -> None:
    wide = pd.read_csv(RESULTS / "coverage_scores.csv")
    long = pd.read_csv(RESULTS / "coverage_scores_long.csv")

    assert len(wide) == 133
    assert wide["feature"].is_unique
    assert wide["feature_type"].value_counts().to_dict() == {
        "visual": 86,
        "linguistic": 34,
        "audio": 13,
    }
    assert wide.groupby("source_dataset")["n_videos"].unique().apply(list).to_dict() == {
        "full": [18519],
        "linguistic": [10331],
    }
    expected_long = wide_to_long(wide)
    pd.testing.assert_frame_equal(long, expected_long, check_dtype=False)
    assert len(long) == 133 * 3
    assert long["representation"].value_counts().to_dict() == {
        "openclip": 133,
        "lco": 133,
        "f2llm": 133,
    }


def test_released_key_coverage_values_match_verified_results() -> None:
    scores = pd.read_csv(RESULTS / "coverage_scores.csv").set_index("feature")
    assert scores.loc["words_per_minute", "lco_r2"] == pytest.approx(
        0.506, abs=0.0006
    )
    assert scores.loc["largest_bb_people_mean", "openclip_r2"] == pytest.approx(
        0.761, abs=0.0006
    )
    assert scores.loc["nb_hands_sd", "f2llm_r2"] == pytest.approx(
        0.358, abs=0.0006
    )


def test_released_audience_scores_have_only_aggregate_results() -> None:
    scores = pd.read_csv(RESULTS / "audience_relevance_scores.csv")
    forbidden = {
        "video_id",
        "participant_id",
        "gender",
        "education",
        "age",
        "demographic_permutation_p",
        "demographic_fdr_q",
        "demographic_fdr_significant",
        "age_partial_cv_r2",
        "gender_partial_cv_r2",
        "education_partial_cv_r2",
    }
    assert forbidden.isdisjoint(scores.columns)
    assert len(scores) == 133
    assert scores["feature"].is_unique
    assert set(scores["demographic_n_donors"]) == {1008, 910}
    assert scores["audience_ols_adjusted_r2"].median() == pytest.approx(
        0.031, abs=0.0006
    )
    assert scores["audience_ols_adjusted_r2"].max() == pytest.approx(
        0.144, abs=0.0006
    )


def test_sanitized_settings_report_only_cohort_totals_and_model_settings() -> None:
    path = RESULTS / "sanitized_analysis_settings.json"
    settings = json.loads(path.read_text(encoding="utf-8"))
    serialized = json.dumps(settings).lower()
    assert settings["vocabulary"]["total_probes"] == 133
    assert settings["video_cohorts"]["full_probing_rows"] == 18519
    assert settings["video_cohorts"]["linguistic_probing_rows"] == 10331
    assert settings["audience_analysis"]["full_cohort_eligible_donors"] == 1008
    assert settings["audience_analysis"]["linguistic_cohort_eligible_donors"] == 910
    assert "gender_counts" not in serialized
    assert "education_counts" not in serialized
    assert "participant_id" not in serialized
    assert "video_id" not in serialized


def test_supplied_manuscript_figures_are_present_and_nonempty() -> None:
    stems = [
        "coverage_profiles_overview",
        "combined_coverage-audio-mean",
        "combined_coverage-visual-mean",
        "combined_coverage-linguistic-mean",
        "audience_relevance_scatter-audience-article",
        "audience_relevance_scatter-coverage-gap-article",
        "audience_relevance_scatter-none-article",
    ]
    for stem in stems:
        png = FIGURES / f"{stem}.png"
        pdf = FIGURES / f"{stem}.pdf"
        assert png.stat().st_size > 1000
        assert pdf.read_bytes().startswith(b"%PDF")


def test_overview_can_be_regenerated_from_aggregate_scores(tmp_path: Path) -> None:
    paths = plot_overview(RESULTS / "coverage_scores.csv", tmp_path)
    assert {path.name for path in paths} == {
        "coverage_profiles_overview.png",
        "coverage_profiles_overview.pdf",
    }
    assert all(path.stat().st_size > 1000 for path in paths)
