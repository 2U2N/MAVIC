from __future__ import annotations

import pandas as pd
import pytest

from mavic.plotting import plot_coverage_scores, validate_long_scores


def _scores() -> pd.DataFrame:
    rows = []
    features = [
        ("audio", "Audio", "audio", False, "full", 20),
        ("visual_sd", "Visual SD", "visual", True, "full", 20),
        ("words", "Words", "linguistic", False, "linguistic", 12),
    ]
    for representation, offset in (("first", 0.0), ("second", 0.1)):
        for index, values in enumerate(features):
            feature, display, kind, variability, cohort, count = values
            rows.append(
                {
                    "feature": feature,
                    "display_name": display,
                    "feature_type": kind,
                    "is_variability": variability,
                    "source_cohort": cohort,
                    "n_videos": count,
                    "representation": representation,
                    "r2": -0.1 + offset + index * 0.2,
                }
            )
    return pd.DataFrame(rows)


def test_dynamic_plotting_writes_png_and_pdf(tmp_path) -> None:
    outputs = plot_coverage_scores(_scores(), tmp_path)
    assert {path.name for path in outputs} == {
        "coverage-first.png",
        "coverage-first.pdf",
        "coverage-second.png",
        "coverage-second.pdf",
    }
    assert all(path.stat().st_size > 0 for path in outputs)


def test_long_scores_require_each_feature_in_each_representation() -> None:
    scores = _scores()
    scores = scores.loc[
        ~((scores["representation"] == "second") & (scores["feature"] == "words"))
    ]
    with pytest.raises(ValueError, match="does not contain every feature"):
        validate_long_scores(scores)
