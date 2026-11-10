from __future__ import annotations

import pandas as pd
import pytest

from mavic.vocabulary import (
    MAVIC_V1_COUNTS,
    MAVIC_V1_TOTAL,
    load_vocabulary,
    validate_vocabulary,
)


def test_packaged_vocabulary_is_authoritative_mavic_v1() -> None:
    vocabulary = load_vocabulary()
    assert len(vocabulary) == MAVIC_V1_TOTAL == 133
    assert vocabulary["feature_type"].value_counts().to_dict() == {
        "visual": MAVIC_V1_COUNTS["visual"],
        "linguistic": MAVIC_V1_COUNTS["linguistic"],
        "audio": MAVIC_V1_COUNTS["audio"],
    }
    assert vocabulary["feature"].is_unique
    assert vocabulary["display_name"].is_unique
    assert set(vocabulary.loc[vocabulary.source_cohort == "full", "feature_type"]) == {
        "audio",
        "visual",
    }


def test_vocabulary_rejects_duplicate_names() -> None:
    frame = pd.DataFrame(
        [
            {
                "feature": "x",
                "display_name": "X",
                "feature_type": "visual",
                "is_variability": False,
                "source_cohort": "full",
                "vocabulary_version": "test",
            },
            {
                "feature": "x",
                "display_name": "Y",
                "feature_type": "visual",
                "is_variability": False,
                "source_cohort": "full",
                "vocabulary_version": "test",
            },
        ]
    )
    with pytest.raises(ValueError, match="duplicate feature"):
        validate_vocabulary(frame, require_mavic_v1=False)


def test_vocabulary_rejects_modality_cohort_mismatch() -> None:
    frame = pd.DataFrame(
        [
            {
                "feature": "words",
                "display_name": "Words",
                "feature_type": "linguistic",
                "is_variability": False,
                "source_cohort": "full",
                "vocabulary_version": "test",
            }
        ]
    )
    with pytest.raises(ValueError, match="full features"):
        validate_vocabulary(frame, require_mavic_v1=False)
