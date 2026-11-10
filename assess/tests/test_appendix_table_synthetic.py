import csv
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

from paper.analysis.build_correlation_appendix_table import (
    AUDIT_COLUMNS,
    FULL_STAGE,
    LINGUISTIC_STAGE,
    group_relationships,
    load_relationships,
    run_generation,
)


def _row(
    stage: str,
    n_rows: int,
    kept: str,
    kept_display: str,
    dropped: str,
    dropped_display: str,
    correlation: float,
) -> dict[str, object]:
    return {
        "selection_stage": stage,
        "n_rows": n_rows,
        "kept_feature": kept,
        "kept_feature_display": kept_display,
        "dropped_feature": dropped,
        "dropped_feature_display": dropped_display,
        "absolute_correlation": correlation,
    }


def _write_audits(tmp_path: Path) -> tuple[Path, Path]:
    full = pd.DataFrame(
        [
            _row(
                FULL_STAGE,
                4,
                "duration_seconds",
                "Duration in Seconds",
                "bandwidth",
                "Mean Frequency Bandwidth",
                0.86,
            ),
            _row(
                FULL_STAGE,
                4,
                "std_centroid",
                "SD of Frequency Centroid",
                "bandwidth",
                "Mean Frequency Bandwidth",
                0.91,
            ),
        ],
        columns=AUDIT_COLUMNS,
    )
    linguistic = pd.DataFrame(
        [
            _row(
                LINGUISTIC_STAGE,
                3,
                "alpha_ratio",
                "Ratio of Words With >0 Alphabetic Chars.",
                "punctuation",
                "Proportion of Punctuation",
                0.97,
            )
        ],
        columns=AUDIT_COLUMNS,
    )
    full_path = tmp_path / "full.csv"
    linguistic_path = tmp_path / "linguistic.csv"
    full.to_csv(full_path, index=False)
    linguistic.to_csv(linguistic_path, index=False)
    return full_path, linguistic_path


def test_synthetic_table_preserves_repeated_relationships_and_word_formats(
    tmp_path: Path,
) -> None:
    full, linguistic = _write_audits(tmp_path)
    relationships = load_relationships(full, linguistic)
    grouped = group_relationships(relationships)
    assert len(relationships) == 3
    assert len(grouped) == 3
    assert relationships.loc[
        relationships["dropped_feature"].eq("bandwidth"), "kept_feature"
    ].nunique() == 2

    output_dir = tmp_path / "output"
    _, paths = run_generation(full, linguistic, output_dir)
    assert {path.name for path in paths} == {
        "correlation_appendix_table.html",
        "correlation_appendix_table.tsv",
        "correlation_appendix_caption.txt",
    }
    html = (output_dir / "correlation_appendix_table.html").read_text()
    assert "<ul>" in html
    assert "Ratio of Words With &gt;0 Alphabetic Chars." in html
    assert html.count("Mean Frequency Bandwidth</li>") == 2

    tsv_path = output_dir / "correlation_appendix_table.tsv"
    assert tsv_path.read_bytes().startswith(b"\xef\xbb\xbf")
    rows = list(
        csv.reader(
            StringIO(tsv_path.read_text(encoding="utf-8-sig")),
            delimiter="\t",
        )
    )
    assert all(len(row) == 2 for row in rows)


def test_real_aggregate_audits_reproduce_verified_counts() -> None:
    relationships = load_relationships()
    grouped = group_relationships(relationships)
    assert len(relationships) == 80
    assert relationships["dropped_feature"].nunique() == 66
    assert len(grouped) == 41
    assert grouped["selection_stage"].value_counts().to_dict() == {
        FULL_STAGE: 34,
        LINGUISTIC_STAGE: 7,
    }


def test_invalid_threshold_and_duplicate_pairs_are_rejected(tmp_path: Path) -> None:
    full, linguistic = _write_audits(tmp_path)
    duplicate = pd.read_csv(full)
    pd.concat([duplicate, duplicate.iloc[[0]]], ignore_index=True).to_csv(
        full, index=False
    )
    with pytest.raises(ValueError, match="duplicate correlation pairs"):
        load_relationships(full, linguistic)

    full, linguistic = _write_audits(tmp_path)
    invalid = pd.read_csv(full)
    invalid.loc[0, "absolute_correlation"] = 0.8
    invalid.to_csv(full, index=False)
    with pytest.raises(ValueError, match="absolute correlations"):
        load_relationships(full, linguistic)
