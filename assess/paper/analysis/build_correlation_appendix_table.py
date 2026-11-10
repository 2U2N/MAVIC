"""Build the Word-compatible appendix table from ID-free correlation audits."""

from __future__ import annotations

import argparse
import csv
from html import escape
from io import StringIO
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from .common import GENERATED_DIR, PAPER_RESULTS_DIR
except ImportError:  # Allow direct execution from a source checkout.
    from common import GENERATED_DIR, PAPER_RESULTS_DIR  # type: ignore


DEFAULT_FULL_AUDIT = PAPER_RESULTS_DIR / "correlation_audit_full-dataset.csv"
DEFAULT_LINGUISTIC_AUDIT = (
    PAPER_RESULTS_DIR / "correlation_audit_linguistic-dataset.csv"
)
DEFAULT_OUTPUT_DIR = GENERATED_DIR / "tables"
DEFAULT_THRESHOLD = 0.8

FULL_STAGE = "canonical_full_dataset"
LINGUISTIC_STAGE = "linguistic_additions"
STAGE_SPECS = (
    (FULL_STAGE, "Audio/visual feature selection", 0),
    (LINGUISTIC_STAGE, "Linguistic additions", 1),
)
STAGE_LABELS = {stage: label for stage, label, _ in STAGE_SPECS}
STAGE_ORDER = {stage: order for stage, _, order in STAGE_SPECS}

AUDIT_COLUMNS = (
    "selection_stage",
    "n_rows",
    "kept_feature",
    "kept_feature_display",
    "dropped_feature",
    "dropped_feature_display",
    "absolute_correlation",
)
STRING_COLUMNS = (
    "selection_stage",
    "kept_feature",
    "kept_feature_display",
    "dropped_feature",
    "dropped_feature_display",
)
TABLE_TITLE = (
    "Strongly correlated MAVIC feature groups retained and removed during "
    "probe selection"
)


def _load_audit(
    path: Path, *, expected_stage: str, threshold: float
) -> pd.DataFrame:
    """Load one aggregate audit without requiring restricted feature rows."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"correlation audit not found: {path}")
    try:
        audit = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        raise ValueError(f"could not read correlation audit {path}: {error}") from error
    missing = sorted(set(AUDIT_COLUMNS) - set(audit.columns))
    if missing:
        raise ValueError(f"{path}: missing columns: " + ", ".join(missing))
    if audit.empty:
        raise ValueError(f"{path}: audit contains no relationships")
    audit = audit.loc[:, AUDIT_COLUMNS].copy()

    for column in STRING_COLUMNS:
        values = audit[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"{path}: {column} must contain non-empty strings")
        audit[column] = values.astype(str).str.strip()
    stages = audit["selection_stage"].unique().tolist()
    if stages != [expected_stage]:
        raise ValueError(
            f"{path}: expected selection_stage {expected_stage!r}, found {stages}"
        )

    for column in ("n_rows", "absolute_correlation"):
        try:
            audit[column] = pd.to_numeric(audit[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}: {column} must be numeric") from error
        if not np.isfinite(audit[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{path}: {column} must be finite")
    row_counts = audit["n_rows"].to_numpy(dtype=float)
    if (
        audit["n_rows"].nunique() != 1
        or (row_counts <= 0).any()
        or not np.equal(row_counts, np.floor(row_counts)).all()
    ):
        raise ValueError(f"{path}: n_rows must be one positive integer")
    audit["n_rows"] = row_counts.astype(np.int64)
    correlations = audit["absolute_correlation"].to_numpy(dtype=float)
    if (correlations <= threshold).any() or (correlations > 1.0).any():
        raise ValueError(
            f"{path}: absolute correlations must be > {threshold:g} and <= 1"
        )
    if audit.duplicated(["kept_feature", "dropped_feature"]).any():
        raise ValueError(f"{path}: duplicate correlation pairs")
    if (audit["kept_feature"] == audit["dropped_feature"]).any():
        raise ValueError(f"{path}: a feature cannot be retained and dropped in one pair")
    return audit


def load_relationships(
    full_audit_path: Path = DEFAULT_FULL_AUDIT,
    linguistic_audit_path: Path = DEFAULT_LINGUISTIC_AUDIT,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Load and combine the two stage-specific, ID-free audits."""
    if not np.isfinite(threshold) or not 0.0 <= threshold < 1.0:
        raise ValueError("threshold must be finite and between 0 and 1")
    full = _load_audit(
        full_audit_path, expected_stage=FULL_STAGE, threshold=threshold
    )
    linguistic = _load_audit(
        linguistic_audit_path,
        expected_stage=LINGUISTIC_STAGE,
        threshold=threshold,
    )
    relationships = pd.concat([full, linguistic], ignore_index=True)
    if relationships.duplicated(
        ["selection_stage", "kept_feature", "dropped_feature"]
    ).any():
        raise ValueError("combined audits contain duplicate relationship pairs")
    relationships["stage_order"] = relationships["selection_stage"].map(STAGE_ORDER)
    relationships["section_label"] = relationships["selection_stage"].map(
        STAGE_LABELS
    )
    return relationships


def group_relationships(relationships: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic retained-feature groups for the publication table."""
    ordered = relationships.sort_values(
        [
            "stage_order",
            "kept_feature_display",
            "kept_feature",
            "dropped_feature_display",
            "dropped_feature",
        ],
        kind="mergesort",
    )
    return (
        ordered.groupby(
            [
                "stage_order",
                "selection_stage",
                "section_label",
                "n_rows",
                "kept_feature",
                "kept_feature_display",
            ],
            sort=False,
            as_index=False,
        )
        .agg(
            dropped_features=(
                "dropped_feature_display",
                lambda values: tuple(dict.fromkeys(values)),
            )
        )
        .reset_index(drop=True)
    )


def build_note(grouped: pd.DataFrame, *, threshold: float) -> str:
    counts = {
        row.selection_stage: int(row.n_rows)
        for row in grouped[["selection_stage", "n_rows"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    threshold_text = f"{threshold:.2f}".lstrip("0")
    return (
        "Note. Each row lists a retained feature and every rejected feature "
        f"with which it had an absolute Pearson correlation greater than {threshold_text}. "
        f"Audio/visual correlations were evaluated on {counts[FULL_STAGE]:,} videos; "
        f"linguistic additions were evaluated on {counts[LINGUISTIC_STAGE]:,} videos. "
        "The deterministic selection order prioritized duration and variability "
        "measures, and the canonical audio/visual vocabulary was retained when "
        "linguistic additions were selected. A rejected feature may appear in "
        "more than one row when it correlates strongly with multiple retained features."
    )


def render_html(grouped: pd.DataFrame, *, note: str) -> str:
    """Render a standalone HTML table that can be copied into Word."""
    rows: list[str] = []
    for stage, _, _ in STAGE_SPECS:
        stage_rows = grouped.loc[grouped["selection_stage"].eq(stage)]
        if stage_rows.empty:
            continue
        n_rows = int(stage_rows["n_rows"].iloc[0])
        section = f"{STAGE_LABELS[stage]} (n = {n_rows:,})"
        rows.append(
            '<tr class="section"><th colspan="2" scope="rowgroup">'
            f"{escape(section)}</th></tr>"
        )
        for row in stage_rows.itertuples(index=False):
            bullets = "".join(
                f"<li>{escape(str(feature))}</li>" for feature in row.dropped_features
            )
            rows.append(
                "<tr>"
                f'<td class="kept">{escape(str(row.kept_feature_display))}</td>'
                f'<td class="dropped"><ul>{bullets}</ul></td>'
                "</tr>"
            )

    table_rows = "\n".join(rows)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{escape(TABLE_TITLE)}</title>
<style>
@page {{ size: A4; margin: 1.8cm; }}
body {{ font-family: "Times New Roman", Times, serif; font-size: 10pt; color: #111; }}
.table-title {{ font-size: 11pt; font-weight: bold; margin: 0 0 8pt 0; }}
table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
thead {{ display: table-header-group; }}
tr {{ page-break-inside: avoid; }}
th, td {{ border: 1px solid #555; padding: 5pt 6pt; vertical-align: top; }}
thead th {{ background: #d9e2f3; text-align: left; font-weight: bold; }}
.section th {{ background: #eaf0f8; text-align: left; font-weight: bold; }}
.kept {{ width: 37%; font-weight: bold; }}
.dropped {{ width: 63%; }}
ul {{ margin: 0; padding-left: 16pt; }}
li {{ margin: 0 0 2pt 0; }}
.note {{ margin-top: 8pt; font-size: 9pt; line-height: 1.2; }}
</style>
</head>
<body>
<p class="table-title">{escape(TABLE_TITLE)}</p>
<table>
<colgroup><col style="width:37%"><col style="width:63%"></colgroup>
<thead><tr><th scope="col">Kept feature</th><th scope="col">Dropped features</th></tr></thead>
<tbody>
{table_rows}
</tbody>
</table>
<p class="note">{escape(note)}</p>
</body>
</html>
"""


def render_tsv(grouped: pd.DataFrame) -> str:
    output = StringIO(newline="")
    writer = csv.writer(output, delimiter="\t", lineterminator="\r\n")
    writer.writerow(["Kept feature", "Dropped features"])
    for stage, _, _ in STAGE_SPECS:
        stage_rows = grouped.loc[grouped["selection_stage"].eq(stage)]
        if stage_rows.empty:
            continue
        n_rows = int(stage_rows["n_rows"].iloc[0])
        writer.writerow([f"{STAGE_LABELS[stage]} (n = {n_rows:,})", ""])
        for row in stage_rows.itertuples(index=False):
            writer.writerow(
                [row.kept_feature_display, "; ".join(row.dropped_features)]
            )
    return "\ufeff" + output.getvalue()


def run_generation(
    full_audit_path: Path = DEFAULT_FULL_AUDIT,
    linguistic_audit_path: Path = DEFAULT_LINGUISTIC_AUDIT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[pd.DataFrame, list[Path]]:
    """Validate the aggregate audits and write Word-compatible table files."""
    relationships = load_relationships(
        full_audit_path, linguistic_audit_path, threshold=threshold
    )
    grouped = group_relationships(relationships)
    note = build_note(grouped, threshold=threshold)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "correlation_appendix_table.html",
        output_dir / "correlation_appendix_table.tsv",
        output_dir / "correlation_appendix_caption.txt",
    ]
    paths[0].write_text(render_html(grouped, note=note), encoding="utf-8")
    paths[1].write_text(render_tsv(grouped), encoding="utf-8")
    paths[2].write_text(f"{TABLE_TITLE}\n\n{note}\n", encoding="utf-8")
    print(
        f"Built {len(grouped)} retained-feature groups from "
        f"{len(relationships)} relationships."
    )
    return grouped, paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the appendix table from ID-free correlation audits"
    )
    parser.add_argument("--full-audit", type=Path, default=DEFAULT_FULL_AUDIT)
    parser.add_argument(
        "--linguistic-audit", type=Path, default=DEFAULT_LINGUISTIC_AUDIT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        run_generation(
            args.full_audit,
            args.linguistic_audit,
            args.output_dir,
            threshold=args.threshold,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
