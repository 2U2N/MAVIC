# Input contract

## Candidate and selected feature tables

Both the full and linguistic CSVs must contain a `video_id` column with
nonempty, unique string values. All measurement columns used by MAVIC must be
numeric. The full table supplies visual and audio candidates. The linguistic
table supplies the transcript-usable subset, including the same selected
audio/visual columns plus linguistic candidates.

The linguistic identifiers must be a subset of the full identifiers. Shared
feature values must agree for shared videos. The selection command rejects
duplicate IDs, nonnumeric measurements, nonfinite values, and incompatible
cohorts with an explicit error.

No identifier convention is assumed: identifiers are opaque strings used only
for alignment. Do not use row order to align datasets.

## Representation inputs

A consolidated representation NPZ must contain:

| Array | Requirement |
| --- | --- |
| `video_ids` | one-dimensional, unique, nonempty strings |
| `embeddings` | finite, two-dimensional numeric array |

The arrays must have the same first dimension. Embedding width may differ
between representations but must be positive and constant within one input.

A compatible run directory contains a `manifest.json` plus referenced NPZ
shards. Shards must collectively satisfy the same requirements and may not
repeat identifiers. Additional diagnostic arrays are ignored by generic
probing.

## Generic score output

`mavic probe` writes `coverage_scores_long.csv`, with one row per
feature–representation pair:

| Field | Meaning |
| --- | --- |
| `feature` | stable machine-readable MAVIC name |
| `display_name` | readable vocabulary label |
| `feature_type` | `audio`, `visual`, or `linguistic` |
| `is_variability` | whether the probe captures temporal variability |
| `source_cohort` | `full` or `linguistic` |
| `n_videos` | aligned videos used for that cohort |
| `representation` | user-supplied representation name |
| `r2` | out-of-fold coefficient of determination |
