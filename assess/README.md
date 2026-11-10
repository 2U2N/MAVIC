# MAVIC assessment toolkit

MAVIC is an interpretable vocabulary and assessment pipeline for studying which
formal properties of videos are recoverable from computational
representations. This folder contains the final MAVIC v1
probe vocabulary, deterministic feature-selection and linear-probing tools,
optional reference embedding pipelines, tests, and the aggregate materials
needed to regenerate the manuscript figures.

## What is included

- an authoritative 133-probe vocabulary: 86 visual, 13 audio, and 34
  linguistic features;
- reusable commands for selecting a nonredundant vocabulary, probing arbitrary
  video representations, and plotting coverage;
- optional OpenCLIP, LCO, and F2LLM reference embedding implementations;
- deterministic synthetic tests that exercise the complete reusable workflow;
- identifier-free aggregate paper results, analysis scripts, tables, and final
  figures.

Raw videos, row-level MAVIC measurements, embeddings, donor metadata, video
identifiers, manuscripts, and notebooks are not distributed. Consequently,
this folder supports code audit, synthetic end-to-end verification, and
aggregate-figure reproduction. Complete numerical regeneration requires the
restricted research inputs.

## Installation

MAVIC supports Python 3.10–3.12. The reusable selection, probing, and plotting
tools have no deep-learning dependency:

```bash
python -m pip install -e '.[test,analysis]'
pytest -m "not integration and not model and not lco_model"
```

`requirements/py312.lock` records the Python 3.12.3 analysis environment
used for validation.

## Input contract

Feature tables are CSV files with one `video_id` column and numeric candidate
feature columns. The linguistic candidate table contains the transcript-usable
subset and must use identifiers drawn from the full table. The fixed probe
vocabulary is stored in `src/mavic/data/mavic_v1.csv`.

A representation can be either:

- a consolidated NPZ containing aligned `video_ids` and a finite, two-
  dimensional `embeddings` array; or
- a compatible embedding-run directory containing `manifest.json` and the
  referenced shards.

Identifiers are treated as opaque strings and outputs are sorted
deterministically. See [`docs/input_contract.md`](docs/input_contract.md) for
the complete schema.

## Select MAVIC features

```bash
mavic select \
  --full-candidates FULL.csv \
  --linguistic-candidates LINGUISTIC.csv \
  --config config/mavic-v1-selection.yaml \
  --output-dir OUTPUT
```

Selection applies the 30-second duration filter, completeness and finiteness
checks, the ten-distinct-value threshold, deterministic preference for standard
deviation over volatility alternatives, and greedy correlation rejection at
absolute Pearson correlation greater than `.80`. Audio/visual selections from
the full cohort are reused unchanged in the linguistic cohort.

## Probe video representations

```bash
mavic probe \
  --full-features FULL.csv \
  --linguistic-features LINGUISTIC.csv \
  --representation OpenCLIP=/path/to/openclip.npz \
  --representation LCO=/path/to/lco-run \
  --representation F2LLM=/path/to/f2llm.npz \
  --output-dir OUTPUT
```

One or more arbitrary representations may be supplied. MAVIC intersects their
available identifiers, constructs shared shuffled five-fold assignments, and
fits a separate ridge probe for every feature. Embedding dimensions are
standardized from each training split only; `RidgeCV` chooses a target-specific
penalty from the paper grid. Linguistic videos inherit their full-cohort fold.
The long-form output `coverage_scores_long.csv` contains:

```text
feature, display_name, feature_type, is_variability, source_cohort,
n_videos, representation, r2
```

Negative held-out R² values are retained.

## Plot generic coverage

```bash
mavic plot coverage \
  --scores OUTPUT/coverage_scores_long.csv \
  --output-dir OUTPUT/figures
```

The plotting command discovers representation names dynamically. It does not
require or reload embeddings.

## Reference embedding pipelines

The reference implementations are optional. The default test suite exercises
their control flow with mocks; installing the large model-specific extras is
not required:

```bash
python -m pip install -e '.[openclip]'
mavic-openclip-embed INPUT_DIR OUTPUT_DIR

python -m pip install -e '.[lco]'
mavic-lco-embed INPUT_DIR OUTPUT_DIR

python -m pip install -e '.[f2llm]'
mavic-f2llm-embed INPUT_DIR OUTPUT_DIR
```

Each command retains resumable shards, manifests, validation, and a
`consolidate` subcommand. Model weights are downloaded from their providers and
are not redistributed. Exact model revisions, prompts, dimensions, pooling,
and normalization are recorded in
`config/reference-representations.yaml`. Real model execution requires suitable
hardware and is outside the automated checks.

## Reproduce paper figures and tables

The paper layer reads only aggregate, identifier-free tables:

```bash
python -m paper.analysis.reproduce_figures
```

Verified submission outputs live in `paper/figures/` and the released appendix
table lives in `paper/results/`. Regenerated copies are written below
`paper/generated/` so they do not overwrite the verified artifacts. Exact
settings are in `paper/configs/`; the retained wide
coverage table provides traceability to the three manuscript representations,
while the reusable `mavic probe` interface emits dynamic long-form data.
Detailed commands and limitations are in
[`docs/reproduction.md`](docs/reproduction.md).

## Upstream feature extraction

MAVIC begins from time-aggregated audio, visual, and linguistic measurements.
The extractor in [`../extract`](../extract) produces them from a folder of
videos. Its `full_features.csv` and `linguistic_features.csv` exports are the
`--full-features` and `--linguistic-features` inputs of `mavic probe`; its
`candidates.csv` export holds the candidate measurements that `mavic select`
starts from.
