# Reproduction

## Scope

This folder permits three levels of verification:

1. **Unit and synthetic workflow tests.** These generate deterministic
   feature tables and embedding arrays at runtime and exercise selection,
   alignment, probing, and plotting without restricted inputs.
2. **Reference-pipeline inspection.** OpenCLIP, LCO, and F2LLM loaders,
   batching, sharding, resumption, validation, and consolidation are tested with
   mocks. No model weights or GPU are required.
3. **Aggregate paper reproduction.** Identifier-free score tables regenerate
   the supplied manuscript figures and appendix table.

The restricted videos, row-level feature matrices, embeddings, and donor-event
data are not included, so the aggregate numerical estimates cannot be refitted
from the distributed files.

## Validation without restricted inputs

```bash
python -m pip install -e '.[test,analysis]'
pytest -m "not integration and not model and not lco_model"
```

## Figure and table regeneration

From the repository root:

```bash
python -m paper.analysis.reproduce_figures
```

The scripts read from `paper/results/` and write regenerated copies below
`paper/generated/`. They do not access private data or refit embeddings.

## Full-data input sequence

Authorized users with the restricted inputs can run:

```bash
mavic select --full-candidates FULL.csv \
  --linguistic-candidates LINGUISTIC.csv \
  --config config/mavic-v1-selection.yaml --output-dir selected

mavic probe --full-features selected/decorrelated_df_full-dataset.csv \
  --linguistic-features selected/decorrelated_df_linguistic-dataset.csv \
  --representation NAME=/path/to/representation.npz \
  --output-dir coverage
```

Reference embedding configuration is recorded in
`config/reference-representations.yaml`. Real model runs are deliberately not
part of CPU acceptance.
