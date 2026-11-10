# Paper reproduction layer

This directory contains:

- `analysis/` contains the submitted paper's analysis and plotting code.
- `results/` contains aggregate, identifier-free score tables and correlation
  audits.
- `figures/` contains the supplied manuscript figures.
- `configs/` records the exact feature-selection, probing, representation, and
  figure settings.

The released aggregate tables are sufficient to regenerate every supplied
figure and the correlation appendix table:

```bash
python -m paper.analysis.reproduce_figures
```

By default, regenerated files are written below `paper/generated/`, leaving
the verified submission artifacts unchanged.

The donor-level analysis code is included for inspection and synthetic tests,
but complete numerical regeneration is not possible from the
distributed files. It requires restricted watch-event data, row-level MAVIC feature
tables, video representations, and source videos, none of which are
distributed. The release contains no participant IDs, video IDs, donor
metadata, embeddings, or row-level measurements.

The wide `coverage_scores.csv` is retained to trace the submitted figures.
`coverage_scores_long.csv` is the equivalent tidy representation used by the
reusable interface: one row per feature and representation.
