# Deterministic feature selection

MAVIC v1 uses two related cohorts because transcript-derived measurements are
not available for every video.

For the full audio/visual cohort, selection:

1. keeps videos longer than 30 seconds;
2. removes candidate columns containing missing or infinite values;
3. removes candidates with fewer than ten distinct values;
4. removes each volatility alternative when its standard-deviation counterpart
   remains available;
5. orders duration first, then variability measurements, then ordinary
   measurements; and
6. greedily keeps candidates in that order, rejecting a candidate when its
   absolute Pearson correlation with any already retained feature exceeds
   `.80`.

The retained audio/visual list becomes canonical. In the transcript-usable
cohort, those canonical columns are protected and only linguistic additions
compete during correlation rejection. This prevents sample-dependent
correlations from silently replacing an audio/visual feature between cohorts.

Selection writes the selected tables and explicit audit files for data-quality,
SD-over-volatility, and correlation decisions. Rejected volatility alternatives
cannot return later if their preferred SD feature is itself rejected.

The distributed `mavic_v1.csv` is the authoritative final 133-probe vocabulary,
not a catalogue of every candidate measurement considered during development.
