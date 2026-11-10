# Representation probing

Representation coverage is operationalized as linear recoverability. For every
MAVIC target and representation, the probe:

1. intersects video identifiers across all supplied representations and the
   relevant MAVIC cohort;
2. sorts identifiers and constructs five shuffled full-cohort folds with seed
   42;
3. assigns linguistic videos to the fold they inherited from the full cohort;
4. standardizes embedding dimensions using each outer training split only;
5. selects a target-specific ridge penalty from
   `{0.1, 1, 10, 100, 1000, 10000, 100000}` with `RidgeCV` on that training
   split;
6. predicts the held-out fold; and
7. calculates R² over the combined out-of-fold predictions.

The same video-to-fold assignment is reused for every representation, making
coverage profiles directly comparable. Negative R² values are retained because
they are informative evidence that a probe generalized worse than the held-out
mean baseline.

Coverage measures access through this standardized linear decoder. It is not a
general ranking of model quality, proof that a representation explicitly
encodes a human concept, or evidence of causal preservation.
