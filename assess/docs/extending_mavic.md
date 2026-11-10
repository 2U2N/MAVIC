# Extending MAVIC

The probing interface is representation-neutral. To assess another video
representation:

1. create one embedding vector per video;
2. store the aligned identifiers and embeddings in the NPZ schema described in
   `input_contract.md`;
3. add `--representation NAME=PATH` to `mavic probe`; and
4. pass the resulting long-form score table to `mavic plot coverage`.

No package source changes are required, and one, two, three, or more
representations can be evaluated together. Because the shared identifier
intersection depends on all supplied representations, report the resulting
cohort sizes whenever coverage profiles are compared.

Extending the *feature vocabulary* is a separate operation. Candidate
measurements should have a documented extraction definition, display name,
modality, variability status, and eligible cohort. Run deterministic selection
on the intended corpus rather than manually appending correlated targets to
`mavic_v1.csv`.
