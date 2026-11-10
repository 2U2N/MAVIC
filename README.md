# MAVIC

Code for *Probing Computational Video Representations with MAVIC: A Multimodal
Assessment Vocabulary for Interpretability and Coverage*
([preprint](https://osf.io/preprints/socarxiv/nvzhy)).

MAVIC describes videos with 133 named visual, audio, and linguistic features
and measures how well each one can be predicted from a video representation
such as an embedding. The result is a coverage profile: what a representation
captures, and what it misses.

## Two steps, two folders

| Step | Folder | Input | Output |
| --- | --- | --- | --- |
| 1. Measure the videos | [`extract/`](extract) | A folder of video files | `full_features.csv` (99 audiovisual features) and `linguistic_features.csv` (133 features, videos with usable speech) |
| 2. Assess representations | [`assess/`](assess) | The two tables from step 1 and one or more embeddings of the same videos | Coverage scores per feature and representation, and plots |

The folders install separately because their requirements differ. `extract/`
needs Python 3.11 with pinned media and speech libraries and is run through
four short notebooks. `assess/` is a light command-line package for Python
3.10–3.12. Each folder's README has the installation and usage steps.

`assess/paper/` holds the aggregate results and the scripts that regenerate
the paper's figures and tables.

## What you can reproduce

- Run the extractor on your own videos and assess your own representations.
- Regenerate the paper's figures and tables from the released aggregate scores.
- Run the test suites, which use generated data only.

The donated watch histories, the sampled videos, the per-video measurements,
and the embeddings cannot be shared for reasons of participant privacy and
copyright. The paper's estimates therefore cannot be refitted from this
repository alone.

## Citation and licence

See [`CITATION.cff`](CITATION.cff). The code is released under the
[MIT licence](LICENSE). The non-code material is released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): the aggregate
results in `assess/paper/results`, the figures in `assess/paper/figures`, the
vocabulary table `assess/src/mavic/data/mavic_v1.csv`, and the feature
dictionary `extract/src/mavic_extract/data/registry.csv`. Third-party models
keep their own terms; see [`extract/MODELS.md`](extract/MODELS.md).

Contact: David Wegmann, DATALAB, Aarhus University.
