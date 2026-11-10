# MAVIC feature extractor

Measure visual appearance, motion, sound and language in a folder of videos. Notebook 00 generates synthetic test clips locally. One moving pattern has a pure tone and the four-second gray clip has no audio stream. Use a video from your own collection to try speech transcription.

## Install and open the notebooks

Install Python 3.11 and FFmpeg (including ffprobe) using your platform's installer or package manager. On macOS with Homebrew, use `brew install python@3.11 ffmpeg`. Check that `ffmpeg -version` and `ffprobe -version` work in your terminal. From the repository directory on macOS or Linux, run:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install "pip==25.3"
python -m pip install --build-constraint build-constraints.txt -e .
python -m jupyter lab
```

On Windows create the environment with `py -3.11 -m venv .venv` and activate it with `.venv\Scripts\activate` in Command Prompt. Historical Whisper needs a setuptools release that includes pkg_resources when building its package. The build constraint supplies it.

1. Open **notebooks/00_start_here.ipynb**. Leave the example settings in place for the first run to generate local clips. Run each cell with Shift+Enter.
2. Open **01_visual_and_audio**. The three CPU stages save their work separately. The appearance stage obtains detector files on its first run. See [model setup](MODELS.md).
3. Open **02_language_and_transcription** for linguistic analysis. Whisper large screens the opening 30 seconds and Whisper turbo transcribes qualifying files in the original language. Choose CPU or a configured GPU in the settings cell.
4. Open **03_text_and_export**. Its editable list contains the 24 study language models. On the first run, installation needs internet access and disk space. Comment out languages you do not need. To export only audiovisual results, follow the skip instructions in the notebook.

Reopen the same project to resume. Successful per-video results are saved atomically. Failures appear in the stage summary and are retried on the next run. Rerun notebook 00 after editing the video collection. Each file stem must be unique. Input clips, checkpoints, transcripts and project results belong outside public version control.

## Read the results

Under the project's exports directory, candidates.csv contains 271 measurements. ancillary.csv contains 39 MFCC summaries. Empty cells are explained in missing.csv. The feature dictionary identifies modality, producer, units, sampling, aggregation, minimum observations, and vocabulary membership. Counts of frames, audio blocks and flow samples are in the per-video records. Motion records include the exact zero-based frame pairs.

The two fixed tables contain 99 audiovisual or 133 combined features plus video_id. The combined rows reuse the full table's audiovisual values. A failed linguistic measurement affects only the combined cohort. See cohorts.csv for exclusions. Use the fixed tables directly with the assessment toolkit's probing interface. The original study filtered by duration. This extractor attempts measurements for short videos and exports each row wherever all required values exist.

## Measurement conventions

Appearance samples the middle of each complete ten-second interval, or the middle of a shorter video. Frames are ordered numerically. Warmth counts OpenCV hue below 30 or above 110. Colorfulness intentionally preserves unsigned 8-bit channel arithmetic, and grayscale entropy preserves the original BGR input to rgb2gray. The largest-confidence feature uses the maximum detection score. People and face boxes use the study models and resizing. Hands use the original static-image landmark settings.

Motion compares the first frame with the sample near five seconds and then successive samples ten seconds apart. Direction uses the original within-frame arithmetic average of angles and across-sample circular summaries. Information entropy uses the histogram of flow magnitudes. Audio uses mono 22,050 Hz samples, complete ten-second blocks or a single shorter block, and band ratios from spectral magnitudes. Population means require one observation, dispersion measures require two, and dispersion of first differences requires three. The acoustic implementation also leaves volatility undefined for an series of zero differences as in the study code.

Speech eligibility uses strictly greater than .98 language confidence. Unsupported languages remain eligible for audiovisual export. Norwegian no maps to the Bokmål nb pipeline. Words per minute counts spaCy tokens excluding whitespace and punctuation and divides by full file duration in minutes. This operational definition is specified for this release.

The locally generated patterns and tone use FFmpeg. See [validation media provenance](examples/CREDITS.md). See ../LICENSE for project code and MODELS.md for third-party resources.

TextDescriptives 2.8.4 excludes apostrophe-containing tokens from n_tokens. The words-per-minute numerator follows the specified punctuation and whitespace rule and is recorded separately as word_count.

The generated example-videos directory is ignored by Git. No media files are distributed with this repository.
