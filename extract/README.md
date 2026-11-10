# MAVIC feature extractor

Measure visual appearance, motion, sound and language in a folder of videos. Notebook 00 generates synthetic test clips locally. One moving pattern has a pure tone and the four-second gray clip has no audio stream. Use a video from your own collection to try speech transcription.

## Install and open the notebooks

Install Python 3.11 and FFmpeg (including ffprobe) using your platform's installer or package manager. From the repository directory on macOS or Linux, run:

``` sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install "pip==25.3"
python -m pip install --build-constraint build-constraints.txt -e .
python -m jupyter lab
```

On Windows create the environment with `py -3.11 -m venv .venv` and activate it with `.venv\Scripts\activate` in Command Prompt. The utilized version of Whisper needs a setuptools release that includes pkg_resources when building its package. The build constraint supplies it.

1.  Open **notebooks/00_start_here.ipynb**. Leave the example settings in place for the first run to generate local clips.
2.  Open **01_visual_and_audio**. The three CPU stages save their work separately. The appearance stage obtains detector files on its first run. See [model setup](MODELS.md).
3.  Open **02_language_and_transcription** for linguistic analysis. Whisper large screens the opening 30 seconds and Whisper turbo transcribes qualifying files in the original language. Choose CPU or a configured GPU in the settings cell.
4.  Open **03_text_and_export**. Its editable list contains the 24 study language models. On the first run, installation needs internet access and disk space. Comment out languages you do not need. To export only audiovisual results, follow the skip instructions in the notebook.

Reopen the same project to resume. Successful per-video results are saved atomically. Failures appear in the stage summary and are retried on the next run. Rerun notebook 00 after editing the video collection. Each video file stem must be unique.

## Read the results

Under the project's exports directory, candidates.csv contains 271 measurements. Empty cells are explained in missing.csv. The feature dictionary identifies modality, producer, units, sampling, aggregation, minimum observations, and vocabulary membership. Counts of frames, audio blocks and flow samples are in the per-video records.

The two fixed tables contain 99 audiovisual or 133 combined features plus video_id. The combined rows reuse the full table's audiovisual values. The original study filtered by duration. This extractor attempts measurements for short videos and exports each row wherever all required values exist.