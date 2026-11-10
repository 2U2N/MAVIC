# Validation

Tested on an Apple M1 Mac in a fresh Python 3.11 environment. The three published detector downloads passed the study checksum checks. Five regression tests passed, including interruption and resume, input changes, confidence exactly at .98, the Norwegian alias, unsupported language, missing values, and chronological sampling of 12 frames.

On the 32-second moving-pattern fixture, all 175 appearance, 69 acoustic (including MFCC), and 13 motion/shot outputs matched the supplied study functions within rtol 1e-8 and atol 1e-9. Motion used frame pairs 0–60, 60–180 and 180–300 at 12 fps. This comparison used generated media. The original private study corpus was unavailable for comparison.

Notebook 00, notebook 01, and notebook 03 completed in Jupyter. The text check selected the English 3.8.0 pipeline. A known seven-word transcript over 30 seconds yielded 14 words/minute. The assessment toolkit loaded and validated the nested 99- and 133-feature exports using real audiovisual measurements and a supplied reference transcript.

Notebook 02 ran the actual Whisper large language detector on two speech fixtures. One scored below .98 and the other qualified. Full transcription of the eligible 25-second example on CPU exceeded the 20-minute validation cell limit twice. Full ASR-to-linguistic export remains to be verified on a suitable GPU before claiming end-to-end release validation. The notebook does not impose that test timeout on interactive users.

The MediaPipe 0.10.21 wheel installed and executed the hand detector on this Mac. Its internal wheel metadata reports an x86_64 tag, which pip check flagged on arm64. Only the English spaCy pipeline was exercised. Independent reader testing remains open. For that test, ask a reader to follow the README in a clean environment and locate an exported row and its cohort eligibility without assistance.

After this validation the transcription checkpoint was changed from Whisper large to Whisper turbo, as in the study. Screening still uses large. Transcription with turbo has not yet been run in this release.
