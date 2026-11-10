# Model files and third-party terms

MAVIC's source code is MIT licensed. Dependencies and model files retain the terms of their respective providers. External weights are retrieved separately.

Notebook 01 downloads MobileNet SSD configuration and weights from the [MobileNet SSD project](https://github.com/chuanqi305/MobileNet-SSD) and YuNet weights from [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet). See those projects for their distribution terms. The extractor checks each file against the SHA-256 digest listed in src/mavic_extract/assets.py. For offline use set MAVIC_MODELS to the folder containing the three matching files. MediaPipe supplies the Hands model through its Python package.

Notebook 02 uses two checkpoints of [Whisper 20240930](https://github.com/openai/whisper/tree/v20240930): large for language screening and turbo for transcription. Notebook 03 obtains the selected medium [spaCy pipelines](https://github.com/explosion/spacy-models/releases) at 3.8.0. Each pipeline has its own model card, credits and license. Notebook reproducibility uses TextDescriptives 2.8.4 and spaCy 3.8.4.

The upstream Caffe configuration has a final newline. The downloader removes trailing line endings before comparing it to the study copy. The two weight files are checked byte for byte.
