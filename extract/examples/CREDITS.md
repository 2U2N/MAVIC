# Validation media provenance

The synthetic patterns, tones, silence and deliberately truncated clips used for validation were generated with FFmpeg. Notebook 00 generates a moving pattern with a tone and a short gray clip without audio on the user’s computer.

Private speech validation used the Whisper project’s JFK fixture and a LibriSpeech recording of *The Ashiel Mystery*, chapter 2, narrated by Garth Comira. The narrated test clip combined generated video with repeated English narration and AAC encoding. The source narration is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). [Recording credits](https://librosa.org/data/audio/5703-47212-0000.txt) · [LibriSpeech collection](https://www.openslr.org/12/) · [Whisper speech fixture](https://github.com/openai/whisper/blob/main/tests/jfk.flac).

These recordings and adapted clips are kept outside the public repository. The notebooks do not download speech examples automatically.
