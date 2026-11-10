"""Original-language screening and transcription."""
from functools import lru_cache
import subprocess
import numpy as np

SCREENING_MODEL = "large"
TRANSCRIPTION_MODEL = "turbo"

@lru_cache(maxsize=1)
def model(name, device):
    import whisper
    import os
    return whisper.load_model(name, device=device, download_root=os.environ.get("WHISPER_CACHE"))

def eligible(confidence):
    return confidence > 0.98

def screen(path, device):
    import whisper
    raw = subprocess.check_output(["ffmpeg","-loglevel","error","-i",str(path),"-t","30","-vn","-ac","1","-ar","16000","-f","s16le","pipe:1"])
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
    if not len(audio):
        raise ValueError("Empty audio")
    encoder = model(SCREENING_MODEL, device)
    mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio), n_mels=encoder.dims.n_mels)
    _, probabilities = encoder.detect_language(mel.to(encoder.device))
    language = max(probabilities, key=probabilities.get)
    return {"lang_confidence": probabilities[language]}, {"language":language, "eligible":eligible(probabilities[language])}

def transcribe(path, device, screening):
    if not screening["detail"]["eligible"]:
        return {}, {"reason":screening["detail"].get("reason", "Language confidence did not exceed .98")}
    language = screening["detail"]["language"]
    result = model(TRANSCRIPTION_MODEL, device).transcribe(str(path), language=language, task="transcribe", fp16=device.startswith("cuda"))
    transcript = result["text"].strip()
    if not transcript:
        raise ValueError("Whisper returned an empty transcript")
    return {}, {"text":transcript, "language":language}
