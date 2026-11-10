from functools import lru_cache
from importlib import metadata
import subprocess
import sys

def pipeline_language(code):
    return "nb" if code == "no" else code

def install_models(names):
    for name in names:
        try:
            if metadata.version(name) == "3.8.0":
                continue
        except metadata.PackageNotFoundError:
            pass
        filename = f"{name}-3.8.0-py3-none-any.whl"
        url = f"https://github.com/explosion/spacy-models/releases/download/{name}-3.8.0/{filename}"
        subprocess.run([sys.executable, "-m", "pip", "install", url], check=True)

@lru_cache(maxsize=1)
def pipeline(name):
    import spacy
    import textdescriptives
    if metadata.version(name) != "3.8.0":
        raise ValueError(f"Expected {name} version 3.8.0")
    nlp = spacy.load(name)
    for group in ("quality","descriptive_stats","readability","pos_proportions","dependency_distance"):
        nlp.add_pipe(f"textdescriptives/{group}")
    return nlp

def measure(transcription, duration, names):
    import textdescriptives
    item = transcription["detail"]
    if not item.get("text"):
        return {}, {"reason":item.get("reason","No transcript")}
    lang = pipeline_language(item["language"])
    name = next((n for n in names if n.startswith(lang + "_")), None)
    if name is None:
        return {}, {"reason":f"No selected pipeline for {lang}"}
    doc = pipeline(name)(item["text"])
    result = textdescriptives.extract_df(doc, include_text=False).iloc[0].to_dict()
    count = sum(not (t.is_space or t.is_punct) for t in doc)
    result["words_per_minute"] = 60 * count / duration if duration > 0 else None
    return result, {"pipeline":name,"word_count":count}
