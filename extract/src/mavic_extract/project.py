"""Sequential per-video processing with atomic checkpoints."""
import hashlib
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

STAGES = ("appearance", "motion", "acoustic", "language", "transcription", "text")
PARENT = {"transcription": "language", "text": "transcription"}
VERSION = "mavic-extract-1.0.0"

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf8") as f:
        json.dump(data, f, allow_nan=False, ensure_ascii=False)
        temp = f.name
    os.replace(temp, path)

def number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None

def probe(path):
    args = ["ffprobe","-v","error","-show_streams","-show_format","-of","json",str(path)]
    media = json.loads(subprocess.check_output(args, stderr=subprocess.PIPE))
    stream = next((s for s in media["streams"] if s["codec_type"] == "video"), None)
    if stream is None:
        raise ValueError("Missing video stream")
    duration = number(stream.get("duration")) or number(media["format"].get("duration"))
    if duration is None or duration <= 0:
        raise ValueError("Missing positive video duration")
    subprocess.run(["ffmpeg","-loglevel","error","-xerror","-i",str(path),"-an","-frames:v","1","-f","null","-"], check=True, capture_output=True)
    return duration, any(s["codec_type"] == "audio" for s in media["streams"])

class Project:
    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        self.manifest = json.loads((self.folder / "manifest.json").read_text())

    @classmethod
    def inspect(cls, videos, results):
        videos, results = Path(videos).resolve(), Path(results).resolve()
        if not videos.is_dir():
            raise ValueError(f"Video folder does not exist: {videos}")
        rows = []
        ids = set()
        for path in sorted(videos.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {".mp4",".mkv",".mov",".webm",".avi"}:
                continue
            if path.stem in ids:
                raise ValueError(f"Duplicate video ID: {path.stem}")
            ids.add(path.stem)
            row = dict(video_id=path.stem, path=str(path), sha256=digest(path), duration_seconds=None, error="")
            try:
                row["duration_seconds"], row["has_audio"] = probe(path)
            except (ValueError, subprocess.CalledProcessError) as e:
                row["error"] = str(e)
            rows.append(row)
        if not rows:
            raise ValueError(f"No supported videos found in {videos}")
        atomic(results / "manifest.json", rows)
        return cls(results)

    def _file(self, row, stage):
        key = hashlib.sha256(row["video_id"].encode()).hexdigest()
        return self.folder / "records" / key / (stage + ".json")

    def record(self, row, stage):
        file = self._file(row, stage)
        if not file.exists():
            return {}
        item = json.loads(file.read_text())
        if item["input"] != row["sha256"] or item["version"] != VERSION:
            return {}
        if stage in PARENT:
            parent = self.record(row, PARENT[stage])
            if not parent or item["parent"] != parent["signature"]:
                return {}
        return item

    def verify_inputs(self):
        for row in self.manifest:
            path = Path(row["path"])
            if not path.is_file() or digest(path) != row["sha256"]:
                raise ValueError(f"{path.name} changed. Rerun notebook 00 to refresh the manifest.")

    def run(self, stage, *, device="cpu", models=()):
        self.verify_inputs()
        if stage not in STAGES:
            raise ValueError(f"Unknown stage {stage}")
        options = dict(device=device) if stage in ("language","transcription") else {}
        if stage == "text":
            options["models"] = sorted(models)
        summary = []
        for row in self.manifest:
            parent = self.record(row, PARENT[stage]) if stage in PARENT else {}
            file = self._file(row, stage)
            old = self.record(row, stage)
            state = "cached"
            if not (old.get("ok") and not old["detail"].get("errors") and old["options"] == options):
                item = dict(version=VERSION, input=row["sha256"], parent=parent.get("signature"), options=options, values={}, detail={}, ok=False)
                try:
                    if row["error"]:
                        raise ValueError(row["error"])
                    if stage in PARENT and not parent.get("ok"):
                        raise ValueError(f"Complete the {PARENT[stage]} stage first")
                    file.parent.mkdir(parents=True, exist_ok=True)
                    values, detail = self._compute(stage, row, file, device, models, parent)
                    item.update(ok=True, values={k:number(v) for k,v in values.items()}, detail=detail)
                    state = "partial" if detail.get("errors") else ("skipped" if detail.get("reason") else "completed")
                except Exception as e:
                    item["detail"] = {"reason":str(e)}
                    state = "failed"
                item["signature"] = hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest()
                atomic(file,item)
                old = item
            summary.append({"video_id":row["video_id"],"status":state,"reason":old["detail"].get("reason") or "; ".join(old["detail"].get("errors",{}).values())})
        return pd.DataFrame(summary)

    def _compute(self, stage, row, file, device, models, parent):
        path = Path(row["path"])
        duration = row["duration_seconds"]
        if stage in ("acoustic","language") and row.get("has_audio") is False:
            return {}, {"reason":"No audio stream", "eligible":False}
        if stage == "appearance":
            from .appearance import measure
            return measure(path, duration, file.with_name("preview.png"))
        if stage == "acoustic":
            from .acoustic import measure
            return measure(path)
        if stage == "motion":
            from .motion import process_video
            return process_video(path)
        if stage == "language":
            from .speech import screen
            return screen(path, device)
        if stage == "transcription":
            from .speech import transcribe
            return transcribe(path, device, parent)
        from .linguistic import measure
        return measure(parent, duration, models)
