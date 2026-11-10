"""Generate synthetic demonstration media locally with FFmpeg."""
from pathlib import Path
import subprocess


def create_examples(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    recipes = {
        "moving-pattern.mp4": [
            "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=12",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=22050",
            "-t", "32", "-c:v", "libx264", "-c:a", "aac",
        ],
        "short-without-audio.mp4": [
            "-f", "lavfi", "-i", "color=gray:size=160x120:rate=12",
            "-t", "4", "-c:v", "libx264",
        ],
    }
    for name, arguments in recipes.items():
        path = folder / name
        if path.exists():
            continue
        temporary = path.with_name(path.stem + ".partial.mp4")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", *arguments, str(temporary)],
                check=True,
            )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return folder
