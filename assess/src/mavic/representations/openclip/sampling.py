from __future__ import annotations

from collections import Counter
from pathlib import Path


def bin_center_timestamps(
    duration_seconds: float, samples: int = 16, start_seconds: float = 0.0
) -> list[float]:
    """Return the center timestamp of each equal-duration bin."""
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if samples <= 0:
        raise ValueError("samples must be positive")
    width = duration_seconds / samples
    return [start_seconds + (index + 0.5) * width for index in range(samples)]


def discover_videos(input_dir: Path) -> list[Path]:
    """Discover MP4 files in one directory in deterministic filename order."""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"input directory does not exist: {input_dir}")
    videos = sorted(
        (path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".mp4"),
        key=lambda path: path.name,
    )
    ids = [path.stem for path in videos]
    duplicates = sorted(video_id for video_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate video IDs: {', '.join(duplicates)}")
    return videos
