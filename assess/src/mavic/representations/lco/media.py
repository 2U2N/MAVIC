from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence


ANALYSIS_WINDOW_SECONDS = 16.0


class MediaPreparationError(RuntimeError):
    """Base error for probing or preparing an LCO analysis clip."""


class MediaProbeError(MediaPreparationError):
    """Raised when media metadata is missing, invalid, or cannot be read."""


class MediaClipError(MediaPreparationError):
    """Raised when a synchronized analysis clip cannot be prepared."""


@dataclass(frozen=True, slots=True)
class MediaInfo:
    duration_seconds: float
    video_stream_index: int
    audio_stream_index: int


@dataclass(frozen=True, slots=True)
class AnalysisWindow:
    start_seconds: float
    end_seconds: float

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass(frozen=True, slots=True)
class PreparedAnalysisClip:
    source_path: Path
    clip_path: Path
    source_duration_seconds: float
    window_start_seconds: float
    window_end_seconds: float

    @property
    def window_duration_seconds(self) -> float:
        return self.window_end_seconds - self.window_start_seconds


def _error_detail(stderr: str | None) -> str:
    detail = (stderr or "").strip()
    if not detail:
        return "no diagnostic output"
    # Avoid placing an unbounded FFmpeg log into a per-video manifest error.
    return detail[-2_000:]


def _run_process(command: Sequence[str], *, operation: str) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise MediaPreparationError(
            f"{operation} failed: executable {command[0]!r} was not found"
        ) from exc
    except OSError as exc:
        raise MediaPreparationError(f"{operation} failed to start: {exc}") from exc

    if result.returncode != 0:
        raise MediaPreparationError(
            f"{operation} failed with exit code {result.returncode}: "
            f"{_error_detail(result.stderr)}"
        )
    return result


def _run_json(command: Sequence[str], *, operation: str) -> dict[str, object]:
    result = _run_process(command, operation=operation)
    try:
        payload = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise MediaProbeError(f"{operation} returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise MediaProbeError(f"{operation} returned a non-object JSON value")
    return payload


def probe_media(path: Path, *, ffprobe: str = "ffprobe") -> MediaInfo:
    """Return validated duration and the first video/audio stream indexes."""
    path = Path(path)
    if not path.is_file():
        raise MediaProbeError(f"media file does not exist: {path}")

    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=index,codec_type",
        "-of",
        "json",
        str(path),
    ]
    try:
        payload = _run_json(command, operation=f"probing {path}")
    except MediaProbeError:
        raise
    except MediaPreparationError as exc:
        raise MediaProbeError(str(exc)) from exc

    format_info = payload.get("format")
    if not isinstance(format_info, dict):
        raise MediaProbeError(f"{path} has no format metadata")
    try:
        duration = float(format_info["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MediaProbeError(f"{path} has no usable duration metadata") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise MediaProbeError(f"{path} duration is not positive and finite")

    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise MediaProbeError(f"{path} has no stream metadata")
    video_indexes: list[int] = []
    audio_indexes: list[int] = []
    for stream in streams:
        if not isinstance(stream, dict):
            continue
        try:
            index = int(stream["index"])
        except (KeyError, TypeError, ValueError):
            continue
        if stream.get("codec_type") == "video":
            video_indexes.append(index)
        elif stream.get("codec_type") == "audio":
            audio_indexes.append(index)

    if not video_indexes:
        raise MediaProbeError(f"{path} contains no video stream")
    if not audio_indexes:
        raise MediaProbeError(f"{path} contains no audio stream")
    return MediaInfo(
        duration_seconds=duration,
        video_stream_index=video_indexes[0],
        audio_stream_index=audio_indexes[0],
    )


def centered_analysis_window(
    duration_seconds: float,
    *,
    maximum_seconds: float = ANALYSIS_WINDOW_SECONDS,
) -> AnalysisWindow:
    """Select the whole short video or a centered window from a long video."""
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive and finite")
    if not math.isfinite(maximum_seconds) or maximum_seconds <= 0:
        raise ValueError("maximum_seconds must be positive and finite")
    window_duration = min(duration_seconds, maximum_seconds)
    start = max(0.0, (duration_seconds - window_duration) / 2.0)
    return AnalysisWindow(start_seconds=start, end_seconds=start + window_duration)


def _format_seconds(value: float) -> str:
    return f"{value:.6f}"


def build_clip_command(
    source: Path,
    destination: Path,
    window: AnalysisWindow,
    *,
    ffmpeg: str = "ffmpeg",
) -> list[str]:
    """Build the deterministic FFmpeg command used for synchronized clipping."""
    if window.start_seconds < 0 or window.duration_seconds <= 0:
        raise ValueError("analysis window must have a non-negative start and positive duration")
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-ss",
        _format_seconds(window.start_seconds),
        "-i",
        str(source),
        "-t",
        _format_seconds(window.duration_seconds),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-sn",
        "-dn",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-avoid_negative_ts",
        "make_zero",
        "-movflags",
        "+faststart",
        str(destination),
    ]


def transcode_analysis_clip(
    source: Path,
    destination: Path,
    window: AnalysisWindow,
    *,
    ffmpeg: str = "ffmpeg",
) -> None:
    command = build_clip_command(source, destination, window, ffmpeg=ffmpeg)
    try:
        _run_process(command, operation=f"creating analysis clip for {source}")
    except MediaPreparationError as exc:
        raise MediaClipError(str(exc)) from exc
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise MediaClipError(f"FFmpeg produced no analysis clip for {source}")


def validate_decoded_audio(path: Path, *, ffprobe: str = "ffprobe") -> int:
    """Require at least one decoded audio frame containing one or more samples."""
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_frames",
        "-show_entries",
        "frame=nb_samples",
        "-of",
        "json",
        str(path),
    ]
    try:
        payload = _run_json(command, operation=f"decoding audio from {path}")
    except MediaPreparationError as exc:
        raise MediaClipError(str(exc)) from exc

    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise MediaClipError(f"analysis clip {path} returned no decoded audio frames")
    total_samples = 0
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        try:
            samples = int(frame.get("nb_samples", 0))
        except (TypeError, ValueError):
            continue
        if samples > 0:
            total_samples += samples
    if total_samples <= 0:
        raise MediaClipError(f"analysis clip {path} contains no decoded audio samples")
    return total_samples


@contextmanager
def prepare_analysis_clip(
    source: Path,
    *,
    temp_dir: Path | None = None,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> Iterator[PreparedAnalysisClip]:
    """Prepare a temporary central audio/video clip and always remove it."""
    source = Path(source)
    media = probe_media(source, ffprobe=ffprobe)
    window = centered_analysis_window(media.duration_seconds)

    try:
        descriptor, raw_path = tempfile.mkstemp(
            prefix="lco-analysis-",
            suffix=".mp4",
            dir=str(temp_dir) if temp_dir is not None else None,
        )
    except OSError as exc:
        raise MediaClipError(f"could not create temporary analysis clip: {exc}") from exc

    clip_path = Path(raw_path)
    try:
        os.close(descriptor)
    except OSError as exc:
        clip_path.unlink(missing_ok=True)
        raise MediaClipError(f"could not initialize temporary analysis clip: {exc}") from exc

    try:
        transcode_analysis_clip(source, clip_path, window, ffmpeg=ffmpeg)
        validate_decoded_audio(clip_path, ffprobe=ffprobe)
        yield PreparedAnalysisClip(
            source_path=source,
            clip_path=clip_path,
            source_duration_seconds=media.duration_seconds,
            window_start_seconds=window.start_seconds,
            window_end_seconds=window.end_seconds,
        )
    finally:
        clip_path.unlink(missing_ok=True)
