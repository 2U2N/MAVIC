from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import mavic.representations.lco.media as media
from mavic.representations.lco.media import (
    AnalysisWindow,
    MediaClipError,
    MediaInfo,
    MediaProbeError,
    build_clip_command,
    centered_analysis_window,
    prepare_analysis_clip,
    probe_media,
    validate_decoded_audio,
)


def _completed(payload: object, *, returncode: int = 0, stderr: str = ""):
    return subprocess.CompletedProcess(
        args=[],
        returncode=returncode,
        stdout=json.dumps(payload),
        stderr=stderr,
    )


@pytest.mark.parametrize(
    ("duration", "expected"),
    [
        (8.0, AnalysisWindow(0.0, 8.0)),
        (16.0, AnalysisWindow(0.0, 16.0)),
        (30.0, AnalysisWindow(7.0, 23.0)),
    ],
)
def test_centered_analysis_window(duration: float, expected: AnalysisWindow) -> None:
    assert centered_analysis_window(duration) == expected


@pytest.mark.parametrize("duration", [0.0, -1.0, float("inf"), float("nan")])
def test_centered_analysis_window_rejects_invalid_duration(duration: float) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        centered_analysis_window(duration)


def test_probe_media_returns_first_video_and_audio_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "id.mp4"
    source.write_bytes(b"container")
    seen: list[str] = []

    def fake_run(command, **kwargs):
        seen.extend(command)
        return _completed(
            {
                "streams": [
                    {"index": 4, "codec_type": "audio"},
                    {"index": 2, "codec_type": "video"},
                    {"index": 8, "codec_type": "audio"},
                ],
                "format": {"duration": "21.25"},
            }
        )

    monkeypatch.setattr(media.subprocess, "run", fake_run)
    result = probe_media(source)

    assert result == MediaInfo(21.25, video_stream_index=2, audio_stream_index=4)
    assert seen == [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=index,codec_type",
        "-of",
        "json",
        str(source),
    ]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"format": {"duration": "0"}, "streams": []}, "not positive"),
        (
            {"format": {"duration": "1"}, "streams": [{"index": 0, "codec_type": "audio"}]},
            "no video stream",
        ),
        (
            {"format": {"duration": "1"}, "streams": [{"index": 0, "codec_type": "video"}]},
            "no audio stream",
        ),
    ],
)
def test_probe_media_rejects_invalid_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
    message: str,
) -> None:
    source = tmp_path / "bad.mp4"
    source.touch()
    monkeypatch.setattr(media.subprocess, "run", lambda *args, **kwargs: _completed(payload))

    with pytest.raises(MediaProbeError, match=message):
        probe_media(source)


def test_probe_media_reports_tool_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "bad.mp4"
    source.touch()
    monkeypatch.setattr(
        media.subprocess,
        "run",
        lambda *args, **kwargs: _completed({}, returncode=1, stderr="invalid container"),
    )

    with pytest.raises(MediaProbeError, match="invalid container"):
        probe_media(source)


def test_build_clip_command_is_deterministic() -> None:
    command = build_clip_command(
        Path("source.mp4"),
        Path("clip.mp4"),
        AnalysisWindow(7.0, 23.0),
    )

    assert command == [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-ss",
        "7.000000",
        "-i",
        "source.mp4",
        "-t",
        "16.000000",
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
        "clip.mp4",
    ]


def test_validate_decoded_audio_accepts_samples_without_amplitude_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        media.subprocess,
        "run",
        lambda *args, **kwargs: _completed(
            {"frames": [{"nb_samples": 1024}, {"nb_samples": "960"}]}
        ),
    )

    assert validate_decoded_audio(Path("clip.mp4")) == 1984


@pytest.mark.parametrize("frames", [[], [{}], [{"nb_samples": "N/A"}], [{"nb_samples": 0}]])
def test_validate_decoded_audio_rejects_no_samples(
    monkeypatch: pytest.MonkeyPatch, frames: list[dict[str, object]]
) -> None:
    monkeypatch.setattr(
        media.subprocess,
        "run",
        lambda *args, **kwargs: _completed({"frames": frames}),
    )

    with pytest.raises(MediaClipError, match="no decoded audio samples"):
        validate_decoded_audio(Path("clip.mp4"))


def test_prepare_analysis_clip_yields_metadata_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.mp4"
    source.touch()
    monkeypatch.setattr(media, "probe_media", lambda *args, **kwargs: MediaInfo(30.0, 0, 1))

    def fake_transcode(source_path, destination, window, **kwargs):
        assert source_path == source
        assert window == AnalysisWindow(7.0, 23.0)
        destination.write_bytes(b"prepared clip")

    monkeypatch.setattr(media, "transcode_analysis_clip", fake_transcode)
    monkeypatch.setattr(media, "validate_decoded_audio", lambda *args, **kwargs: 1024)

    with prepare_analysis_clip(source, temp_dir=tmp_path) as prepared:
        clip_path = prepared.clip_path
        assert clip_path.is_file()
        assert prepared.source_duration_seconds == 30.0
        assert prepared.window_start_seconds == 7.0
        assert prepared.window_end_seconds == 23.0
        assert prepared.window_duration_seconds == 16.0

    assert not clip_path.exists()


def test_prepare_analysis_clip_cleans_up_when_audio_validation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.mp4"
    source.touch()
    created: list[Path] = []
    monkeypatch.setattr(media, "probe_media", lambda *args, **kwargs: MediaInfo(5.0, 0, 1))

    def fake_transcode(source_path, destination, window, **kwargs):
        destination.write_bytes(b"prepared clip")
        created.append(destination)

    def fail_audio(*args, **kwargs):
        raise MediaClipError("no decoded audio samples")

    monkeypatch.setattr(media, "transcode_analysis_clip", fake_transcode)
    monkeypatch.setattr(media, "validate_decoded_audio", fail_audio)

    with pytest.raises(MediaClipError, match="no decoded audio samples"):
        with prepare_analysis_clip(source, temp_dir=tmp_path):
            pytest.fail("invalid clip must not be yielded")

    assert len(created) == 1
    assert not created[0].exists()


def test_prepare_analysis_clip_preserves_exception_from_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.mp4"
    source.touch()
    monkeypatch.setattr(media, "probe_media", lambda *args, **kwargs: MediaInfo(5.0, 0, 1))

    def fake_transcode(source_path, destination, window, **kwargs):
        destination.write_bytes(b"prepared clip")

    monkeypatch.setattr(media, "transcode_analysis_clip", fake_transcode)
    monkeypatch.setattr(media, "validate_decoded_audio", lambda *args, **kwargs: 1024)

    with pytest.raises(OSError, match="caller failed"):
        with prepare_analysis_clip(source, temp_dir=tmp_path) as prepared:
            clip_path = prepared.clip_path
            raise OSError("caller failed")

    assert not clip_path.exists()
