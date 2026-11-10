from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import av
import torch
from PIL import Image

from .sampling import bin_center_timestamps


@dataclass
class PreparedVideo:
    video_id: str
    path: str
    frames: torch.Tensor | None = None
    error: str | None = None


def _frame_time_seconds(frame: av.VideoFrame, stream: av.video.stream.VideoStream) -> float | None:
    if frame.time is not None:
        return float(frame.time)
    if frame.pts is not None and stream.time_base is not None:
        return float(frame.pts * stream.time_base)
    return None


def _stream_timing(
    container: av.container.InputContainer, stream: av.video.stream.VideoStream
) -> tuple[float, float]:
    start = float(stream.start_time * stream.time_base) if stream.start_time is not None else 0.0
    if stream.duration is not None and stream.time_base is not None:
        duration = float(stream.duration * stream.time_base)
    elif container.duration is not None:
        duration = float(container.duration / av.time_base)
    else:
        raise ValueError("video has no usable duration metadata")
    if duration <= 0:
        raise ValueError("video duration is not positive")
    return start, duration


def decode_uniform_frames(path: Path, samples: int = 16) -> list[Image.Image]:
    """Seek to each bin center and return the closest decodable RGB frame."""
    result: list[Image.Image] = []
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("file contains no video stream")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        start, duration = _stream_timing(container, stream)
        targets = bin_center_timestamps(duration, samples=samples, start_seconds=start)

        if stream.time_base is None:
            raise ValueError("video stream has no time base")
        for target in targets:
            seek_offset = max(0, int(target / float(stream.time_base)))
            container.seek(seek_offset, stream=stream, backward=True, any_frame=False)
            closest: av.VideoFrame | None = None
            closest_distance = float("inf")
            for frame in container.decode(stream):
                timestamp = _frame_time_seconds(frame, stream)
                if timestamp is None:
                    continue
                distance = abs(timestamp - target)
                if distance < closest_distance:
                    closest = frame
                    closest_distance = distance
                if timestamp >= target:
                    break
            if closest is None:
                raise ValueError(f"could not decode a frame near {target:.3f}s")
            result.append(closest.to_image().convert("RGB"))
    return result


class VideoDataset(torch.utils.data.Dataset[PreparedVideo]):
    def __init__(
        self,
        paths: list[Path],
        preprocess: Callable[[Image.Image], torch.Tensor],
        samples: int,
    ) -> None:
        self.paths = paths
        self.preprocess = preprocess
        self.samples = samples

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> PreparedVideo:
        path = self.paths[index]
        try:
            images = decode_uniform_frames(path, samples=self.samples)
            frames = torch.stack([self.preprocess(image) for image in images])
            return PreparedVideo(video_id=path.stem, path=str(path), frames=frames)
        except Exception as exc:  # Batch processing must isolate bad media.
            return PreparedVideo(
                video_id=path.stem,
                path=str(path),
                error=f"{type(exc).__name__}: {exc}",
            )


def collate_prepared(items: list[PreparedVideo]) -> dict[str, object]:
    successful = [item for item in items if item.frames is not None]
    errors = [item for item in items if item.frames is None]
    frames = torch.stack([item.frames for item in successful]) if successful else None
    return {
        "video_ids": [item.video_id for item in successful],
        "paths": [item.path for item in successful],
        "frames": frames,
        "errors": {item.video_id: item.error for item in errors},
    }
