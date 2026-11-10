from pathlib import Path

import pytest
import torch

from mavic.representations.openclip.decoding import (
    PreparedVideo,
    VideoDataset,
    collate_prepared,
    decode_uniform_frames,
)


def test_collate_continues_after_failed_video() -> None:
    good = PreparedVideo("good", "good.mp4", frames=torch.zeros(2, 3, 4, 4))
    bad = PreparedVideo("bad", "bad.mp4", error="broken")
    batch = collate_prepared([good, bad])
    assert batch["video_ids"] == ["good"]
    assert batch["frames"].shape == (1, 2, 3, 4, 4)
    assert batch["errors"] == {"bad": "broken"}


def test_dataset_reports_invalid_video_instead_of_raising(tmp_path: Path) -> None:
    invalid = tmp_path / "arbitrary-id.mp4"
    invalid.write_bytes(b"not an mp4")
    dataset = VideoDataset([invalid], preprocess=lambda image: image, samples=16)

    item = dataset[0]

    assert item.video_id == "arbitrary-id"
    assert item.frames is None
    assert item.error is not None


@pytest.mark.integration
def test_decode_all_existing_videos() -> None:
    video_dir = Path(__file__).resolve().parents[1] / "videos"
    videos = sorted(video_dir.glob("*.mp4"))
    if not videos:
        pytest.skip("videos/ corpus is not present")
    assert len(videos) == 15
    for video in videos:
        frames = decode_uniform_frames(video, samples=16)
        assert len(frames) == 16
        assert all(frame.mode == "RGB" and frame.width > 0 and frame.height > 0 for frame in frames)
