from pathlib import Path

import pytest

from mavic.representations.openclip.sampling import bin_center_timestamps, discover_videos


def test_bin_center_timestamps() -> None:
    assert bin_center_timestamps(8.0, samples=4) == [1.0, 3.0, 5.0, 7.0]
    assert bin_center_timestamps(4.0, samples=2, start_seconds=10.0) == [11.0, 13.0]


@pytest.mark.parametrize("duration,samples", [(0, 16), (-1, 16), (1, 0)])
def test_bin_centers_reject_invalid_values(duration: float, samples: int) -> None:
    with pytest.raises(ValueError):
        bin_center_timestamps(duration, samples)


def test_discovery_is_flat_case_insensitive_and_deterministic(tmp_path: Path) -> None:
    (tmp_path / "z.mp4").touch()
    (tmp_path / "arbitrary name.MP4").touch()
    (tmp_path / "ignore.mov").touch()
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "hidden.mp4").touch()

    found = discover_videos(tmp_path)

    assert [path.name for path in found] == ["arbitrary name.MP4", "z.mp4"]
    assert [path.stem for path in found] == ["arbitrary name", "z"]


def test_discovery_rejects_duplicate_stems(tmp_path: Path) -> None:
    (tmp_path / "same.mp4").touch()
    (tmp_path / "same.MP4").touch()
    if len(list(tmp_path.iterdir())) < 2:
        pytest.skip("case-insensitive filesystem cannot hold both files")
    with pytest.raises(ValueError, match="duplicate video IDs"):
        discover_videos(tmp_path)
