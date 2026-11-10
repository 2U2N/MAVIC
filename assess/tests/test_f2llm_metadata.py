from __future__ import annotations

import json
from pathlib import Path

import pytest

from mavic.representations.f2llm.metadata import (
    PROMPT,
    MetadataCatalog,
    MetadataError,
    format_metadata,
    metadata_id_for_video,
)


def _sidecar(path: Path, video_id: str, **values: object) -> Path:
    payload = {"id": video_id, "title": f"Title {video_id}", **values}
    target = path / f"{video_id}.info.json"
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return target


def test_format_metadata_preserves_raw_text_and_uses_fixed_prompt(tmp_path: Path) -> None:
    source = tmp_path / "id.info.json"
    text = format_metadata(
        {
            "title": "  Blå titel  ",
            "description": "Line one\n\nhttps://example.com  ",
            "categories": ["Film", "Kultur"],
            "tags": ["ét", "to"],
        },
        source=source,
    )
    assert text == (
        PROMPT
        + "Title:   Blå titel  \n"
        "Description:\nLine one\n\nhttps://example.com  \n"
        "Categories: Film, Kultur\nTags: ét, to"
    )

    optional = format_metadata({"title": "Only title"}, source=source)
    assert optional.endswith(
        "Title: Only title\nDescription:\n\nCategories: \nTags: "
    )


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({}, "title"),
        ({"title": "x", "description": 1}, "description"),
        ({"title": "x", "categories": "Film"}, "categories"),
        ({"title": "x", "tags": ["ok", 1]}, "tags"),
    ],
)
def test_format_metadata_rejects_invalid_fields(
    tmp_path: Path, metadata: dict, message: str
) -> None:
    with pytest.raises(MetadataError, match=message):
        format_metadata(metadata, source=tmp_path / "bad.info.json")


def test_catalog_matches_format_suffix_and_ignores_nested_sidecars(tmp_path: Path) -> None:
    video = tmp_path / "source.f633.mp4"
    video.touch()
    _sidecar(tmp_path, "source", description="raw")
    checkpoint = tmp_path / ".ipynb_checkpoints"
    checkpoint.mkdir()
    _sidecar(checkpoint, "source", description="duplicate")

    record = MetadataCatalog(tmp_path).record_for_video(video)

    assert metadata_id_for_video(video.stem) == "source"
    assert record.video_id == "source.f633"
    assert record.metadata_id == "source"
    assert record.sidecar_path == tmp_path / "source.info.json"
    assert len(record.text_sha256) == 64


def test_catalog_reports_missing_malformed_mismatched_and_duplicate_sidecars(
    tmp_path: Path,
) -> None:
    (tmp_path / "missing.mp4").touch()
    catalog = MetadataCatalog(tmp_path)
    with pytest.raises(MetadataError, match="no metadata sidecar"):
        catalog.record_for_video(tmp_path / "missing.mp4")

    (tmp_path / "bad.mp4").touch()
    (tmp_path / "bad.info.json").write_text("{", encoding="utf-8")
    with pytest.raises(MetadataError, match="could not read"):
        MetadataCatalog(tmp_path).record_for_video(tmp_path / "bad.mp4")

    _sidecar(tmp_path, "wrong", title="wrong")
    (tmp_path / "wrong.info.json").write_text(
        json.dumps({"id": "another", "title": "wrong"}), encoding="utf-8"
    )
    with pytest.raises(MetadataError, match="does not match"):
        MetadataCatalog(tmp_path).record_for_video(tmp_path / "wrong.mp4")

    _sidecar(tmp_path, "same")
    (tmp_path / "duplicate.info.json").write_text(
        json.dumps({"id": "same", "title": "duplicate"}), encoding="utf-8"
    )
    with pytest.raises(MetadataError, match="ambiguous"):
        MetadataCatalog(tmp_path).record_for_video(tmp_path / "same.mp4")
