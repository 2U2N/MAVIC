from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROMPT = (
    "Instruct: Represent the video metadata by its primary one-word topic.\n"
    "Query:\n"
)
METADATA_TEMPLATE = (
    "Title: {title}\n"
    "Description:\n{description}\n"
    "Categories: {categories}\n"
    "Tags: {tags}"
)

_FORMAT_SUFFIX = re.compile(r"\.f\d+\Z")
_INFO_SUFFIX = ".info.json"


class MetadataError(ValueError):
    """A sidecar cannot be associated with or rendered for a video."""


@dataclass(frozen=True)
class MetadataRecord:
    video_id: str
    metadata_id: str
    sidecar_path: Path
    text: str
    text_sha256: str


def metadata_id_for_video(video_id: str) -> str:
    """Map format-specific MP4 stems back to the sidecar's source video ID."""
    return _FORMAT_SUFFIX.sub("", video_id)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise MetadataError(f"could not read metadata sidecar {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MetadataError(f"metadata sidecar must contain a JSON object: {path}")
    return value


def _optional_string(metadata: dict[str, Any], field: str, path: Path) -> str:
    value = metadata.get(field)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise MetadataError(f"{path}: {field} must be a string or null")
    return value


def _optional_string_list(metadata: dict[str, Any], field: str, path: Path) -> list[str]:
    value = metadata.get(field)
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise MetadataError(f"{path}: {field} must be an array of strings or null")
    return value


def format_metadata(metadata: dict[str, Any], *, source: Path) -> str:
    title = metadata.get("title")
    if not isinstance(title, str) or not title.strip():
        raise MetadataError(f"{source}: title must be a non-empty string")
    description = _optional_string(metadata, "description", source)
    categories = _optional_string_list(metadata, "categories", source)
    tags = _optional_string_list(metadata, "tags", source)
    body = METADATA_TEMPLATE.format(
        title=title,
        description=description,
        categories=", ".join(categories),
        tags=", ".join(tags),
    )
    return PROMPT + body


class MetadataCatalog:
    """Index direct yt-dlp info sidecars by their internal source ID."""

    def __init__(self, input_dir: Path) -> None:
        self.input_dir = Path(input_dir)
        self._by_id: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
        for path in sorted(self.input_dir.glob(f"*{_INFO_SUFFIX}"), key=lambda item: item.name):
            try:
                metadata = _read_json(path)
            except MetadataError:
                # A malformed exact-name sidecar is surfaced when its video is looked up.
                continue
            metadata_id = metadata.get("id")
            if isinstance(metadata_id, str) and metadata_id:
                self._by_id.setdefault(metadata_id, []).append((path, metadata))

    def record_for_video(self, video_path: Path) -> MetadataRecord:
        video_path = Path(video_path)
        video_id = video_path.stem
        metadata_id = metadata_id_for_video(video_id)
        candidates = self._by_id.get(metadata_id, [])
        if len(candidates) > 1:
            names = ", ".join(path.name for path, _ in candidates)
            raise MetadataError(f"ambiguous metadata sidecars for {metadata_id}: {names}")

        exact_path = self.input_dir / f"{metadata_id}{_INFO_SUFFIX}"
        if exact_path.is_file():
            metadata = _read_json(exact_path)
            internal_id = metadata.get("id")
            if internal_id != metadata_id:
                raise MetadataError(
                    f"{exact_path}: internal id {internal_id!r} does not match {metadata_id!r}"
                )
            if candidates and candidates[0][0] != exact_path:
                raise MetadataError(f"ambiguous metadata sidecars for {metadata_id}")
            sidecar_path = exact_path
        elif len(candidates) == 1:
            sidecar_path, metadata = candidates[0]
        else:
            raise MetadataError(f"no metadata sidecar found for video {video_id}")

        text = format_metadata(metadata, source=sidecar_path)
        return MetadataRecord(
            video_id=video_id,
            metadata_id=metadata_id,
            sidecar_path=sidecar_path,
            text=text,
            text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )
