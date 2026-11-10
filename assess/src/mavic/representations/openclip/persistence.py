from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_manifest(output_dir: Path) -> dict[str, Any] | None:
    path = Path(output_dir) / MANIFEST_NAME
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported manifest schema: {manifest.get('schema_version')}")
    return manifest


def validate_compatible(existing: dict[str, Any], config: dict[str, Any]) -> None:
    if existing.get("config") != config:
        raise ValueError(
            "output contains an incompatible run configuration; "
            "choose another output directory or pass --overwrite"
        )


@dataclass
class StoredState:
    manifest: dict[str, Any]
    completed_ids: set[str]
    next_shard_index: int


def load_state(output_dir: Path, config: dict[str, Any]) -> StoredState:
    manifest = load_manifest(output_dir)
    if manifest is None:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "config": config,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "shards": [],
            "errors": {},
            "stats": {},
            "last_run": {},
            "runs": [],
        }
        return StoredState(manifest, set(), 0)

    validate_compatible(manifest, config)
    completed: set[str] = set()
    max_index = -1
    for shard in manifest.get("shards", []):
        shard_path = output_dir / shard["file"]
        if not shard_path.is_file():
            raise ValueError(f"manifest references missing shard: {shard_path}")
        with np.load(shard_path, allow_pickle=False) as data:
            ids = [str(value) for value in data["video_ids"].tolist()]
        overlap = completed.intersection(ids)
        if overlap:
            raise ValueError(f"duplicate IDs across shards: {', '.join(sorted(overlap))}")
        completed.update(ids)
        max_index = max(max_index, int(shard["index"]))
    return StoredState(manifest, completed, max_index + 1)


def write_shard(
    output_dir: Path,
    index: int,
    video_ids: list[str],
    embeddings: np.ndarray,
    retained_frame_counts: np.ndarray,
) -> dict[str, Any]:
    filename = f"shard-{index:05d}.npz"
    atomic_npz(
        output_dir / filename,
        video_ids=np.asarray(video_ids, dtype=np.str_),
        embeddings=np.asarray(embeddings, dtype=np.float32),
        retained_frame_counts=np.asarray(retained_frame_counts, dtype=np.uint8),
    )
    return {"index": index, "file": filename, "count": len(video_ids)}


def reset_generated_output(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(output_dir)
    if manifest is not None:
        for shard in manifest.get("shards", []):
            (output_dir / shard["file"]).unlink(missing_ok=True)
        (output_dir / MANIFEST_NAME).unlink(missing_ok=True)


def consolidate(output_dir: Path, destination: Path) -> int:
    manifest = load_manifest(output_dir)
    if manifest is None:
        raise ValueError(f"no manifest found in {output_dir}")

    ids: list[str] = []
    embeddings: list[np.ndarray] = []
    counts: list[np.ndarray] = []
    for shard in sorted(manifest.get("shards", []), key=lambda item: int(item["index"])):
        with np.load(output_dir / shard["file"], allow_pickle=False) as data:
            shard_ids = np.asarray(data["video_ids"], dtype=np.str_)
            shard_embeddings = np.asarray(data["embeddings"], dtype=np.float32)
            shard_counts = np.asarray(data["retained_frame_counts"], dtype=np.uint8)
        if shard_embeddings.ndim != 2 or len(shard_ids) != len(shard_embeddings):
            raise ValueError(f"invalid arrays in {shard['file']}")
        if len(shard_counts) != len(shard_ids):
            raise ValueError(f"invalid retained counts in {shard['file']}")
        ids.extend(shard_ids.tolist())
        embeddings.append(shard_embeddings)
        counts.append(shard_counts)

    if not ids:
        raise ValueError("no completed embeddings to consolidate")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate video IDs found while consolidating")

    matrix = np.concatenate(embeddings, axis=0)
    retained = np.concatenate(counts, axis=0)
    order = np.argsort(np.asarray(ids, dtype=np.str_))
    atomic_npz(
        destination,
        video_ids=np.asarray(ids, dtype=np.str_)[order],
        embeddings=matrix[order],
        retained_frame_counts=retained[order],
    )
    return len(ids)
