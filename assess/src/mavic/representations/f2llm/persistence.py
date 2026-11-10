from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .embedding import EMBEDDING_DIMENSION


PIPELINE_ID = "f2llm-v2-4b-metadata"
SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
DEFAULT_CONSOLIDATED_NAME = "f2llm-embeddings.npz"

_SHARD_KEYS = frozenset(
    {"video_ids", "embeddings", "token_counts", "truncated", "text_sha256"}
)
_SHARD_NAME = re.compile(r"f2llm-shard-(\d{5})\.npz\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class ShardArrays:
    video_ids: np.ndarray
    embeddings: np.ndarray
    token_counts: np.ndarray
    truncated: np.ndarray
    text_sha256: np.ndarray

    @property
    def count(self) -> int:
        return len(self.video_ids)


@dataclass
class StoredState:
    manifest: dict[str, Any]
    completed_ids: set[str]
    completed_text_hashes: dict[str, str]
    next_shard_index: int


def create_manifest(
    config: Mapping[str, Any], *, provenance: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    timestamp = utc_now()
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_id": PIPELINE_ID,
        "config": dict(config),
        "provenance": dict(provenance or {}),
        "created_at": timestamp,
        "updated_at": timestamp,
        "shards": [],
        "errors": {},
        "stats": {},
        "last_run": {},
        "runs": [],
    }


def _validate_manifest_header(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("F2LLM manifest must be a JSON object")
    if manifest.get("pipeline_id") != PIPELINE_ID:
        raise ValueError(f"output does not belong to the {PIPELINE_ID!r} pipeline")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported F2LLM manifest schema: {manifest.get('schema_version')}"
        )
    if not isinstance(manifest.get("config"), dict):
        raise ValueError("F2LLM manifest config must be an object")
    if not isinstance(manifest.get("shards"), list):
        raise ValueError("F2LLM manifest shards must be an array")
    if not isinstance(manifest.get("errors", {}), dict):
        raise ValueError("F2LLM manifest errors must be an object")
    return manifest


def load_manifest(output_dir: Path) -> dict[str, Any] | None:
    path = Path(output_dir) / MANIFEST_NAME
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError(f"F2LLM manifest is not a file: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read F2LLM manifest {path}: {exc}") from exc
    return _validate_manifest_header(manifest)


def validate_compatible(existing: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    manifest = _validate_manifest_header(dict(existing))
    if manifest["config"] != dict(config):
        raise ValueError(
            "output contains an incompatible F2LLM run configuration; "
            "choose another output directory or pass --overwrite"
        )


def _validate_shard_entry(entry: Any) -> tuple[int, str, int]:
    if not isinstance(entry, dict):
        raise ValueError("F2LLM shard inventory entries must be objects")
    index = entry.get("index")
    count = entry.get("count")
    filename = entry.get("file")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError(f"invalid F2LLM shard index: {index!r}")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError(f"invalid F2LLM shard count at index {index}: {count!r}")
    expected = f"f2llm-shard-{index:05d}.npz"
    if filename != expected or _SHARD_NAME.fullmatch(str(filename)) is None:
        raise ValueError(f"invalid F2LLM shard filename at index {index}: {filename!r}")
    return index, expected, count


def _coerce_shard_arrays(
    video_ids: Sequence[str] | np.ndarray,
    embeddings: np.ndarray,
    token_counts: Sequence[int] | np.ndarray,
    truncated: Sequence[bool] | np.ndarray,
    text_sha256: Sequence[str] | np.ndarray,
) -> ShardArrays:
    raw_counts = np.asarray(token_counts)
    if (
        raw_counts.ndim != 1
        or not np.issubdtype(raw_counts.dtype, np.integer)
        or np.issubdtype(raw_counts.dtype, np.bool_)
        or np.any(raw_counts < 1)
        or np.any(raw_counts > np.iinfo(np.uint32).max)
    ):
        raise ValueError("token counts must be positive uint32-compatible integers")
    raw_truncated = np.asarray(truncated)
    if raw_truncated.ndim != 1 or not np.issubdtype(raw_truncated.dtype, np.bool_):
        raise ValueError("truncated must contain boolean values")
    return ShardArrays(
        video_ids=np.asarray(video_ids, dtype=np.str_),
        embeddings=np.asarray(embeddings, dtype=np.float32),
        token_counts=raw_counts.astype(np.uint32, copy=False),
        truncated=raw_truncated.astype(np.bool_, copy=False),
        text_sha256=np.asarray(text_sha256, dtype=np.str_),
    )


def validate_shard_arrays(arrays: ShardArrays, *, source: str = "F2LLM shard") -> None:
    ids = arrays.video_ids
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.str_):
        raise ValueError(f"{source}: video_ids must be a one-dimensional string array")
    count = len(ids)
    if count == 0:
        raise ValueError(f"{source}: empty shards are not allowed")
    id_values = ids.tolist()
    if any(not value for value in id_values):
        raise ValueError(f"{source}: video IDs must be non-empty")
    if len(id_values) != len(set(id_values)):
        raise ValueError(f"{source}: duplicate video IDs")

    embeddings = arrays.embeddings
    if embeddings.dtype != np.float32:
        raise ValueError(f"{source}: embeddings must have dtype float32")
    if embeddings.shape != (count, EMBEDDING_DIMENSION):
        raise ValueError(
            f"{source}: embeddings must have shape (N, {EMBEDDING_DIMENSION})"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError(f"{source}: embeddings contain non-finite values")
    if not np.allclose(
        np.linalg.norm(embeddings, axis=1), 1.0, rtol=1e-4, atol=1e-4
    ):
        raise ValueError(f"{source}: embeddings must be L2-normalized")

    if arrays.token_counts.dtype != np.uint32 or arrays.token_counts.shape != (count,):
        raise ValueError(f"{source}: token_counts must be a uint32 vector of length N")
    if np.any(arrays.token_counts == 0):
        raise ValueError(f"{source}: token_counts must be positive")
    if arrays.truncated.dtype != np.bool_ or arrays.truncated.shape != (count,):
        raise ValueError(f"{source}: truncated must be a boolean vector of length N")
    hashes = arrays.text_sha256
    if hashes.ndim != 1 or not np.issubdtype(hashes.dtype, np.str_) or len(hashes) != count:
        raise ValueError(f"{source}: text_sha256 must be a string vector of length N")
    if any(_SHA256.fullmatch(value) is None for value in hashes.tolist()):
        raise ValueError(f"{source}: invalid text SHA-256 value")


def read_shard(path: Path) -> ShardArrays:
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as data:
            keys = frozenset(data.files)
            if keys != _SHARD_KEYS:
                missing = sorted(_SHARD_KEYS - keys)
                extra = sorted(keys - _SHARD_KEYS)
                raise ValueError(
                    f"invalid arrays in {path.name}; missing={missing}, extra={extra}"
                )
            arrays = ShardArrays(
                video_ids=np.array(data["video_ids"], copy=True),
                embeddings=np.array(data["embeddings"], copy=True),
                token_counts=np.array(data["token_counts"], copy=True),
                truncated=np.array(data["truncated"], copy=True),
                text_sha256=np.array(data["text_sha256"], copy=True),
            )
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("invalid arrays"):
            raise
        raise ValueError(f"could not read F2LLM shard {path}: {exc}") from exc
    validate_shard_arrays(arrays, source=path.name)
    return arrays


def load_state(output_dir: Path, config: Mapping[str, Any]) -> StoredState:
    output_dir = Path(output_dir)
    manifest = load_manifest(output_dir)
    if manifest is None:
        return StoredState(create_manifest(config), set(), {}, 0)
    validate_compatible(manifest, config)

    completed_hashes: dict[str, str] = {}
    indexes: set[int] = set()
    filenames: set[str] = set()
    max_index = -1
    for entry in manifest["shards"]:
        index, filename, expected_count = _validate_shard_entry(entry)
        if index in indexes or filename in filenames:
            raise ValueError("duplicate entries in F2LLM shard inventory")
        indexes.add(index)
        filenames.add(filename)
        shard_path = output_dir / filename
        if not shard_path.is_file():
            raise ValueError(f"manifest references missing F2LLM shard: {shard_path}")
        arrays = read_shard(shard_path)
        if arrays.count != expected_count:
            raise ValueError(
                f"manifest count does not match {filename}: "
                f"{expected_count} != {arrays.count}"
            )
        for video_id, text_hash in zip(
            arrays.video_ids.tolist(), arrays.text_sha256.tolist(), strict=True
        ):
            if video_id in completed_hashes:
                raise ValueError(f"duplicate video ID across F2LLM shards: {video_id}")
            completed_hashes[video_id] = text_hash
        max_index = max(max_index, index)
    return StoredState(manifest, set(completed_hashes), completed_hashes, max_index + 1)


def write_shard(
    output_dir: Path,
    index: int,
    video_ids: Sequence[str] | np.ndarray,
    embeddings: np.ndarray,
    token_counts: Sequence[int] | np.ndarray,
    truncated: Sequence[bool] | np.ndarray,
    text_sha256: Sequence[str] | np.ndarray,
) -> dict[str, Any]:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("F2LLM shard index must be a non-negative integer")
    arrays = _coerce_shard_arrays(
        video_ids, embeddings, token_counts, truncated, text_sha256
    )
    filename = f"f2llm-shard-{index:05d}.npz"
    validate_shard_arrays(arrays, source=filename)
    atomic_npz(
        Path(output_dir) / filename,
        video_ids=arrays.video_ids,
        embeddings=arrays.embeddings,
        token_counts=arrays.token_counts,
        truncated=arrays.truncated,
        text_sha256=arrays.text_sha256,
    )
    return {"index": index, "file": filename, "count": arrays.count}


def add_shard(manifest: dict[str, Any], shard: Mapping[str, Any]) -> None:
    _validate_manifest_header(manifest)
    index, filename, _ = _validate_shard_entry(dict(shard))
    for existing in manifest["shards"]:
        existing_index, existing_filename, _ = _validate_shard_entry(existing)
        if existing_index == index or existing_filename == filename:
            raise ValueError(f"F2LLM shard {filename} is already in the manifest")
    manifest["shards"].append(dict(shard))


def _validate_consolidated_entry(entry: Any) -> tuple[str, int]:
    if not isinstance(entry, dict):
        raise ValueError("invalid consolidated artifact in F2LLM manifest")
    filename = entry.get("file")
    count = entry.get("count")
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or filename == MANIFEST_NAME
        or _SHARD_NAME.fullmatch(filename) is not None
    ):
        raise ValueError("invalid consolidated artifact filename in F2LLM manifest")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("invalid consolidated artifact count in F2LLM manifest")
    return filename, count


def checkpoint_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    _validate_manifest_header(manifest)
    indexes: set[int] = set()
    filenames: set[str] = set()
    for entry in manifest["shards"]:
        index, filename, _ = _validate_shard_entry(entry)
        if index in indexes or filename in filenames:
            raise ValueError("duplicate entries in F2LLM shard inventory")
        indexes.add(index)
        filenames.add(filename)
        if not (Path(output_dir) / filename).is_file():
            raise ValueError(f"cannot checkpoint missing F2LLM shard: {filename}")
    if manifest.get("consolidated") is not None:
        _validate_consolidated_entry(manifest["consolidated"])
    manifest["updated_at"] = utc_now()
    atomic_json(Path(output_dir) / MANIFEST_NAME, manifest)


def reset_generated_output(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(output_dir)
    if manifest is None:
        return
    shard_filenames = [_validate_shard_entry(entry)[1] for entry in manifest["shards"]]
    consolidated_filename: str | None = None
    if manifest.get("consolidated") is not None:
        consolidated_filename, _ = _validate_consolidated_entry(manifest["consolidated"])
    for filename in shard_filenames:
        (output_dir / filename).unlink(missing_ok=True)
    if consolidated_filename is not None:
        (output_dir / consolidated_filename).unlink(missing_ok=True)
    (output_dir / MANIFEST_NAME).unlink(missing_ok=True)


def consolidate(output_dir: Path, destination: Path) -> int:
    output_dir = Path(output_dir)
    destination = Path(destination)
    manifest = load_manifest(output_dir)
    if manifest is None:
        raise ValueError(f"no F2LLM manifest found in {output_dir}")
    try:
        destination_is_local = destination.resolve().parent == output_dir.resolve()
    except OSError:
        destination_is_local = False
    if destination_is_local and (
        destination.name == MANIFEST_NAME
        or _SHARD_NAME.fullmatch(destination.name) is not None
    ):
        raise ValueError(
            "F2LLM consolidation destination must not replace the manifest or a shard"
        )

    array_groups: dict[str, list[np.ndarray]] = {key: [] for key in _SHARD_KEYS}
    seen: set[str] = set()
    validated = sorted(
        (_validate_shard_entry(entry) for entry in manifest["shards"]),
        key=lambda value: value[0],
    )
    seen_indexes: set[int] = set()
    for index, filename, expected_count in validated:
        if index in seen_indexes:
            raise ValueError("duplicate entries in F2LLM shard inventory")
        seen_indexes.add(index)
        arrays = read_shard(output_dir / filename)
        if arrays.count != expected_count:
            raise ValueError(f"manifest count does not match {filename}")
        ids = arrays.video_ids.tolist()
        overlap = seen.intersection(ids)
        if overlap:
            raise ValueError(
                "duplicate video IDs found while consolidating F2LLM shards: "
                + ", ".join(sorted(overlap))
            )
        seen.update(ids)
        for key in _SHARD_KEYS:
            array_groups[key].append(getattr(arrays, key))
    if not array_groups["video_ids"]:
        raise ValueError("no completed F2LLM embeddings to consolidate")

    combined = {key: np.concatenate(values) for key, values in array_groups.items()}
    order = np.argsort(combined["video_ids"], kind="stable")
    atomic_npz(destination, **{key: value[order] for key, value in combined.items()})
    if destination_is_local:
        manifest["consolidated"] = {
            "file": destination.name,
            "count": len(combined["video_ids"]),
        }
        checkpoint_manifest(output_dir, manifest)
    return len(combined["video_ids"])
