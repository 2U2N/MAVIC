from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PIPELINE_ID = "lco-embedding-omni-7b-av"
SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
DEFAULT_CONSOLIDATED_NAME = "lco-embeddings.npz"
EMBEDDING_DIMENSION = 3584
MAX_SAMPLED_FRAMES = 32

_SHARD_KEYS = frozenset(
    {
        "video_ids",
        "embeddings",
        "source_duration_seconds",
        "window_start_seconds",
        "window_end_seconds",
        "sampled_frame_counts",
    }
)
_SHARD_NAME = re.compile(r"lco-shard-(\d{5})\.npz\Z")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fsync_directory(path: Path) -> None:
    """Persist a completed rename on filesystems that support directory fsync."""
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
    source_duration_seconds: np.ndarray
    window_start_seconds: np.ndarray
    window_end_seconds: np.ndarray
    sampled_frame_counts: np.ndarray

    @property
    def count(self) -> int:
        return len(self.video_ids)


@dataclass
class StoredState:
    manifest: dict[str, Any]
    completed_ids: set[str]
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
        raise ValueError("LCO manifest must be a JSON object")
    if manifest.get("pipeline_id") != PIPELINE_ID:
        raise ValueError(
            f"output does not belong to the {PIPELINE_ID!r} pipeline"
        )
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported LCO manifest schema: {manifest.get('schema_version')}"
        )
    if not isinstance(manifest.get("config"), dict):
        raise ValueError("LCO manifest config must be an object")
    if not isinstance(manifest.get("shards"), list):
        raise ValueError("LCO manifest shards must be an array")
    if not isinstance(manifest.get("errors", {}), dict):
        raise ValueError("LCO manifest errors must be an object")
    return manifest


def load_manifest(output_dir: Path) -> dict[str, Any] | None:
    path = Path(output_dir) / MANIFEST_NAME
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError(f"LCO manifest is not a file: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read LCO manifest {path}: {error}") from error
    return _validate_manifest_header(manifest)


def validate_compatible(existing: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    manifest = _validate_manifest_header(dict(existing))
    if manifest["config"] != dict(config):
        raise ValueError(
            "output contains an incompatible LCO run configuration; "
            "choose another output directory or pass --overwrite"
        )


def _validate_shard_entry(entry: Any) -> tuple[int, str, int]:
    if not isinstance(entry, dict):
        raise ValueError("LCO shard inventory entries must be objects")
    index = entry.get("index")
    count = entry.get("count")
    filename = entry.get("file")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError(f"invalid LCO shard index: {index!r}")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError(f"invalid LCO shard count at index {index}: {count!r}")
    expected = f"lco-shard-{index:05d}.npz"
    if filename != expected or _SHARD_NAME.fullmatch(str(filename)) is None:
        raise ValueError(
            f"invalid LCO shard filename at index {index}: {filename!r}"
        )
    return index, expected, count


def _coerce_shard_arrays(
    video_ids: Sequence[str] | np.ndarray,
    embeddings: np.ndarray,
    source_duration_seconds: Sequence[float] | np.ndarray,
    window_start_seconds: Sequence[float] | np.ndarray,
    window_end_seconds: Sequence[float] | np.ndarray,
    sampled_frame_counts: Sequence[int] | np.ndarray,
) -> ShardArrays:
    raw_frame_counts = np.asarray(sampled_frame_counts)
    if (
        raw_frame_counts.ndim != 1
        or not np.issubdtype(raw_frame_counts.dtype, np.integer)
        or np.issubdtype(raw_frame_counts.dtype, np.bool_)
        or np.any(raw_frame_counts < 1)
        or np.any(raw_frame_counts > MAX_SAMPLED_FRAMES)
    ):
        raise ValueError(
            "sampled frame counts must be integer values between 1 and "
            f"{MAX_SAMPLED_FRAMES}"
        )
    return ShardArrays(
        video_ids=np.asarray(video_ids, dtype=np.str_),
        embeddings=np.asarray(embeddings, dtype=np.float32),
        source_duration_seconds=np.asarray(source_duration_seconds, dtype=np.float64),
        window_start_seconds=np.asarray(window_start_seconds, dtype=np.float64),
        window_end_seconds=np.asarray(window_end_seconds, dtype=np.float64),
        sampled_frame_counts=raw_frame_counts.astype(np.uint16, copy=False),
    )


def validate_shard_arrays(arrays: ShardArrays, *, source: str = "LCO shard") -> None:
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
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4):
        raise ValueError(f"{source}: embeddings must be L2-normalized")

    float_fields = (
        ("source_duration_seconds", arrays.source_duration_seconds),
        ("window_start_seconds", arrays.window_start_seconds),
        ("window_end_seconds", arrays.window_end_seconds),
    )
    for name, values in float_fields:
        if values.dtype != np.float64 or values.shape != (count,):
            raise ValueError(f"{source}: {name} must be a float64 vector of length N")
        if not np.isfinite(values).all():
            raise ValueError(f"{source}: {name} contains non-finite values")

    durations = arrays.source_duration_seconds
    starts = arrays.window_start_seconds
    ends = arrays.window_end_seconds
    if np.any(durations <= 0):
        raise ValueError(f"{source}: source durations must be positive")
    if np.any(starts < 0) or np.any(ends <= starts):
        raise ValueError(f"{source}: invalid analysis window bounds")
    if np.any(ends > durations + 1e-6):
        raise ValueError(f"{source}: analysis windows exceed source durations")

    frame_counts = arrays.sampled_frame_counts
    if frame_counts.dtype != np.uint16 or frame_counts.shape != (count,):
        raise ValueError(
            f"{source}: sampled_frame_counts must be a uint16 vector of length N"
        )
    if np.any(frame_counts == 0) or np.any(frame_counts > MAX_SAMPLED_FRAMES):
        raise ValueError(
            f"{source}: sampled frame counts must be between 1 and "
            f"{MAX_SAMPLED_FRAMES}"
        )


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
                source_duration_seconds=np.array(
                    data["source_duration_seconds"], copy=True
                ),
                window_start_seconds=np.array(data["window_start_seconds"], copy=True),
                window_end_seconds=np.array(data["window_end_seconds"], copy=True),
                sampled_frame_counts=np.array(data["sampled_frame_counts"], copy=True),
            )
    except (OSError, ValueError, KeyError) as error:
        if isinstance(error, ValueError) and str(error).startswith("invalid arrays"):
            raise
        raise ValueError(f"could not read LCO shard {path}: {error}") from error
    validate_shard_arrays(arrays, source=path.name)
    return arrays


def load_state(output_dir: Path, config: Mapping[str, Any]) -> StoredState:
    output_dir = Path(output_dir)
    manifest = load_manifest(output_dir)
    if manifest is None:
        return StoredState(create_manifest(config), set(), 0)

    validate_compatible(manifest, config)
    completed: set[str] = set()
    indexes: set[int] = set()
    filenames: set[str] = set()
    max_index = -1
    for entry in manifest["shards"]:
        index, filename, expected_count = _validate_shard_entry(entry)
        if index in indexes or filename in filenames:
            raise ValueError("duplicate entries in LCO shard inventory")
        indexes.add(index)
        filenames.add(filename)
        shard_path = output_dir / filename
        if not shard_path.is_file():
            raise ValueError(f"manifest references missing LCO shard: {shard_path}")
        arrays = read_shard(shard_path)
        if arrays.count != expected_count:
            raise ValueError(
                f"manifest count does not match {filename}: "
                f"{expected_count} != {arrays.count}"
            )
        shard_ids = set(arrays.video_ids.tolist())
        overlap = completed.intersection(shard_ids)
        if overlap:
            raise ValueError(
                "duplicate video IDs across LCO shards: "
                + ", ".join(sorted(overlap))
            )
        completed.update(shard_ids)
        max_index = max(max_index, index)
    return StoredState(manifest, completed, max_index + 1)


def write_shard(
    output_dir: Path,
    index: int,
    video_ids: Sequence[str] | np.ndarray,
    embeddings: np.ndarray,
    source_duration_seconds: Sequence[float] | np.ndarray,
    window_start_seconds: Sequence[float] | np.ndarray,
    window_end_seconds: Sequence[float] | np.ndarray,
    sampled_frame_counts: Sequence[int] | np.ndarray,
) -> dict[str, Any]:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("LCO shard index must be a non-negative integer")
    arrays = _coerce_shard_arrays(
        video_ids,
        embeddings,
        source_duration_seconds,
        window_start_seconds,
        window_end_seconds,
        sampled_frame_counts,
    )
    filename = f"lco-shard-{index:05d}.npz"
    validate_shard_arrays(arrays, source=filename)
    atomic_npz(
        Path(output_dir) / filename,
        video_ids=arrays.video_ids,
        embeddings=arrays.embeddings,
        source_duration_seconds=arrays.source_duration_seconds,
        window_start_seconds=arrays.window_start_seconds,
        window_end_seconds=arrays.window_end_seconds,
        sampled_frame_counts=arrays.sampled_frame_counts,
    )
    return {"index": index, "file": filename, "count": arrays.count}


def add_shard(manifest: dict[str, Any], shard: Mapping[str, Any]) -> None:
    _validate_manifest_header(manifest)
    index, filename, _ = _validate_shard_entry(dict(shard))
    for existing in manifest["shards"]:
        existing_index, existing_filename, _ = _validate_shard_entry(existing)
        if existing_index == index or existing_filename == filename:
            raise ValueError(f"LCO shard {filename} is already in the manifest")
    manifest["shards"].append(dict(shard))


def checkpoint_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    _validate_manifest_header(manifest)
    # Validate inventory structure before making it durable. Deep shard validation is
    # deliberately performed by load_state so checkpointing does not reread arrays.
    seen_indexes: set[int] = set()
    seen_files: set[str] = set()
    for entry in manifest["shards"]:
        index, filename, _ = _validate_shard_entry(entry)
        if index in seen_indexes or filename in seen_files:
            raise ValueError("duplicate entries in LCO shard inventory")
        seen_indexes.add(index)
        seen_files.add(filename)
        if not (Path(output_dir) / filename).is_file():
            raise ValueError(f"cannot checkpoint missing LCO shard: {filename}")
    consolidated = manifest.get("consolidated")
    if consolidated is not None:
        _validate_consolidated_entry(consolidated)
    manifest["updated_at"] = utc_now()
    atomic_json(Path(output_dir) / MANIFEST_NAME, manifest)


def _validate_consolidated_entry(entry: Any) -> tuple[str, int]:
    if not isinstance(entry, dict):
        raise ValueError("invalid consolidated artifact in LCO manifest")
    filename = entry.get("file")
    count = entry.get("count")
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or filename == MANIFEST_NAME
        or _SHARD_NAME.fullmatch(filename) is not None
    ):
        raise ValueError("invalid consolidated artifact filename in LCO manifest")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("invalid consolidated artifact count in LCO manifest")
    return filename, count


def reset_generated_output(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(output_dir)
    if manifest is None:
        return
    shard_filenames: list[str] = []
    for entry in manifest["shards"]:
        _, filename, _ = _validate_shard_entry(entry)
        shard_filenames.append(filename)

    consolidated = manifest.get("consolidated")
    consolidated_filename: str | None = None
    if consolidated is not None:
        consolidated_filename, _ = _validate_consolidated_entry(consolidated)

    # Validate every deletion target before touching any artifact.
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
        raise ValueError(f"no LCO manifest found in {output_dir}")

    try:
        destination_is_local = destination.resolve().parent == output_dir.resolve()
    except OSError:
        destination_is_local = False
    if destination_is_local and (
        destination.name == MANIFEST_NAME
        or _SHARD_NAME.fullmatch(destination.name) is not None
    ):
        raise ValueError(
            "LCO consolidation destination must not replace the manifest or a shard"
        )

    all_ids: list[np.ndarray] = []
    all_embeddings: list[np.ndarray] = []
    all_durations: list[np.ndarray] = []
    all_starts: list[np.ndarray] = []
    all_ends: list[np.ndarray] = []
    all_frame_counts: list[np.ndarray] = []
    seen: set[str] = set()
    seen_indexes: set[int] = set()
    validated_entries = [
        (_validate_shard_entry(entry), entry) for entry in manifest["shards"]
    ]
    for (index, filename, expected_count), _ in sorted(
        validated_entries, key=lambda item: item[0][0]
    ):
        if index in seen_indexes:
            raise ValueError("duplicate entries in LCO shard inventory")
        seen_indexes.add(index)
        arrays = read_shard(output_dir / filename)
        if arrays.count != expected_count:
            raise ValueError(f"manifest count does not match {filename}")
        ids = arrays.video_ids.tolist()
        overlap = seen.intersection(ids)
        if overlap:
            raise ValueError(
                "duplicate video IDs found while consolidating LCO shards: "
                + ", ".join(sorted(overlap))
            )
        seen.update(ids)
        all_ids.append(arrays.video_ids)
        all_embeddings.append(arrays.embeddings)
        all_durations.append(arrays.source_duration_seconds)
        all_starts.append(arrays.window_start_seconds)
        all_ends.append(arrays.window_end_seconds)
        all_frame_counts.append(arrays.sampled_frame_counts)

    if not all_ids:
        raise ValueError("no completed LCO embeddings to consolidate")

    ids = np.concatenate(all_ids)
    order = np.argsort(ids, kind="stable")
    atomic_npz(
        destination,
        video_ids=ids[order],
        embeddings=np.concatenate(all_embeddings, axis=0)[order],
        source_duration_seconds=np.concatenate(all_durations)[order],
        window_start_seconds=np.concatenate(all_starts)[order],
        window_end_seconds=np.concatenate(all_ends)[order],
        sampled_frame_counts=np.concatenate(all_frame_counts)[order],
    )

    # Only track an artifact that is directly inside the output directory. External
    # destinations must never become deletion targets of --overwrite.
    if destination_is_local:
        manifest["consolidated"] = {"file": destination.name, "count": len(ids)}
        checkpoint_manifest(output_dir, manifest)
    return len(ids)
