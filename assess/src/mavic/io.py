"""Strict input/output helpers for MAVIC feature and representation tables."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd


VIDEO_ID_COLUMN = "video_id"
REPRESENTATION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class FeatureTable:
    """A validated feature matrix aligned to unique video IDs."""

    video_ids: np.ndarray
    feature_names: tuple[str, ...]
    values: np.ndarray


@dataclass(frozen=True)
class Representation:
    """A validated embedding matrix aligned to unique video IDs."""

    name: str
    video_ids: np.ndarray
    embeddings: np.ndarray
    source: Path


def _validate_ids(ids: np.ndarray, *, source: Path) -> np.ndarray:
    ids = np.asarray(ids)
    if ids.ndim != 1:
        raise ValueError(f"{source}: video_ids must be one-dimensional")
    if ids.dtype.kind not in {"U", "S"}:
        # Object arrays require pickle and are intentionally unsupported.
        ids = ids.astype(np.str_)
    else:
        ids = ids.astype(np.str_)
    if len(ids) == 0:
        raise ValueError(f"{source}: no video IDs")
    if any(not value for value in ids.tolist()):
        raise ValueError(f"{source}: video IDs must be non-empty")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError(f"{source}: duplicate video IDs")
    return ids


def load_feature_table(path: Path | str) -> FeatureTable:
    """Load a CSV with a ``video_id`` column and finite numeric features."""
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"feature CSV not found: {source}")
    try:
        frame = pd.read_csv(source, dtype={VIDEO_ID_COLUMN: "string"})
    except (OSError, pd.errors.ParserError) as error:
        raise ValueError(f"could not read feature CSV {source}: {error}") from error
    if frame.columns.duplicated().any():
        raise ValueError(f"{source}: duplicate column names")
    if VIDEO_ID_COLUMN not in frame.columns:
        raise ValueError(f"{source}: missing {VIDEO_ID_COLUMN!r} column")
    if frame[VIDEO_ID_COLUMN].isna().any():
        raise ValueError(f"{source}: video IDs may not be missing")
    ids = _validate_ids(frame[VIDEO_ID_COLUMN].astype(str).to_numpy(), source=source)
    names = tuple(column for column in frame.columns if column != VIDEO_ID_COLUMN)
    if not names:
        raise ValueError(f"{source}: no feature columns")
    try:
        numeric = frame.loc[:, names].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source}: all feature values must be numeric") from error
    values = numeric.to_numpy(dtype=np.float64)
    if values.ndim != 2 or values.shape != (len(ids), len(names)):
        raise ValueError(f"{source}: invalid feature matrix shape")
    if not np.isfinite(values).all():
        raise ValueError(f"{source}: feature values must be complete and finite")
    return FeatureTable(ids, names, values)


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {"video_ids", "embeddings"}
            missing = sorted(required - set(archive.files))
            if missing:
                raise ValueError(
                    f"{path}: missing arrays: {', '.join(missing)}"
                )
            ids = np.asarray(archive["video_ids"])
            embeddings = np.asarray(archive["embeddings"])
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith(str(path)):
            raise
        raise ValueError(f"could not read representation archive {path}: {error}") from error
    return ids, embeddings


def _validate_embedding_arrays(
    ids: np.ndarray, embeddings: np.ndarray, *, source: Path
) -> tuple[np.ndarray, np.ndarray]:
    ids = _validate_ids(ids, source=source)
    matrix = np.asarray(embeddings)
    if matrix.ndim != 2:
        raise ValueError(f"{source}: embeddings must be two-dimensional")
    if matrix.shape[0] != len(ids):
        raise ValueError(f"{source}: embeddings must align with video_ids")
    if matrix.shape[1] == 0:
        raise ValueError(f"{source}: embeddings must have at least one dimension")
    if not np.issubdtype(matrix.dtype, np.number):
        raise ValueError(f"{source}: embeddings must be numeric")
    matrix = matrix.astype(np.float32, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{source}: embeddings contain non-finite values")
    return ids, matrix


def _load_manifest_directory(path: Path) -> tuple[np.ndarray, np.ndarray]:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"representation directory has no manifest.json: {path}")
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {manifest_path}: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path}: manifest must be an object")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"{manifest_path}: manifest contains no shards")
    if any(not isinstance(item, dict) for item in shards):
        raise ValueError(f"{manifest_path}: shard entries must be objects")
    indices_unsorted = [item.get("index") for item in shards]
    if any(
        isinstance(index, bool) or not isinstance(index, int)
        for index in indices_unsorted
    ):
        raise ValueError(f"{manifest_path}: shard indices must be integers")
    ordered = sorted(shards, key=lambda item: item["index"])
    indices = [item["index"] for item in ordered]
    if indices != list(range(len(ordered))):
        raise ValueError(f"{manifest_path}: shard indices must be contiguous from zero")

    id_parts: list[np.ndarray] = []
    matrix_parts: list[np.ndarray] = []
    seen: set[str] = set()
    dimension: int | None = None
    for item in ordered:
        filename = item.get("file")
        count = item.get("count")
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"{manifest_path}: invalid shard filename")
        if Path(filename).name != filename:
            raise ValueError(f"{manifest_path}: shard filenames must be basenames")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
        ):
            raise ValueError(f"{manifest_path}: invalid shard count for {filename}")
        shard_path = path / filename
        if not shard_path.is_file():
            raise ValueError(f"{manifest_path}: missing shard {filename}")
        ids, matrix = _validate_embedding_arrays(
            *_load_npz(shard_path), source=shard_path
        )
        if len(ids) != count:
            raise ValueError(
                f"{manifest_path}: count mismatch for {filename}: {count} != {len(ids)}"
            )
        overlap = sorted(seen.intersection(ids.tolist()))
        if overlap:
            raise ValueError(
                f"{manifest_path}: duplicate video IDs across shards: "
                + ", ".join(overlap)
            )
        seen.update(ids.tolist())
        if dimension is None:
            dimension = matrix.shape[1]
        elif matrix.shape[1] != dimension:
            raise ValueError(f"{manifest_path}: embedding dimensions differ across shards")
        id_parts.append(ids)
        matrix_parts.append(matrix)
    return np.concatenate(id_parts), np.concatenate(matrix_parts, axis=0)


def validate_representation_name(name: str) -> str:
    """Validate a CLI-safe, output-safe representation label."""
    if not isinstance(name, str) or not REPRESENTATION_NAME.fullmatch(name):
        raise ValueError(
            "representation names must start with an alphanumeric character and "
            "contain only letters, digits, '.', '_' or '-'"
        )
    return name


def load_representation(name: str, path: Path | str) -> Representation:
    """Load a consolidated NPZ or a compatible manifest/shard directory."""
    name = validate_representation_name(name)
    source = Path(path)
    if source.is_file():
        if source.suffix.lower() != ".npz":
            raise ValueError(f"{source}: consolidated representations must be .npz files")
        ids, matrix = _validate_embedding_arrays(*_load_npz(source), source=source)
    elif source.is_dir():
        ids, matrix = _load_manifest_directory(source)
    else:
        raise ValueError(f"representation input not found: {source}")
    return Representation(name, ids, matrix, source)


def parse_representation_specs(specs: Iterable[str]) -> list[tuple[str, Path]]:
    """Parse repeatable ``NAME=PATH`` command-line arguments."""
    parsed: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"invalid representation {spec!r}; expected NAME=NPZ_OR_MANIFEST_DIR"
            )
        name, raw_path = spec.split("=", 1)
        validate_representation_name(name)
        if not raw_path:
            raise ValueError(f"representation {name!r} has an empty path")
        if name in seen:
            raise ValueError(f"duplicate representation name: {name}")
        seen.add(name)
        parsed.append((name, Path(raw_path)))
    if not parsed:
        raise ValueError("at least one --representation is required")
    return parsed
