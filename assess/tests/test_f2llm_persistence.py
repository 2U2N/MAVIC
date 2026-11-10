from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mavic.representations.f2llm.embedding import EMBEDDING_DIMENSION
from mavic.representations.f2llm.persistence import (
    MANIFEST_NAME,
    PIPELINE_ID,
    add_shard,
    checkpoint_manifest,
    consolidate,
    create_manifest,
    load_manifest,
    load_state,
    read_shard,
    reset_generated_output,
    write_shard,
)


def _embeddings(count: int, offset: int = 0) -> np.ndarray:
    values = np.zeros((count, EMBEDDING_DIMENSION), dtype=np.float32)
    for index in range(count):
        values[index, offset + index] = 1.0
    return values


def _hash(character: str) -> str:
    return character * 64


def _write(path: Path, index: int, ids: list[str], offset: int = 0) -> dict:
    return write_shard(
        path,
        index,
        ids,
        _embeddings(len(ids), offset),
        np.arange(10, 10 + len(ids), dtype=np.uint32),
        np.asarray([False] * len(ids), dtype=np.bool_),
        [_hash(chr(ord("a") + offset + value)) for value in range(len(ids))],
    )


def test_write_read_state_and_consolidation_preserve_schema(tmp_path: Path) -> None:
    first = _write(tmp_path, 0, ["z"])
    second = _write(tmp_path, 1, ["a"], offset=1)
    manifest = create_manifest({"max_tokens": 4096})
    add_shard(manifest, first)
    add_shard(manifest, second)
    checkpoint_manifest(tmp_path, manifest)

    arrays = read_shard(tmp_path / first["file"])
    assert arrays.embeddings.shape == (1, EMBEDDING_DIMENSION)
    assert arrays.token_counts.dtype == np.uint32
    assert arrays.truncated.dtype == np.bool_

    state = load_state(tmp_path, {"max_tokens": 4096})
    assert state.completed_ids == {"a", "z"}
    assert state.completed_text_hashes == {"z": _hash("a"), "a": _hash("b")}
    assert state.next_shard_index == 2

    destination = tmp_path / "f2llm-embeddings.npz"
    assert consolidate(tmp_path, destination) == 2
    with np.load(destination, allow_pickle=False) as data:
        assert set(data.files) == {
            "video_ids",
            "embeddings",
            "token_counts",
            "truncated",
            "text_sha256",
        }
        assert data["video_ids"].tolist() == ["a", "z"]
        assert data["text_sha256"].tolist() == [_hash("b"), _hash("a")]
    loaded = load_manifest(tmp_path)
    assert loaded is not None
    assert loaded["pipeline_id"] == PIPELINE_ID
    assert loaded["consolidated"] == {"file": destination.name, "count": 2}


def test_shard_validation_rejects_bad_vectors_hashes_and_missing_arrays(
    tmp_path: Path,
) -> None:
    bad = _embeddings(1)
    bad[0, 0] = 2.0
    with pytest.raises(ValueError, match="L2-normalized"):
        write_shard(tmp_path, 0, ["id"], bad, [1], [False], [_hash("a")])
    with pytest.raises(ValueError, match="SHA-256"):
        write_shard(tmp_path, 0, ["id"], _embeddings(1), [1], [False], ["bad"])

    corrupt = tmp_path / "corrupt.npz"
    np.savez(corrupt, video_ids=np.asarray(["id"], dtype=np.str_))
    with pytest.raises(ValueError, match="invalid arrays"):
        read_shard(corrupt)


def test_state_rejects_wrong_manifest_missing_shard_and_duplicate_ids(
    tmp_path: Path,
) -> None:
    clip_manifest = {"schema_version": 1, "config": {}, "shards": []}
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(clip_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="does not belong"):
        load_manifest(tmp_path)

    (tmp_path / MANIFEST_NAME).unlink()
    first = _write(tmp_path, 0, ["same"])
    second = _write(tmp_path, 1, ["same"], offset=1)
    manifest = create_manifest({})
    manifest["shards"] = [first, second]
    checkpoint_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="duplicate video ID"):
        load_state(tmp_path, {})


def test_reset_only_deletes_referenced_f2llm_files(tmp_path: Path) -> None:
    shard = _write(tmp_path, 0, ["id"])
    manifest = create_manifest({})
    add_shard(manifest, shard)
    checkpoint_manifest(tmp_path, manifest)
    consolidate(tmp_path, tmp_path / "f2llm-embeddings.npz")
    unrelated = tmp_path / "shard-00000.npz"
    unrelated.write_bytes(b"clip")

    reset_generated_output(tmp_path)

    assert not (tmp_path / shard["file"]).exists()
    assert not (tmp_path / "f2llm-embeddings.npz").exists()
    assert not (tmp_path / MANIFEST_NAME).exists()
    assert unrelated.read_bytes() == b"clip"
