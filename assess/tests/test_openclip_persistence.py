import json
from pathlib import Path

import numpy as np
import pytest

from mavic.representations.openclip.persistence import (
    MANIFEST_NAME,
    SCHEMA_VERSION,
    atomic_json,
    consolidate,
    load_state,
    validate_compatible,
    write_shard,
)


def _manifest(config: dict, shards: list[dict] | None = None) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "config": config,
        "created_at": "now",
        "updated_at": "now",
        "shards": shards or [],
        "errors": {},
        "stats": {},
        "last_run": {},
        "runs": [],
    }


def test_manifest_compatibility_is_exact() -> None:
    validate_compatible(_manifest({"samples": 16}), {"samples": 16})
    with pytest.raises(ValueError, match="incompatible"):
        validate_compatible(_manifest({"samples": 16}), {"samples": 8})


def test_load_state_recovers_completed_ids_and_next_index(tmp_path: Path) -> None:
    config = {"samples": 16}
    shard = write_shard(
        tmp_path,
        3,
        ["id-a", "id-b"],
        np.eye(2, dtype=np.float32),
        np.array([1, 2], dtype=np.uint8),
    )
    atomic_json(tmp_path / MANIFEST_NAME, _manifest(config, [shard]))

    state = load_state(tmp_path, config)

    assert state.completed_ids == {"id-a", "id-b"}
    assert state.next_shard_index == 4


def test_consolidate_sorts_ids_and_preserves_values(tmp_path: Path) -> None:
    config = {"samples": 16}
    first = write_shard(
        tmp_path,
        0,
        ["z"],
        np.array([[0.0, 1.0]], dtype=np.float32),
        np.array([2], dtype=np.uint8),
    )
    second = write_shard(
        tmp_path,
        1,
        ["a"],
        np.array([[1.0, 0.0]], dtype=np.float32),
        np.array([1], dtype=np.uint8),
    )
    atomic_json(tmp_path / MANIFEST_NAME, _manifest(config, [first, second]))
    destination = tmp_path / "all.npz"

    assert consolidate(tmp_path, destination) == 2
    with np.load(destination, allow_pickle=False) as data:
        assert data["video_ids"].tolist() == ["a", "z"]
        np.testing.assert_array_equal(data["embeddings"], [[1.0, 0.0], [0.0, 1.0]])
        assert data["retained_frame_counts"].tolist() == [1, 2]


def test_missing_shard_is_rejected(tmp_path: Path) -> None:
    manifest = _manifest(
        {"samples": 16}, [{"index": 0, "file": "missing.npz", "count": 1}]
    )
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="missing shard"):
        load_state(tmp_path, {"samples": 16})
