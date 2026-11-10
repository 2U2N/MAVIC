import json
from pathlib import Path

import numpy as np
import pytest

from mavic.representations.lco.persistence import (
    EMBEDDING_DIMENSION,
    MANIFEST_NAME,
    PIPELINE_ID,
    SCHEMA_VERSION,
    add_shard,
    checkpoint_manifest,
    consolidate,
    create_manifest,
    load_manifest,
    load_state,
    read_shard,
    reset_generated_output,
    validate_compatible,
    write_shard,
)


def _embeddings(rows: int, offset: int = 0) -> np.ndarray:
    values = np.zeros((rows, EMBEDDING_DIMENSION), dtype=np.float32)
    for row in range(rows):
        values[row, offset + row] = 1.0
    return values


def _write(
    output_dir: Path,
    index: int,
    ids: list[str],
    *,
    offset: int = 0,
) -> dict:
    count = len(ids)
    return write_shard(
        output_dir,
        index,
        ids,
        _embeddings(count, offset),
        np.asarray([20.0 + value for value in range(count)], dtype=np.float64),
        np.asarray([2.0 + value for value in range(count)], dtype=np.float64),
        np.asarray([18.0 + value for value in range(count)], dtype=np.float64),
        np.asarray([32 - value for value in range(count)], dtype=np.uint16),
    )


def test_manifest_is_pipeline_specific_and_configuration_is_exact(tmp_path: Path) -> None:
    manifest = create_manifest({"model_revision": "abc", "fps": 2})
    checkpoint_manifest(tmp_path, manifest)

    loaded = load_manifest(tmp_path)
    assert loaded is not None
    assert loaded["pipeline_id"] == PIPELINE_ID
    assert loaded["schema_version"] == SCHEMA_VERSION
    validate_compatible(loaded, {"model_revision": "abc", "fps": 2})

    with pytest.raises(ValueError, match="incompatible"):
        validate_compatible(loaded, {"model_revision": "def", "fps": 2})

    loaded["pipeline_id"] = "openclip-vit-b32-openai"
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(loaded), encoding="utf-8")
    with pytest.raises(ValueError, match="does not belong"):
        load_manifest(tmp_path)


def test_state_validates_shards_and_only_shard_ids_are_completed(tmp_path: Path) -> None:
    config = {"fps": 2}
    shard = _write(tmp_path, 3, ["done-a", "done-b"])
    manifest = create_manifest(config)
    manifest["errors"] = {"retry-me": {"stage": "ffmpeg", "message": "bad"}}
    add_shard(manifest, shard)
    checkpoint_manifest(tmp_path, manifest)

    state = load_state(tmp_path, config)

    assert state.completed_ids == {"done-a", "done-b"}
    assert "retry-me" not in state.completed_ids
    assert state.next_shard_index == 4


def test_write_and_read_enforce_lco_array_schema(tmp_path: Path) -> None:
    shard = _write(tmp_path, 0, ["id-a"])
    assert shard == {"index": 0, "file": "lco-shard-00000.npz", "count": 1}

    arrays = read_shard(tmp_path / shard["file"])
    assert arrays.video_ids.tolist() == ["id-a"]
    assert arrays.embeddings.dtype == np.float32
    assert arrays.embeddings.shape == (1, EMBEDDING_DIMENSION)
    assert arrays.source_duration_seconds.dtype == np.float64
    assert arrays.sampled_frame_counts.dtype == np.uint16

    bad = _embeddings(1)
    bad[0, 0] = 2.0
    with pytest.raises(ValueError, match="L2-normalized"):
        write_shard(
            tmp_path,
            1,
            ["bad"],
            bad,
            [20.0],
            [2.0],
            [18.0],
            [32],
        )

    with pytest.raises(ValueError, match="frame counts"):
        write_shard(
            tmp_path,
            1,
            ["bad"],
            _embeddings(1),
            [20.0],
            [2.0],
            [18.0],
            [33],
        )

    with pytest.raises(ValueError, match="frame counts"):
        write_shard(
            tmp_path,
            1,
            ["bad"],
            _embeddings(1),
            [20.0],
            [2.0],
            [18.0],
            [65_537],
        )


def test_corrupt_shard_dtype_and_missing_arrays_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.npz"
    np.savez(
        path,
        video_ids=np.asarray(["id"], dtype=np.str_),
        embeddings=_embeddings(1).astype(np.float64),
        source_duration_seconds=np.asarray([20.0], dtype=np.float64),
        window_start_seconds=np.asarray([2.0], dtype=np.float64),
        window_end_seconds=np.asarray([18.0], dtype=np.float64),
        sampled_frame_counts=np.asarray([32], dtype=np.uint16),
    )
    with pytest.raises(ValueError, match="dtype float32"):
        read_shard(path)

    np.savez(path, video_ids=np.asarray(["id"], dtype=np.str_))
    with pytest.raises(ValueError, match="invalid arrays"):
        read_shard(path)


def test_load_state_rejects_missing_count_mismatch_and_cross_shard_duplicates(
    tmp_path: Path,
) -> None:
    config = {"fps": 2}
    manifest = create_manifest(config)
    manifest["shards"] = [
        {"index": 0, "file": "lco-shard-00000.npz", "count": 1}
    ]
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="missing LCO shard"):
        load_state(tmp_path, config)

    first = _write(tmp_path, 0, ["same"])
    second = _write(tmp_path, 1, ["same"], offset=1)
    manifest["shards"] = [first, second]
    checkpoint_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="across LCO shards"):
        load_state(tmp_path, config)

    manifest["shards"] = [{**first, "count": 2}]
    checkpoint_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="count does not match"):
        load_state(tmp_path, config)


def test_consolidation_sorts_ids_and_preserves_every_array_exactly(
    tmp_path: Path,
) -> None:
    manifest = create_manifest({"fps": 2})
    add_shard(manifest, _write(tmp_path, 0, ["z"], offset=0))
    add_shard(manifest, _write(tmp_path, 1, ["a"], offset=1))
    checkpoint_manifest(tmp_path, manifest)
    destination = tmp_path / "lco-embeddings.npz"

    assert consolidate(tmp_path, destination) == 2

    with np.load(destination, allow_pickle=False) as data:
        assert set(data.files) == {
            "video_ids",
            "embeddings",
            "source_duration_seconds",
            "window_start_seconds",
            "window_end_seconds",
            "sampled_frame_counts",
        }
        assert data["video_ids"].tolist() == ["a", "z"]
        np.testing.assert_array_equal(
            data["embeddings"],
            np.concatenate([_embeddings(1, offset=1), _embeddings(1, offset=0)]),
        )
        np.testing.assert_array_equal(data["source_duration_seconds"], [20.0, 20.0])
        np.testing.assert_array_equal(data["window_start_seconds"], [2.0, 2.0])
        np.testing.assert_array_equal(data["window_end_seconds"], [18.0, 18.0])
        np.testing.assert_array_equal(data["sampled_frame_counts"], [32, 32])

    loaded = load_manifest(tmp_path)
    assert loaded is not None
    assert loaded["consolidated"] == {"file": destination.name, "count": 2}


def test_reset_deletes_only_manifest_referenced_lco_artifacts(tmp_path: Path) -> None:
    referenced = _write(tmp_path, 0, ["id"])
    unreferenced = tmp_path / "lco-shard-99999.npz"
    unrelated_clip = tmp_path / "shard-00000.npz"
    unreferenced.write_bytes(b"keep")
    unrelated_clip.write_bytes(b"keep")

    manifest = create_manifest({"fps": 2})
    add_shard(manifest, referenced)
    checkpoint_manifest(tmp_path, manifest)
    consolidated = tmp_path / "lco-embeddings.npz"
    consolidate(tmp_path, consolidated)

    reset_generated_output(tmp_path)

    assert not (tmp_path / referenced["file"]).exists()
    assert not consolidated.exists()
    assert not (tmp_path / MANIFEST_NAME).exists()
    assert unreferenced.read_bytes() == b"keep"
    assert unrelated_clip.read_bytes() == b"keep"


@pytest.mark.parametrize("destination_name", [MANIFEST_NAME, "lco-shard-00000.npz"])
def test_consolidation_cannot_replace_source_state(
    tmp_path: Path, destination_name: str
) -> None:
    shard = _write(tmp_path, 0, ["id"])
    manifest = create_manifest({"fps": 2})
    add_shard(manifest, shard)
    checkpoint_manifest(tmp_path, manifest)
    manifest_before = (tmp_path / MANIFEST_NAME).read_bytes()
    shard_before = (tmp_path / shard["file"]).read_bytes()

    with pytest.raises(ValueError, match="must not replace"):
        consolidate(tmp_path, tmp_path / destination_name)

    assert (tmp_path / MANIFEST_NAME).read_bytes() == manifest_before
    assert (tmp_path / shard["file"]).read_bytes() == shard_before


def test_reset_refuses_to_touch_a_clip_manifest(tmp_path: Path) -> None:
    clip_manifest = {
        "schema_version": 1,
        "config": {},
        "shards": [{"index": 0, "file": "shard-00000.npz", "count": 1}],
    }
    shard = tmp_path / "shard-00000.npz"
    shard.write_bytes(b"clip")
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(clip_manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="does not belong"):
        reset_generated_output(tmp_path)
    assert shard.read_bytes() == b"clip"
    assert (tmp_path / MANIFEST_NAME).exists()
