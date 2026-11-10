"""Compatibility checks between optional pipelines and generic MAVIC probing I/O."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from mavic.io import load_representation
from mavic.representations.f2llm import persistence as f2llm
from mavic.representations.lco import persistence as lco
from mavic.representations.openclip import persistence as openclip


def _unit_rows(count: int, dimension: int) -> np.ndarray:
    values = np.zeros((count, dimension), dtype=np.float32)
    values[np.arange(count), np.arange(count)] = 1.0
    return values


def test_openclip_manifest_is_accepted_by_generic_loader(tmp_path: Path) -> None:
    output = tmp_path / "openclip"
    state = openclip.load_state(output, {"model": "mock"})
    shard = openclip.write_shard(
        output,
        0,
        ["video-b", "video-a"],
        _unit_rows(2, 4),
        np.asarray([16, 12], dtype=np.uint8),
    )
    state.manifest["shards"].append(shard)
    openclip.atomic_json(output / openclip.MANIFEST_NAME, state.manifest)

    loaded = load_representation("openclip", output)

    assert loaded.video_ids.tolist() == ["video-b", "video-a"]
    assert loaded.embeddings.shape == (2, 4)


def test_lco_manifest_and_consolidated_file_are_generic_inputs(tmp_path: Path) -> None:
    output = tmp_path / "lco"
    manifest = lco.create_manifest({"model": "mock"})
    shard = lco.write_shard(
        output,
        0,
        ["video-b", "video-a"],
        _unit_rows(2, lco.EMBEDDING_DIMENSION),
        [20.0, 10.0],
        [2.0, 0.0],
        [18.0, 10.0],
        [32, 20],
    )
    lco.add_shard(manifest, shard)
    lco.checkpoint_manifest(output, manifest)

    from_manifest = load_representation("lco", output)
    assert from_manifest.video_ids.tolist() == ["video-b", "video-a"]

    destination = output / lco.DEFAULT_CONSOLIDATED_NAME
    assert lco.consolidate(output, destination) == 2
    consolidated = load_representation("lco", destination)
    assert consolidated.video_ids.tolist() == ["video-a", "video-b"]
    assert consolidated.embeddings.shape == (2, lco.EMBEDDING_DIMENSION)


def test_f2llm_manifest_and_consolidated_file_are_generic_inputs(tmp_path: Path) -> None:
    output = tmp_path / "f2llm"
    manifest = f2llm.create_manifest({"model": "mock"})
    hashes = [
        hashlib.sha256(value.encode("utf-8")).hexdigest()
        for value in ("metadata b", "metadata a")
    ]
    shard = f2llm.write_shard(
        output,
        0,
        ["video-b", "video-a"],
        _unit_rows(2, f2llm.EMBEDDING_DIMENSION),
        [100, 5000],
        [False, True],
        hashes,
    )
    f2llm.add_shard(manifest, shard)
    f2llm.checkpoint_manifest(output, manifest)

    from_manifest = load_representation("f2llm", output)
    assert from_manifest.video_ids.tolist() == ["video-b", "video-a"]

    destination = output / f2llm.DEFAULT_CONSOLIDATED_NAME
    assert f2llm.consolidate(output, destination) == 2
    consolidated = load_representation("f2llm", destination)
    assert consolidated.video_ids.tolist() == ["video-a", "video-b"]
    assert consolidated.embeddings.shape == (2, f2llm.EMBEDDING_DIMENSION)
