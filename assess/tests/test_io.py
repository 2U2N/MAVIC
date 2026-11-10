from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from mavic.io import (
    load_feature_table,
    load_representation,
    parse_representation_specs,
)


def test_load_feature_table_and_consolidated_npz(tmp_path) -> None:
    feature_path = tmp_path / "features.csv"
    pd.DataFrame(
        {"video_id": ["b", "a"], "one": [1.0, 2.0], "two": [3, 4]}
    ).to_csv(feature_path, index=False)
    table = load_feature_table(feature_path)
    assert table.feature_names == ("one", "two")
    assert table.values.dtype == np.float64

    representation_path = tmp_path / "embedding.npz"
    np.savez(
        representation_path,
        video_ids=np.asarray(["a", "b"]),
        embeddings=np.eye(2, dtype=np.float32),
    )
    representation = load_representation("model-1", representation_path)
    assert representation.embeddings.shape == (2, 2)


def test_feature_loader_preserves_opaque_leading_zero_ids(tmp_path) -> None:
    path = tmp_path / "features.csv"
    path.write_text("video_id,feature\n001,1\n010,2\n", encoding="utf-8")
    table = load_feature_table(path)
    assert table.video_ids.tolist() == ["001", "010"]


def test_manifest_loading_and_duplicate_detection(tmp_path) -> None:
    np.savez(
        tmp_path / "part-0.npz",
        video_ids=np.asarray(["a", "b"]),
        embeddings=np.ones((2, 3), dtype=np.float32),
    )
    np.savez(
        tmp_path / "part-1.npz",
        video_ids=np.asarray(["c"]),
        embeddings=np.zeros((1, 3), dtype=np.float32),
    )
    manifest = {
        "shards": [
            {"index": 1, "file": "part-1.npz", "count": 1},
            {"index": 0, "file": "part-0.npz", "count": 2},
        ]
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    representation = load_representation("manifest_model", tmp_path)
    assert representation.video_ids.tolist() == ["a", "b", "c"]
    assert representation.embeddings.shape == (3, 3)

    np.savez(
        tmp_path / "part-1.npz",
        video_ids=np.asarray(["b"]),
        embeddings=np.zeros((1, 3), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="duplicate video IDs across shards"):
        load_representation("manifest_model", tmp_path)


def test_representation_validation_errors(tmp_path) -> None:
    bad = tmp_path / "bad.npz"
    np.savez(
        bad,
        video_ids=np.asarray(["a", "a"]),
        embeddings=np.ones((2, 2), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_representation("model", bad)
    with pytest.raises(ValueError, match="expected NAME"):
        parse_representation_specs(["model"])
    with pytest.raises(ValueError, match="duplicate representation"):
        parse_representation_specs(["x=a.npz", "x=b.npz"])
