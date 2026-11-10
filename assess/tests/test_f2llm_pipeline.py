from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import mavic.representations.f2llm.cli as f2llm_cli
import mavic.representations.f2llm.pipeline as pipeline
from mavic.representations.f2llm.embedding import EMBEDDING_DIMENSION, F2LLMBatchResult
from mavic.representations.f2llm.persistence import consolidate, load_manifest
from mavic.representations.f2llm.pipeline import F2LLMRunOptions, run_f2llm_pipeline


def _sidecar(path: Path, video_id: str, *, title: str | None = None) -> Path:
    target = path / f"{video_id}.info.json"
    target.write_text(
        json.dumps(
            {
                "id": video_id,
                "title": title or f"Title {video_id}",
                "description": f"Description {video_id}",
                "categories": ["Category"],
                "tags": ["tag"],
            }
        ),
        encoding="utf-8",
    )
    return target


def _result(count: int) -> F2LLMBatchResult:
    embeddings = np.zeros((count, EMBEDDING_DIMENSION), dtype=np.float32)
    for index in range(count):
        embeddings[index, index] = 1.0
    return F2LLMBatchResult(
        embeddings,
        np.asarray([20 + index for index in range(count)], dtype=np.uint32),
        np.asarray([False] * count, dtype=np.bool_),
    )


class _FakeEmbedder:
    def __init__(self, calls: list[list[str]]) -> None:
        self.calls = calls

    def embed(self, texts):
        self.calls.append(list(texts))
        return _result(len(texts))


@pytest.fixture(autouse=True)
def _stable_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pipeline,
        "_runtime_provenance",
        lambda: {"pipeline_id": "f2llm-v2-4b-metadata", "test": True},
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def test_suffix_matching_missing_sidecar_batching_resume_and_consolidation(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    for name in ("a.mp4", "source.f633.mp4", "missing.mp4"):
        (input_dir / name).touch()
    _sidecar(input_dir, "a")
    _sidecar(input_dir, "source")
    calls: list[list[str]] = []
    options = F2LLMRunOptions(input_dir, output_dir, batch_size=2, shard_size=1)

    manifest = run_f2llm_pipeline(
        options, embedder_factory=lambda: _FakeEmbedder(calls)
    )

    assert len(calls) == 1
    assert manifest["stats"] == {
        "discovered": 3,
        "selected": 3,
        "metadata_matched": 2,
        "succeeded": 2,
        "failed": 1,
        "pending": 0,
    }
    assert manifest["errors"]["missing"]["stage"] == "metadata"
    assert [item["file"] for item in manifest["shards"]] == [
        "f2llm-shard-00000.npz",
        "f2llm-shard-00001.npz",
    ]

    def must_not_load():
        raise AssertionError("completed records must not reload the model")

    repeated = run_f2llm_pipeline(options, embedder_factory=must_not_load)
    assert repeated["last_run"]["new_successes"] == 0
    destination = output_dir / "all.npz"
    assert consolidate(output_dir, destination) == 2
    with np.load(destination, allow_pickle=False) as data:
        assert data["video_ids"].tolist() == ["a", "source.f633"]


def test_changed_completed_metadata_requires_overwrite(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "id.mp4").touch()
    sidecar = _sidecar(input_dir, "id")
    options = F2LLMRunOptions(input_dir, output_dir)
    run_f2llm_pipeline(options, embedder_factory=lambda: _FakeEmbedder([]))

    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["title"] = "Changed"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="metadata text changed"):
        run_f2llm_pipeline(options, embedder_factory=lambda: _FakeEmbedder([]))


def test_batch_failure_falls_back_to_individual_records(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    for video_id in ("a", "b"):
        (input_dir / f"{video_id}.mp4").touch()
        _sidecar(input_dir, video_id)
    calls: list[int] = []

    class BatchFails:
        def embed(self, texts):
            calls.append(len(texts))
            if len(texts) > 1:
                raise ValueError("batch failed")
            return _result(1)

    manifest = run_f2llm_pipeline(
        F2LLMRunOptions(input_dir, output_dir, batch_size=2),
        embedder_factory=BatchFails,
    )
    assert calls == [2, 1, 1]
    assert manifest["stats"]["succeeded"] == 2
    assert manifest["stats"]["failed"] == 0


def test_cuda_oom_is_checkpointed_and_aborts(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "id.mp4").touch()
    _sidecar(input_dir, "id")

    class OomEmbedder:
        def embed(self, texts):
            raise torch.cuda.OutOfMemoryError("test OOM")

    with pytest.raises(torch.cuda.OutOfMemoryError, match="test OOM"):
        run_f2llm_pipeline(
            F2LLMRunOptions(input_dir, output_dir), embedder_factory=OomEmbedder
        )
    manifest = load_manifest(output_dir)
    assert manifest is not None
    assert manifest["errors"]["id"]["type"] == "OutOfMemoryError"
    assert manifest["stats"]["failed"] == 1


def test_cli_processing_and_default_consolidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[F2LLMRunOptions] = []

    def fake_run(options):
        seen.append(options)
        return {"stats": {"succeeded": 10}}

    monkeypatch.setattr(f2llm_cli, "run_f2llm_pipeline", fake_run)
    f2llm_cli.main(["input", "output", "--limit", "10"])
    assert seen == [
        F2LLMRunOptions(Path("input"), Path("output"), limit=10)
    ]
    assert json.loads(capsys.readouterr().out) == {"succeeded": 10}

    consolidated: list[tuple[Path, Path]] = []
    monkeypatch.setattr(
        f2llm_cli,
        "consolidate",
        lambda output, destination: consolidated.append((output, destination)) or 10,
    )
    f2llm_cli.main(["consolidate", "output"])
    assert consolidated == [
        (Path("output"), Path("output/f2llm-embeddings.npz"))
    ]
