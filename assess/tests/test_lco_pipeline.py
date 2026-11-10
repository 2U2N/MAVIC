from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

import mavic.representations.lco.cli as lco_cli
import mavic.representations.lco.pipeline as pipeline
from mavic.representations.lco.embedding import EMBEDDING_DIMENSION, LCOEmbeddingResult
from mavic.representations.lco.persistence import consolidate, load_manifest
from mavic.representations.lco.pipeline import LCORunOptions, run_lco_pipeline


def _unit_embedding(index: int = 0) -> np.ndarray:
    embedding = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
    embedding[index] = 1.0
    return embedding


class _FakeEmbedder:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def embed(self, path: str | Path) -> LCOEmbeddingResult:
        self.calls.append(Path(path).stem)
        return LCOEmbeddingResult(_unit_embedding(len(self.calls) - 1), 12)


@contextmanager
def _prepared(path: Path):
    yield SimpleNamespace(
        clip_path=Path(f"{path.stem}-central.mp4"),
        source_duration_seconds=20.0,
        window_start_seconds=2.0,
        window_end_seconds=18.0,
    )


@pytest.fixture(autouse=True)
def _stable_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "_tool_version", lambda executable: f"{executable}-test")
    monkeypatch.setattr(
        pipeline,
        "_runtime_provenance",
        lambda: {"pipeline_id": "lco-embedding-omni-7b-av", "test": True},
    )
    monkeypatch.setattr(pipeline.torch.cuda, "is_available", lambda: False)


def test_limit_is_deterministic_and_repeat_skips_completed_ids(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    for name in ("z.mp4", "a.MP4", "m.mp4"):
        (input_dir / name).touch()
    (input_dir / "a.json").touch()
    calls: list[str] = []
    options = LCORunOptions(input_dir, output_dir, limit=2, shard_size=1)

    manifest = run_lco_pipeline(
        options,
        embedder_factory=lambda: _FakeEmbedder(calls),
        clip_preparer=_prepared,
    )

    assert calls == ["a-central", "m-central"]
    assert manifest["stats"] == {
        "discovered": 3,
        "selected": 2,
        "succeeded": 2,
        "failed": 0,
        "pending": 0,
    }
    assert [item["file"] for item in manifest["shards"]] == [
        "lco-shard-00000.npz",
        "lco-shard-00001.npz",
    ]

    def must_not_load():
        raise AssertionError("a completed repeat must not load the model")

    repeated = run_lco_pipeline(
        options,
        embedder_factory=must_not_load,
        clip_preparer=_prepared,
    )
    assert repeated["last_run"]["pending_at_start"] == 0
    assert repeated["last_run"]["skipped"] == 2

    destination = output_dir / "all.npz"
    assert consolidate(output_dir, destination) == 2
    with np.load(destination, allow_pickle=False) as data:
        assert data["video_ids"].tolist() == ["a", "m"]
        assert data["embeddings"].shape == (2, EMBEDDING_DIMENSION)


def test_failure_continues_and_failed_id_is_retried(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "bad.mp4").touch()
    (input_dir / "good.mp4").touch()

    @contextmanager
    def sometimes_bad(path: Path):
        if path.stem == "bad":
            raise ValueError("no decoded audio samples")
        with _prepared(path) as prepared:
            yield prepared

    first_calls: list[str] = []
    options = LCORunOptions(input_dir, output_dir, shard_size=10)
    first = run_lco_pipeline(
        options,
        embedder_factory=lambda: _FakeEmbedder(first_calls),
        clip_preparer=sometimes_bad,
    )

    assert first_calls == ["good-central"]
    assert first["stats"]["succeeded"] == 1
    assert first["stats"]["failed"] == 1
    assert first["errors"]["bad"]["stage"] == "media_preparation"

    retry_calls: list[str] = []
    second = run_lco_pipeline(
        options,
        embedder_factory=lambda: _FakeEmbedder(retry_calls),
        clip_preparer=_prepared,
    )
    assert retry_calls == ["bad-central"]
    assert second["stats"]["succeeded"] == 2
    assert second["stats"]["failed"] == 0
    assert second["errors"] == {}


def test_all_failed_is_recorded_then_returns_nonzero(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "bad.mp4").touch()

    @contextmanager
    def always_bad(path: Path):
        raise ValueError(f"cannot decode {path.name}")
        yield  # pragma: no cover

    with pytest.raises(RuntimeError, match="no videos were embedded successfully"):
        run_lco_pipeline(
            LCORunOptions(input_dir, output_dir),
            embedder_factory=lambda: _FakeEmbedder([]),
            clip_preparer=always_bad,
        )

    manifest = load_manifest(output_dir)
    assert manifest is not None
    assert manifest["stats"]["succeeded"] == 0
    assert manifest["stats"]["failed"] == 1
    assert manifest["errors"]["bad"]["type"] == "ValueError"


def test_invalid_embedding_is_an_isolated_inference_failure(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "bad.mp4").touch()

    class InvalidEmbedder:
        def embed(self, path: str | Path) -> LCOEmbeddingResult:
            return LCOEmbeddingResult(np.zeros(EMBEDDING_DIMENSION, dtype=np.float32), 4)

    with pytest.raises(RuntimeError, match="no videos were embedded successfully"):
        run_lco_pipeline(
            LCORunOptions(input_dir, output_dir),
            embedder_factory=InvalidEmbedder,
            clip_preparer=_prepared,
        )
    manifest = load_manifest(output_dir)
    assert manifest is not None
    assert manifest["errors"]["bad"]["stage"] == "inference"


def test_cuda_oom_is_recorded_and_aborts_without_processing_more_videos(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "a.mp4").touch()
    (input_dir / "b.mp4").touch()
    calls: list[str] = []

    class OomEmbedder:
        def embed(self, path: str | Path) -> LCOEmbeddingResult:
            calls.append(Path(path).stem)
            raise pipeline.torch.cuda.OutOfMemoryError("test OOM")

    with pytest.raises(pipeline.torch.cuda.OutOfMemoryError, match="test OOM"):
        run_lco_pipeline(
            LCORunOptions(input_dir, output_dir),
            embedder_factory=OomEmbedder,
            clip_preparer=_prepared,
        )

    assert calls == ["a-central"]
    manifest = load_manifest(output_dir)
    assert manifest is not None
    assert manifest["errors"]["a"]["type"] == "OutOfMemoryError"
    assert manifest["stats"] == {
        "discovered": 2,
        "selected": 2,
        "succeeded": 0,
        "failed": 1,
        "pending": 1,
    }


def test_cli_processing_and_default_consolidation_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[LCORunOptions] = []

    def fake_run(options: LCORunOptions):
        seen.append(options)
        return {"stats": {"succeeded": 10}}

    monkeypatch.setattr(lco_cli, "run_lco_pipeline", fake_run)
    lco_cli.main(["Downloads", "outputs/lco", "--limit", "10", "--shard-size", "2"])
    assert seen == [
        LCORunOptions(
            Path("Downloads"), Path("outputs/lco"), limit=10, shard_size=2
        )
    ]
    assert json.loads(capsys.readouterr().out) == {"succeeded": 10}

    consolidated: list[tuple[Path, Path]] = []

    def fake_consolidate(output_dir: Path, destination: Path) -> int:
        consolidated.append((output_dir, destination))
        return 10

    monkeypatch.setattr(lco_cli, "consolidate", fake_consolidate)
    lco_cli.main(["consolidate", "outputs/lco"])
    assert consolidated == [
        (Path("outputs/lco"), Path("outputs/lco/lco-embeddings.npz"))
    ]
    assert json.loads(capsys.readouterr().out)["consolidated"] == 10


def test_default_lco_output_suppresses_per_video_third_party_noise() -> None:
    logger = logging.getLogger("qwen_omni_utils.v2_5.vision_process")
    old_level = logging.getLogger("qwen_omni_utils").level
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            lco_cli._configure_runtime_output(verbose=False)
            warnings.warn_explicit(
                "librosa.core.audio.__audioread_load\nDeprecated compatibility path",
                FutureWarning,
                filename="audio.py",
                lineno=1,
                module="librosa.core.audio",
            )
        assert captured == []
        assert not logger.isEnabledFor(logging.INFO)
        assert logger.isEnabledFor(logging.WARNING)
    finally:
        logging.getLogger("qwen_omni_utils").setLevel(old_level)
