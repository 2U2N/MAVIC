from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import mavic.representations.openclip.cli as openclip_cli
import mavic.representations.openclip.pipeline as pipeline
from mavic.representations.openclip.persistence import consolidate, load_manifest
from mavic.representations.openclip.pipeline import (
    ProcessingConfig,
    RunOptions,
    run_pipeline,
)


class _FakeModel:
    def __init__(self) -> None:
        self.eval_called = False

    def eval(self) -> None:
        self.eval_called = True

    def encode_image(self, frames: torch.Tensor) -> torch.Tensor:
        encoded = torch.zeros((len(frames), 4), dtype=torch.float32)
        encoded[:, 0] = 1.0
        return encoded


def _install_mock_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    video_ids: list[str],
) -> tuple[_FakeModel, list[tuple[str, str, torch.device]]]:
    model = _FakeModel()
    loads: list[tuple[str, str, torch.device]] = []

    def create_model_and_transforms(model_name, *, pretrained, device):
        loads.append((model_name, pretrained, device))
        return model, None, object()

    monkeypatch.setitem(
        __import__("sys").modules,
        "open_clip",
        SimpleNamespace(create_model_and_transforms=create_model_and_transforms),
    )

    def fake_loader(dataset, **kwargs):
        assert kwargs["shuffle"] is False
        assert kwargs["num_workers"] == 0
        frames = torch.zeros((len(video_ids), 16, 3, 2, 2), dtype=torch.float32)
        return [
            {
                "video_ids": video_ids,
                "frames": frames,
                "errors": {},
            }
        ]

    monkeypatch.setattr(pipeline.torch.utils.data, "DataLoader", fake_loader)
    return model, loads


def test_mocked_openclip_pipeline_resume_and_consolidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    for name in ("b.mp4", "a.MP4"):
        (input_dir / name).touch()

    model, loads = _install_mock_runtime(monkeypatch, video_ids=["a", "b"])
    config = ProcessingConfig(precision="fp32")
    options = RunOptions(
        input_dir=input_dir,
        output_dir=output_dir,
        videos_per_batch=2,
        workers=0,
        device="cpu",
        shard_size=1,
    )

    manifest = run_pipeline(config, options)

    assert model.eval_called
    assert loads == [("ViT-B-32-quickgelu", "openai", torch.device("cpu"))]
    assert manifest["stats"] == {"discovered": 2, "succeeded": 2, "failed": 0}
    assert [item["file"] for item in manifest["shards"]] == [
        "shard-00000.npz",
        "shard-00001.npz",
    ]

    monkeypatch.setitem(
        __import__("sys").modules,
        "open_clip",
        SimpleNamespace(
            create_model_and_transforms=lambda *args, **kwargs: pytest.fail(
                "completed runs must not reload OpenCLIP"
            )
        ),
    )
    repeated = run_pipeline(config, options)
    assert repeated["last_run"]["pending"] == 0

    destination = output_dir / "openclip-embeddings.npz"
    assert consolidate(output_dir, destination) == 2
    with np.load(destination, allow_pickle=False) as data:
        assert data["video_ids"].tolist() == ["a", "b"]
        assert data["embeddings"].shape == (2, 4)
        assert data["retained_frame_counts"].tolist() == [1, 1]


def test_mocked_openclip_batch_failure_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "bad.mp4").touch()

    class _FailingModel(_FakeModel):
        def encode_image(self, frames: torch.Tensor) -> torch.Tensor:
            raise ValueError("synthetic inference failure")

    model, _ = _install_mock_runtime(monkeypatch, video_ids=["bad"])
    failing = _FailingModel()
    monkeypatch.setitem(
        __import__("sys").modules,
        "open_clip",
        SimpleNamespace(
            create_model_and_transforms=lambda *args, **kwargs: (failing, None, object())
        ),
    )

    with pytest.raises(RuntimeError, match="no videos were embedded successfully"):
        run_pipeline(
            ProcessingConfig(precision="fp32"),
            RunOptions(input_dir, output_dir, workers=0, device="cpu"),
        )
    assert model is not failing
    manifest = load_manifest(output_dir)
    assert manifest is not None
    assert manifest["stats"] == {"discovered": 1, "succeeded": 0, "failed": 1}
    assert "synthetic inference failure" in manifest["errors"]["bad"]


def test_openclip_cli_dispatches_run_and_consolidation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[ProcessingConfig, RunOptions]] = []

    def fake_run(config: ProcessingConfig, options: RunOptions):
        calls.append((config, options))
        return {"stats": {"succeeded": 3}}

    monkeypatch.setattr(openclip_cli, "run_pipeline", fake_run)
    openclip_cli.main(
        [
            "input",
            "output",
            "--device",
            "cpu",
            "--precision",
            "fp32",
            "--workers",
            "0",
        ]
    )
    assert calls[0][0].model == "ViT-B-32-quickgelu"
    assert calls[0][1].input_dir == Path("input")
    assert json.loads(capsys.readouterr().out) == {"succeeded": 3}

    consolidated: list[tuple[Path, Path]] = []
    monkeypatch.setattr(
        openclip_cli,
        "consolidate",
        lambda output, destination: consolidated.append((output, destination)) or 3,
    )
    openclip_cli.main(["consolidate", "output", "all.npz"])
    assert consolidated == [(Path("output"), Path("all.npz"))]
