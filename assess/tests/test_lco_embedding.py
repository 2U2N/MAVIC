from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mavic.representations.lco.embedding import (
    ATTENTION_IMPLEMENTATION,
    DEVICE,
    EMBEDDING_DIMENSION,
    FPS,
    MAX_FRAMES,
    MAX_PIXELS,
    MIN_PIXELS,
    MODEL_ID,
    MODEL_REVISION,
    PROMPT,
    LCODependencies,
    LCOEmbedder,
    sampled_frame_count,
)


class _FakeModel:
    def __init__(self, final_vector: torch.Tensor | None = None) -> None:
        self.eval_called = False
        self.to_calls: list[dict[str, object]] = []
        self.calls: list[dict[str, object]] = []
        self.final_vector = (
            torch.arange(1, EMBEDDING_DIMENSION + 1, dtype=torch.float32)
            if final_vector is None
            else final_vector
        )

    def to(self, **kwargs: object) -> _FakeModel:
        self.to_calls.append(kwargs)
        return self

    def eval(self) -> _FakeModel:
        self.eval_called = True
        return self

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        assert torch.is_inference_mode_enabled()
        self.calls.append(kwargs)
        dimension = self.final_vector.numel()
        earlier = torch.full((1, 3, dimension), -100.0)
        final = torch.zeros((1, 3, dimension))
        final[0, -1] = self.final_vector
        return SimpleNamespace(hidden_states=(earlier, final))


class _FakeProcessor:
    def __init__(self, processed: dict[str, torch.Tensor] | None = None) -> None:
        self.template_calls: list[tuple[object, dict[str, object]]] = []
        self.calls: list[dict[str, object]] = []
        self.processed = processed or {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.int64),
            "features": torch.ones((1, 2), dtype=torch.float32),
        }
        self.image_processor = SimpleNamespace(temporal_patch_size=2)

    def apply_chat_template(self, messages: object, **kwargs: object) -> list[str]:
        self.template_calls.append((messages, kwargs))
        return ["rendered"]

    def __call__(self, **kwargs: object) -> dict[str, torch.Tensor]:
        self.calls.append(kwargs)
        return self.processed


def test_fixed_model_and_preprocessing_constants() -> None:
    assert MODEL_ID == "LCO-Embedding/LCO-Embedding-Omni-7B"
    assert MODEL_REVISION == "108f6f1a5de3b2eedd1d1c7b7005aaca6ed3802c"
    assert DEVICE == "cuda:0"
    assert ATTENTION_IMPLEMENTATION == "sdpa"
    assert PROMPT == "\nSummarize the above video in one word:"
    assert FPS == 2
    assert MAX_FRAMES == 32
    assert MIN_PIXELS == 64 * 28 * 28
    assert MAX_PIXELS == 64 * 28 * 28


def test_from_pretrained_loads_pinned_bf16_sdpa_model(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _FakeModel()
    model_loads: list[tuple[object, dict[str, object]]] = []
    processor_loads: list[tuple[object, dict[str, object]]] = []

    class ModelClass:
        @classmethod
        def from_pretrained(cls, model_id: object, **kwargs: object) -> _FakeModel:
            model_loads.append((model_id, kwargs))
            return model

    class ProcessorClass:
        @classmethod
        def from_pretrained(cls, model_id: object, **kwargs: object) -> _FakeProcessor:
            processor_loads.append((model_id, kwargs))
            return _FakeProcessor()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    dependencies = LCODependencies(ModelClass, ProcessorClass, lambda *_args, **_kwargs: ())

    embedder = LCOEmbedder.from_pretrained(dependencies)

    assert embedder.model is model
    assert model.to_calls == [{"device": "cuda:0"}]
    assert model.eval_called
    assert model_loads == [
        (
            MODEL_ID,
            {
                "revision": MODEL_REVISION,
                "torch_dtype": torch.bfloat16,
                "attn_implementation": "sdpa",
            },
        )
    ]
    assert processor_loads == [(MODEL_ID, {"revision": MODEL_REVISION})]


@pytest.mark.parametrize(
    ("cuda_available", "bf16_available", "message"),
    [
        (False, True, "requires CUDA"),
        (True, False, "BF16 support"),
    ],
)
def test_from_pretrained_rejects_unsupported_runtime(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    bf16_available: bool,
    message: str,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: bf16_available)
    with pytest.raises(RuntimeError, match=message):
        LCOEmbedder.from_pretrained(
            LCODependencies(object(), object(), lambda *_args, **_kwargs: ())
        )


def test_embed_uses_joint_audio_video_and_returns_normalized_final_token() -> None:
    processor = _FakeProcessor()
    model = _FakeModel()
    mm_calls: list[tuple[object, dict[str, object]]] = []
    sampled_video = torch.zeros((12, 3, 28, 28))

    def process_mm_info(messages: object, **kwargs: object):
        mm_calls.append((messages, kwargs))
        return [np.ones(1600, dtype=np.float32)], None, [sampled_video]

    embedder = LCOEmbedder(
        model,
        processor,
        process_mm_info,
        device="cpu",
        dtype=torch.float32,
    )
    result = embedder.embed("central-window.mp4")

    messages = mm_calls[0][0]
    video_content = messages[0][0]["content"][0]
    text_content = messages[0][0]["content"][1]
    assert video_content == {
        "type": "video",
        "video": "central-window.mp4",
        "fps": 2,
        "max_frames": 32,
        "min_pixels": 64 * 28 * 28,
        "max_pixels": 64 * 28 * 28,
    }
    assert text_content == {"type": "text", "text": PROMPT}
    assert mm_calls[0][1] == {"use_audio_in_video": True}
    assert processor.template_calls[0][1] == {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    processor_call = processor.calls[0]
    assert processor_call["audio"] is not None
    assert processor_call["videos"] == [sampled_video]
    assert processor_call["use_audio_in_video"] is True
    assert processor_call["return_tensors"] == "pt"
    assert processor_call["padding"] is True
    assert model.calls[0]["output_hidden_states"] is True
    assert model.calls[0]["return_dict"] is True
    assert model.calls[0]["use_audio_in_video"] is True

    expected = torch.arange(1, EMBEDDING_DIMENSION + 1, dtype=torch.float32)
    expected /= torch.linalg.vector_norm(expected)
    assert result.sampled_frame_count == 12
    assert result.embedding.shape == (EMBEDDING_DIMENSION,)
    assert result.embedding.dtype == np.float32
    np.testing.assert_allclose(result.embedding, expected.numpy(), rtol=1e-6, atol=1e-7)
    assert np.linalg.norm(result.embedding) == pytest.approx(1.0, abs=1e-6)


def test_sampled_frame_count_handles_frame_lists_and_grid_fallback() -> None:
    assert sampled_frame_count([[object(), object(), object(), object()]]) == 4
    processor = SimpleNamespace(image_processor=SimpleNamespace(temporal_patch_size=2))
    assert sampled_frame_count(None, {"video_grid_thw": torch.tensor([[7, 2, 2]])}, processor) == 14


def test_sampled_frame_count_rejects_unobservable_or_excess_frames() -> None:
    with pytest.raises(ValueError, match="could not determine"):
        sampled_frame_count(None, {})
    with pytest.raises(ValueError, match="exceeding max_frames"):
        sampled_frame_count([torch.zeros((34, 3, 2, 2))])


@pytest.mark.parametrize(
    ("audio_inputs", "video_inputs", "message"),
    [
        (None, [torch.zeros((4, 3, 2, 2))], "no video audio"),
        ([np.ones(10)], None, "no video frames"),
    ],
)
def test_embed_rejects_missing_modality(
    audio_inputs: object,
    video_inputs: object,
    message: str,
) -> None:
    embedder = LCOEmbedder(
        _FakeModel(),
        _FakeProcessor(),
        lambda *_args, **_kwargs: (audio_inputs, None, video_inputs),
        device="cpu",
        dtype=torch.float32,
    )
    with pytest.raises(ValueError, match=message):
        embedder.embed("video.mp4")


@pytest.mark.parametrize(
    ("vector", "message"),
    [
        (torch.zeros(EMBEDDING_DIMENSION), "zero norm"),
        (torch.full((EMBEDDING_DIMENSION,), float("nan")), "non-finite"),
        (torch.ones(10), "dimension 10"),
    ],
)
def test_embed_rejects_invalid_model_embeddings(vector: torch.Tensor, message: str) -> None:
    embedder = LCOEmbedder(
        _FakeModel(vector),
        _FakeProcessor(),
        lambda *_args, **_kwargs: (
            [np.ones(10)],
            None,
            [torch.zeros((4, 3, 2, 2))],
        ),
        device="cpu",
        dtype=torch.float32,
    )
    with pytest.raises(RuntimeError, match=message):
        embedder.embed("video.mp4")
