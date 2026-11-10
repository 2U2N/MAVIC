from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest
import torch

from mavic.representations.f2llm.embedding import (
    DEVICE,
    EMBEDDING_DIMENSION,
    MODEL_ID,
    MODEL_REVISION,
    F2LLMEmbedder,
)


def _unit_rows(count: int) -> np.ndarray:
    values = np.zeros((count, EMBEDDING_DIMENSION), dtype=np.float32)
    for index in range(count):
        values[index, index] = 1.0
    return values


class _FakeModel:
    def __init__(self, lengths: list[int]) -> None:
        self.lengths = lengths
        self.max_seq_length = 0
        self.eval_called = False
        self.output_scale = 1.0
        self.encode_calls: list[dict] = []

    def tokenizer(self, texts, **kwargs):
        assert kwargs == {
            "add_special_tokens": True,
            "padding": False,
            "truncation": False,
        }
        return {"input_ids": [list(range(length)) for length in self.lengths]}

    def encode(self, texts, **kwargs):
        self.encode_calls.append(kwargs)
        return _unit_rows(len(texts)) * self.output_scale

    def eval(self):
        self.eval_called = True
        return self


def test_embed_records_pre_truncation_lengths_and_normalized_vectors() -> None:
    model = _FakeModel([3, 7])
    model.output_scale = 0.996
    result = F2LLMEmbedder(model, max_tokens=4).embed(["one", "two"])

    assert result.embeddings.shape == (2, EMBEDDING_DIMENSION)
    assert result.embeddings.dtype == np.float32
    assert result.token_counts.tolist() == [3, 7]
    assert result.token_counts.dtype == np.uint32
    assert result.truncated.tolist() == [False, True]
    assert np.allclose(np.linalg.norm(result.embeddings, axis=1), 1.0)
    assert model.encode_calls == [
        {
            "batch_size": 2,
            "show_progress_bar": False,
            "convert_to_numpy": True,
            "normalize_embeddings": True,
        }
    ]


def test_embed_rejects_invalid_model_output() -> None:
    model = _FakeModel([2])
    model.encode = lambda texts, **kwargs: np.zeros(  # type: ignore[method-assign]
        (1, EMBEDDING_DIMENSION), dtype=np.float32
    )
    with pytest.raises(ValueError, match="L2-normalized"):
        F2LLMEmbedder(model, max_tokens=10).embed(["text"])


def test_from_pretrained_uses_pinned_bf16_cuda_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict = {}
    model = _FakeModel([1])

    def constructor(model_id, **kwargs):
        seen["model_id"] = model_id
        seen.update(kwargs)
        return model

    fake_module = ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = constructor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    embedder = F2LLMEmbedder.from_pretrained(max_tokens=4096)

    assert seen == {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "device": DEVICE,
        "model_kwargs": {"torch_dtype": torch.bfloat16},
    }
    assert embedder.max_tokens == 4096
    assert model.max_seq_length == 4096
    assert model.eval_called
