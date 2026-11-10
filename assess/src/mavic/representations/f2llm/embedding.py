from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

# This must be set before importing torch. PyTorch's optional native Triton JIT
# needs Python development headers that are not present on the target runtime.
os.environ["TORCH_DISABLE_NATIVE_JIT"] = "1"

import torch


MODEL_ID = "codefuse-ai/F2LLM-v2-4B"
MODEL_REVISION = "2c0231f93661c5260cb71efb2097a3f7210b5996"
EMBEDDING_DIMENSION = 2560
DEVICE = "cuda:0"
PRECISION = "bf16"
POOLING = "last-token"
NORMALIZATION = "l2"


@dataclass(frozen=True)
class F2LLMBatchResult:
    embeddings: np.ndarray
    token_counts: np.ndarray
    truncated: np.ndarray


class F2LLMEmbedder:
    def __init__(self, model: Any, *, max_tokens: int) -> None:
        self.model = model
        self.max_tokens = max_tokens

    @classmethod
    def from_pretrained(cls, *, max_tokens: int) -> "F2LLMEmbedder":
        if not torch.cuda.is_available():
            raise RuntimeError("F2LLM-v2-4B requires CUDA")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("F2LLM-v2-4B requires CUDA BF16 support")

        # Keep the optional text stack out of normal imports and unit tests.
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            MODEL_ID,
            revision=MODEL_REVISION,
            device=DEVICE,
            model_kwargs={"torch_dtype": torch.bfloat16},
        )
        model.max_seq_length = max_tokens
        model.eval()
        return cls(model, max_tokens=max_tokens)

    def _token_counts(self, texts: Sequence[str]) -> np.ndarray:
        tokenized = self.model.tokenizer(
            list(texts),
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        input_ids = tokenized.get("input_ids")
        if not isinstance(input_ids, list) or len(input_ids) != len(texts):
            raise RuntimeError("F2LLM tokenizer returned invalid input_ids")
        counts = np.asarray([len(values) for values in input_ids], dtype=np.uint32)
        if np.any(counts == 0):
            raise RuntimeError("F2LLM tokenizer returned an empty token sequence")
        return counts

    def embed(self, texts: Sequence[str]) -> F2LLMBatchResult:
        values = list(texts)
        if not values or any(not isinstance(value, str) or not value for value in values):
            raise ValueError("F2LLM input must contain non-empty strings")
        token_counts = self._token_counts(values)
        with torch.inference_mode():
            encoded = self.model.encode(
                values,
                batch_size=len(values),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        embeddings = np.asarray(encoded, dtype=np.float32)
        if embeddings.shape != (len(values), EMBEDDING_DIMENSION):
            raise ValueError(
                "F2LLM embeddings must have shape "
                f"(N, {EMBEDDING_DIMENSION}); got {embeddings.shape}"
            )
        if not np.isfinite(embeddings).all():
            raise ValueError("F2LLM embeddings contain non-finite values")
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.isfinite(norms).all() or np.any(norms == 0):
            raise ValueError("F2LLM embeddings cannot be L2-normalized")
        # Sentence Transformers performs its requested normalization in the
        # model dtype. With BF16 weights, casting that result to float32 can
        # leave norms outside the strict persistence tolerance, so finish the
        # normalization in float32 before validating and storing it.
        embeddings = np.asarray(embeddings / norms[:, np.newaxis], dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4):
            raise ValueError("F2LLM embeddings must be L2-normalized")
        return F2LLMBatchResult(
            embeddings=embeddings,
            token_counts=token_counts,
            truncated=np.asarray(token_counts > self.max_tokens, dtype=np.bool_),
        )
