from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np

# PyTorch 2.13 may replace ordinary eager CUDA operations with optional Triton
# JIT kernels.  They are unrelated to the requested SDPA attention path and
# require system Python development headers on DGX Spark, so keep inference on
# PyTorch's built-in CUDA kernels.
os.environ["TORCH_DISABLE_NATIVE_JIT"] = "1"

import torch


MODEL_ID = "LCO-Embedding/LCO-Embedding-Omni-7B"
MODEL_REVISION = "108f6f1a5de3b2eedd1d1c7b7005aaca6ed3802c"
DEVICE = "cuda:0"
ATTENTION_IMPLEMENTATION = "sdpa"
PROMPT = "\nSummarize the above video in one word:"
FPS = 2
MAX_FRAMES = 32
MAX_PIXELS = 64 * 28 * 28
# qwen-omni-utils defaults video frames to at least 128 visual tokens, which is
# larger than the locked 64-token visual cap.  Pinning the per-item minimum to
# the same budget makes the requested cap valid without changing its value.
MIN_PIXELS = MAX_PIXELS
EMBEDDING_DIMENSION = 3584
USE_AUDIO_IN_VIDEO = True


@dataclass(frozen=True)
class LCODependencies:
    """Optional LCO runtime components, injectable for tests."""

    model_class: Any
    processor_class: Any
    process_mm_info: Callable[..., tuple[Any, Any, Any]]


@dataclass(frozen=True)
class LCOEmbeddingResult:
    embedding: np.ndarray
    sampled_frame_count: int


def load_lco_dependencies() -> LCODependencies:
    """Import the optional LCO stack only when real-model inference is requested."""
    try:
        from transformers import (
            Qwen2_5OmniProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )
    except ImportError as exc:  # pragma: no cover - depends on the optional environment
        raise RuntimeError(
            "LCO dependencies are unavailable; install the project's 'lco' extra"
        ) from exc

    try:
        from qwen_omni_utils import process_mm_info
    except ImportError as exc:  # pragma: no cover - depends on the optional environment
        raise RuntimeError(
            "qwen-omni-utils is unavailable; install the project's 'lco' extra"
        ) from exc

    return LCODependencies(
        model_class=Qwen2_5OmniThinkerForConditionalGeneration,
        processor_class=Qwen2_5OmniProcessor,
        process_mm_info=process_mm_info,
    )


def _shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(dimension) for dimension in shape)
    except (TypeError, ValueError):
        return None


def _frame_count_from_video_item(value: Any) -> int | None:
    """Return T from a qwen-omni-utils video tensor or frame sequence."""
    shape = _shape(value)
    if shape is not None:
        # qwen-omni-utils returns video tensors as TCHW. A three-dimensional
        # value is a single frame and is intentionally not interpreted as T.
        if len(shape) >= 4 and shape[0] > 0:
            return shape[0]
        return None

    if isinstance(value, Mapping):
        for key in ("video", "frames", "pixel_values_videos", "pixel_values"):
            if key in value:
                count = _frame_count_from_video_item(value[key])
                if count is not None:
                    return count
        return None

    if isinstance(value, tuple) and value:
        # Some utility versions pair the TCHW tensor with sampled-FPS metadata.
        count = _frame_count_from_video_item(value[0])
        if count is not None:
            return count

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            return None
        first = value[0]
        nested_count = _frame_count_from_video_item(first)
        if nested_count is not None:
            return nested_count
        # A sequence of PIL images or CHW tensors represents sampled frames.
        first_shape = _shape(first)
        if first_shape is None or len(first_shape) == 3:
            return len(value)
    return None


def _mapping_value(mapping: Any, key: str) -> Any | None:
    if isinstance(mapping, Mapping):
        return mapping.get(key)
    try:
        return mapping[key]
    except (KeyError, TypeError, IndexError):
        return None


def _temporal_patch_size(processor: Any) -> int:
    for owner in (
        processor,
        getattr(processor, "video_processor", None),
        getattr(processor, "image_processor", None),
    ):
        value = getattr(owner, "temporal_patch_size", None)
        if value is not None:
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                return parsed
    # Qwen2.5-Omni's fixed temporal patch factor.
    return 2


def sampled_frame_count(
    video_inputs: Any,
    processed_inputs: Any | None = None,
    processor: Any | None = None,
) -> int:
    """Derive the number of frames actually emitted by multimodal preprocessing.

    The utility output is preferred because it is the post-sampling TCHW tensor.
    ``video_grid_thw`` is a fallback for compatible processor implementations that
    do not expose that tensor directly.
    """
    count = _frame_count_from_video_item(video_inputs)
    if count is None and processed_inputs is not None:
        grid = _mapping_value(processed_inputs, "video_grid_thw")
        grid_shape = _shape(grid)
        if grid_shape is not None:
            try:
                temporal_grid = int(grid[0][0] if len(grid_shape) >= 2 else grid[0])
            except (IndexError, TypeError, ValueError):
                temporal_grid = 0
            if temporal_grid > 0:
                count = temporal_grid * _temporal_patch_size(processor)

    if count is None or count <= 0:
        raise ValueError("could not determine a positive sampled frame count")
    if count > MAX_FRAMES:
        raise ValueError(
            f"preprocessor emitted {count} frames, exceeding max_frames={MAX_FRAMES}"
        )
    return count


def _move_value(value: Any, device: str, dtype: torch.dtype) -> Any:
    if isinstance(value, torch.Tensor):
        if torch.is_floating_point(value):
            return value.to(device=device, dtype=dtype)
        return value.to(device=device)
    if isinstance(value, Mapping):
        return {key: _move_value(item, device, dtype) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_value(item, device, dtype) for item in value)
    if isinstance(value, list):
        return [_move_value(item, device, dtype) for item in value]
    return value


class LCOEmbedder:
    """Single-item audio+video embedder for the pinned LCO Omni checkpoint."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        process_mm_info: Callable[..., tuple[Any, Any, Any]],
        *,
        device: str = DEVICE,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model = model
        self.processor = processor
        self.process_mm_info = process_mm_info
        self.device = device
        self.dtype = dtype

    @classmethod
    def from_pretrained(
        cls, dependencies: LCODependencies | None = None
    ) -> LCOEmbedder:
        """Load the pinned model directly on CUDA 0 in BF16 using SDPA."""
        if not torch.cuda.is_available():
            raise RuntimeError("LCO inference requires CUDA, but CUDA is unavailable")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("LCO inference requires a CUDA device with BF16 support")

        resolved = dependencies or load_lco_dependencies()
        model = resolved.model_class.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            torch_dtype=torch.bfloat16,
            attn_implementation=ATTENTION_IMPLEMENTATION,
        )
        model.to(device=DEVICE)
        model.eval()
        processor = resolved.processor_class.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
        )
        return cls(
            model,
            processor,
            resolved.process_mm_info,
            device=DEVICE,
            dtype=torch.bfloat16,
        )

    def embed(self, video_path: str | Path) -> LCOEmbeddingResult:
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": str(video_path),
                            "fps": FPS,
                            "max_frames": MAX_FRAMES,
                            "min_pixels": MIN_PIXELS,
                            "max_pixels": MAX_PIXELS,
                        },
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ]
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        audio_inputs, image_inputs, video_inputs = self.process_mm_info(
            messages,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
        )
        if audio_inputs is None:
            raise ValueError("multimodal preprocessing returned no video audio")
        if video_inputs is None:
            raise ValueError("multimodal preprocessing returned no video frames")

        inputs = self.processor(
            text=text,
            audio=audio_inputs,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
        )
        frame_count = sampled_frame_count(video_inputs, inputs, self.processor)
        model_inputs = _move_value(inputs, self.device, self.dtype)
        # The processor consumes this flag while constructing aligned tokens but
        # does not include it in BatchFeature.  The Thinker needs the same flag
        # explicitly to apply synchronized audio/video position logic.
        model_inputs["use_audio_in_video"] = USE_AUDIO_IN_VIDEO

        with torch.inference_mode():
            outputs = self.model(
                **model_inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError("LCO model returned no hidden states")
        final_hidden = hidden_states[-1]
        if final_hidden.ndim != 3 or final_hidden.shape[0] != 1 or final_hidden.shape[1] == 0:
            raise RuntimeError(
                "LCO final hidden state must have shape [1, sequence, dimensions]"
            )
        embedding = final_hidden[0, -1, :].float()
        if embedding.numel() != EMBEDDING_DIMENSION:
            raise RuntimeError(
                f"LCO embedding has dimension {embedding.numel()}, "
                f"expected {EMBEDDING_DIMENSION}"
            )
        if not bool(torch.isfinite(embedding).all()):
            raise RuntimeError("LCO embedding contains non-finite values")
        norm = torch.linalg.vector_norm(embedding)
        if not bool(torch.isfinite(norm)) or float(norm) == 0.0:
            raise RuntimeError("LCO embedding has an invalid zero norm")

        normalized = (embedding / norm).cpu().numpy().astype(np.float32, copy=False)
        return LCOEmbeddingResult(
            embedding=normalized,
            sampled_frame_count=frame_count,
        )
