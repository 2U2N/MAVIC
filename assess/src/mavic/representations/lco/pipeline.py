from __future__ import annotations

import importlib.metadata
import logging
import os
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

# This must be set before importing torch; see lco_embedding for rationale.
os.environ["TORCH_DISABLE_NATIVE_JIT"] = "1"

import torch

from .embedding import (
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
    USE_AUDIO_IN_VIDEO,
    LCOEmbedder,
    LCOEmbeddingResult,
)
from .media import ANALYSIS_WINDOW_SECONDS, PreparedAnalysisClip, prepare_analysis_clip
from .persistence import (
    PIPELINE_ID,
    add_shard,
    checkpoint_manifest,
    load_state,
    reset_generated_output,
    utc_now,
    write_shard,
)
from ..openclip.sampling import discover_videos


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LCOProcessingConfig:
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    prompt: str = PROMPT
    window_seconds: float = ANALYSIS_WINDOW_SECONDS
    fps: int = FPS
    max_frames: int = MAX_FRAMES
    min_pixels: int = MIN_PIXELS
    max_pixels: int = MAX_PIXELS
    embedding_dimension: int = EMBEDDING_DIMENSION
    precision: str = "bf16"
    attention_implementation: str = ATTENTION_IMPLEMENTATION
    audio_in_video: bool = USE_AUDIO_IN_VIDEO
    temporal_policy: str = "whole-if-short-otherwise-centered"
    padding_policy: str = "none"


@dataclass(frozen=True)
class LCORunOptions:
    input_dir: Path
    output_dir: Path
    limit: int | None = None
    shard_size: int = 100
    overwrite: bool = False


class _Embedder(Protocol):
    def embed(self, video_path: str | Path) -> LCOEmbeddingResult: ...


ClipPreparer = Callable[[Path], AbstractContextManager[PreparedAnalysisClip]]
EmbedderFactory = Callable[[], _Embedder]


class _GpuUtilizationSampler:
    def __init__(self, device: torch.device, interval_seconds: float = 0.5) -> None:
        self.device = device
        self.interval_seconds = interval_seconds
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not hasattr(torch.cuda, "utilization"):
            return
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.samples.append(float(torch.cuda.utilization(self.device)))
            except Exception:
                # Missing NVML support must not fail model inference.
                return

    def stop(self) -> tuple[float | None, float | None]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds * 2)
        if not self.samples:
            return None, None
        return statistics.fmean(self.samples), max(self.samples)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _tool_version(executable: str) -> str:
    resolved = shutil.which(executable)
    if resolved is None:
        raise RuntimeError(f"required executable is unavailable on PATH: {executable}")
    try:
        result = subprocess.run(
            [resolved, "-version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(f"could not run {executable}: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"{executable} version check failed: {detail}")
    first_line = result.stdout.splitlines()[0] if result.stdout else "unknown"
    return first_line.strip()


def _runtime_provenance() -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "pipeline_id": PIPELINE_ID,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_native_jit_disabled": os.environ.get("TORCH_DISABLE_NATIVE_JIT") == "1",
        "ffmpeg": _tool_version("ffmpeg"),
        "ffprobe": _tool_version("ffprobe"),
        "packages": {
            "nvidia-ml-py": _package_version("nvidia-ml-py"),
            "qwen-omni-utils": _package_version("qwen-omni-utils"),
            "torchcodec": _package_version("torchcodec"),
            "transformers": _package_version("transformers"),
        },
    }
    if torch.cuda.is_available():
        device = torch.device(DEVICE)
        provenance["cuda_device"] = torch.cuda.get_device_name(device)
        provenance["cuda_capability"] = list(torch.cuda.get_device_capability(device))
        provenance["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    return provenance


def _validate_options(options: LCORunOptions) -> None:
    if options.limit is not None and options.limit <= 0:
        raise ValueError("--limit must be positive")
    if options.shard_size <= 0:
        raise ValueError("--shard-size must be positive")


def _validate_result(result: LCOEmbeddingResult) -> tuple[np.ndarray, int]:
    embedding = np.asarray(result.embedding)
    if embedding.dtype != np.float32 or embedding.shape != (EMBEDDING_DIMENSION,):
        raise ValueError(
            f"LCO embedding must be float32 with shape ({EMBEDDING_DIMENSION},)"
        )
    if not np.isfinite(embedding).all():
        raise ValueError("LCO embedding contains non-finite values")
    if not np.isclose(np.linalg.norm(embedding), 1.0, rtol=1e-4, atol=1e-4):
        raise ValueError("LCO embedding is not L2-normalized")
    frame_count = int(result.sampled_frame_count)
    if frame_count <= 0 or frame_count > MAX_FRAMES:
        raise ValueError(f"sampled frame count must be between 1 and {MAX_FRAMES}")
    return embedding, frame_count


def _error_record(stage: str, exc: Exception) -> dict[str, str]:
    return {
        "stage": stage,
        "type": type(exc).__name__,
        "message": str(exc),
    }


def run_lco_pipeline(
    options: LCORunOptions,
    *,
    embedder_factory: EmbedderFactory | None = None,
    clip_preparer: ClipPreparer = prepare_analysis_clip,
) -> dict[str, Any]:
    """Embed a deterministic, optionally limited set of videos sequentially."""
    _validate_options(options)
    videos = discover_videos(options.input_dir)
    if not videos:
        raise ValueError(f"no MP4 videos found in {options.input_dir}")
    selected = videos if options.limit is None else videos[: options.limit]
    selected_ids = {path.stem for path in selected}

    options.output_dir.mkdir(parents=True, exist_ok=True)
    if options.overwrite:
        reset_generated_output(options.output_dir)

    config = asdict(LCOProcessingConfig())
    state = load_state(options.output_dir, config)
    if not state.manifest.get("provenance"):
        state.manifest["provenance"] = _runtime_provenance()
    pending = [path for path in selected if path.stem not in state.completed_ids]

    LOGGER.info(
        "discovered %d videos; selected %d (%d complete, %d pending)",
        len(videos),
        len(selected),
        len(selected) - len(pending),
        len(pending),
    )

    started = time.perf_counter()
    run_started_at = utc_now()
    errors = dict(state.manifest.get("errors", {}))
    completed_ids = set(state.completed_ids)
    new_successes = 0

    buffered_ids: list[str] = []
    buffered_embeddings: list[np.ndarray] = []
    buffered_source_durations: list[float] = []
    buffered_window_starts: list[float] = []
    buffered_window_ends: list[float] = []
    buffered_frame_counts: list[int] = []
    shard_index = state.next_shard_index

    def update_stats() -> None:
        relevant_completed = completed_ids.intersection(selected_ids)
        relevant_errors = selected_ids.intersection(errors).difference(relevant_completed)
        state.manifest["errors"] = {
            video_id: record
            for video_id, record in errors.items()
            if video_id not in completed_ids
        }
        state.manifest["stats"] = {
            "discovered": len(videos),
            "selected": len(selected),
            "succeeded": len(relevant_completed),
            "failed": len(relevant_errors),
            "pending": len(selected_ids - relevant_completed - relevant_errors),
        }

    def checkpoint() -> None:
        update_stats()
        checkpoint_manifest(options.output_dir, state.manifest)

    def flush_buffer() -> None:
        nonlocal shard_index
        if not buffered_ids:
            return
        shard = write_shard(
            options.output_dir,
            shard_index,
            buffered_ids,
            np.stack(buffered_embeddings),
            buffered_source_durations,
            buffered_window_starts,
            buffered_window_ends,
            buffered_frame_counts,
        )
        add_shard(state.manifest, shard)
        completed_ids.update(buffered_ids)
        for video_id in buffered_ids:
            errors.pop(video_id, None)
        buffered_ids.clear()
        buffered_embeddings.clear()
        buffered_source_durations.clear()
        buffered_window_starts.clear()
        buffered_window_ends.clear()
        buffered_frame_counts.clear()
        shard_index += 1
        checkpoint()

    sampler: _GpuUtilizationSampler | None = None
    gpu_mean: float | None = None
    gpu_max: float | None = None
    peak_memory: int | None = None

    if pending:
        # Check tools before paying the model-loading cost.
        _tool_version("ffmpeg")
        _tool_version("ffprobe")
        factory = embedder_factory or LCOEmbedder.from_pretrained
        embedder = factory()

        if torch.cuda.is_available():
            device = torch.device(DEVICE)
            torch.cuda.reset_peak_memory_stats(device)
            sampler = _GpuUtilizationSampler(device)
            sampler.start()

        try:
            for path in pending:
                stage = "media_preparation"
                try:
                    with clip_preparer(path) as prepared:
                        stage = "inference"
                        result = embedder.embed(prepared.clip_path)
                        embedding, frame_count = _validate_result(result)
                        buffered_ids.append(path.stem)
                        buffered_embeddings.append(embedding)
                        buffered_source_durations.append(prepared.source_duration_seconds)
                        buffered_window_starts.append(prepared.window_start_seconds)
                        buffered_window_ends.append(prepared.window_end_seconds)
                        buffered_frame_counts.append(frame_count)
                        errors.pop(path.stem, None)
                        new_successes += 1
                    if len(buffered_ids) >= options.shard_size:
                        flush_buffer()
                except torch.cuda.OutOfMemoryError as exc:
                    # Continuing would repeat the same unsafe input budget for
                    # every remaining video, potentially across the full corpus.
                    LOGGER.exception("LCO %s ran out of CUDA memory for %s", stage, path.name)
                    errors[path.stem] = _error_record(stage, exc)
                    checkpoint()
                    raise
                except Exception as exc:
                    LOGGER.exception("LCO %s failed for %s", stage, path.name)
                    errors[path.stem] = _error_record(stage, exc)
                    checkpoint()
            flush_buffer()
        finally:
            if sampler is not None:
                gpu_mean, gpu_max = sampler.stop()
            if torch.cuda.is_available():
                device = torch.device(DEVICE)
                peak_memory = int(torch.cuda.max_memory_allocated(device))

    elapsed = time.perf_counter() - started
    relevant_completed = completed_ids.intersection(selected_ids)
    run_record = {
        "started_at": run_started_at,
        "elapsed_seconds": elapsed,
        "limit": options.limit,
        "selected": len(selected),
        "pending_at_start": len(pending),
        "new_successes": new_successes,
        "skipped": len(selected) - len(pending),
        "videos_per_second": new_successes / elapsed if elapsed else 0.0,
        "peak_cuda_memory_bytes": peak_memory,
        "gpu_utilization_mean_percent": gpu_mean,
        "gpu_utilization_max_percent": gpu_max,
        "device": DEVICE,
        "shard_size": options.shard_size,
    }
    state.manifest.setdefault("runs", []).append(run_record)
    state.manifest["last_run"] = run_record
    update_stats()
    checkpoint_manifest(options.output_dir, state.manifest)

    if not relevant_completed:
        raise RuntimeError("no videos were embedded successfully")
    return state.manifest
