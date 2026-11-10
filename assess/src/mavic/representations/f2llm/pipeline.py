from __future__ import annotations

import importlib.metadata
import logging
import os
import platform
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np

# PyTorch 2.13 otherwise attempts to compile optional native Triton kernels at
# inference time, which requires system Python development headers.
os.environ["TORCH_DISABLE_NATIVE_JIT"] = "1"

import torch

from .embedding import (
    DEVICE,
    EMBEDDING_DIMENSION,
    MODEL_ID,
    MODEL_REVISION,
    NORMALIZATION,
    POOLING,
    PRECISION,
    F2LLMBatchResult,
    F2LLMEmbedder,
)
from .metadata import METADATA_TEMPLATE, PROMPT, MetadataCatalog, MetadataRecord
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
class F2LLMProcessingConfig:
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    embedding_dimension: int = EMBEDDING_DIMENSION
    precision: str = PRECISION
    pooling: str = POOLING
    normalization: str = NORMALIZATION
    max_tokens: int = 4096
    prompt: str = PROMPT
    metadata_template: str = METADATA_TEMPLATE
    included_fields: str = "title,description,categories,tags"
    cleaning_policy: str = "raw"


@dataclass(frozen=True)
class F2LLMRunOptions:
    input_dir: Path
    output_dir: Path
    limit: int | None = None
    batch_size: int = 16
    max_tokens: int = 4096
    shard_size: int = 100
    overwrite: bool = False


class _Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> F2LLMBatchResult: ...


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


def _runtime_provenance() -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "pipeline_id": PIPELINE_ID,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_native_jit_disabled": os.environ.get("TORCH_DISABLE_NATIVE_JIT") == "1",
        "packages": {
            "sentence-transformers": _package_version("sentence-transformers"),
            "transformers": _package_version("transformers"),
            "huggingface-hub": _package_version("huggingface-hub"),
        },
    }
    if torch.cuda.is_available():
        device = torch.device(DEVICE)
        provenance["cuda_device"] = torch.cuda.get_device_name(device)
        provenance["cuda_capability"] = list(torch.cuda.get_device_capability(device))
        provenance["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    return provenance


def _validate_options(options: F2LLMRunOptions) -> None:
    if options.limit is not None and options.limit <= 0:
        raise ValueError("--limit must be positive")
    if options.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if options.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if options.shard_size <= 0:
        raise ValueError("--shard-size must be positive")


def _validate_batch_result(
    result: F2LLMBatchResult, expected_count: int
) -> F2LLMBatchResult:
    embeddings = np.asarray(result.embeddings)
    token_counts = np.asarray(result.token_counts)
    truncated = np.asarray(result.truncated)
    if embeddings.dtype != np.float32 or embeddings.shape != (
        expected_count,
        EMBEDDING_DIMENSION,
    ):
        raise ValueError(
            "F2LLM embeddings must be float32 with shape "
            f"({expected_count}, {EMBEDDING_DIMENSION})"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("F2LLM embeddings contain non-finite values")
    if not np.allclose(
        np.linalg.norm(embeddings, axis=1), 1.0, rtol=1e-4, atol=1e-4
    ):
        raise ValueError("F2LLM embeddings are not L2-normalized")
    if (
        token_counts.dtype != np.uint32
        or token_counts.shape != (expected_count,)
        or np.any(token_counts == 0)
    ):
        raise ValueError("F2LLM token counts must be positive uint32 values")
    if truncated.dtype != np.bool_ or truncated.shape != (expected_count,):
        raise ValueError("F2LLM truncated flags must be boolean values")
    return F2LLMBatchResult(embeddings, token_counts, truncated)


def _error_record(stage: str, exc: Exception) -> dict[str, str]:
    return {"stage": stage, "type": type(exc).__name__, "message": str(exc)}


def run_f2llm_pipeline(
    options: F2LLMRunOptions,
    *,
    embedder_factory: EmbedderFactory | None = None,
) -> dict[str, Any]:
    _validate_options(options)
    videos = discover_videos(options.input_dir)
    if not videos:
        raise ValueError(f"no MP4 videos found in {options.input_dir}")
    selected = videos if options.limit is None else videos[: options.limit]
    selected_ids = {path.stem for path in selected}

    options.output_dir.mkdir(parents=True, exist_ok=True)
    if options.overwrite:
        reset_generated_output(options.output_dir)
    config = asdict(F2LLMProcessingConfig(max_tokens=options.max_tokens))
    state = load_state(options.output_dir, config)
    if not state.manifest.get("provenance"):
        state.manifest["provenance"] = _runtime_provenance()

    errors = dict(state.manifest.get("errors", {}))
    catalog = MetadataCatalog(options.input_dir)
    records: dict[str, MetadataRecord] = {}
    for path in selected:
        try:
            record = catalog.record_for_video(path)
        except Exception as exc:
            errors[path.stem] = _error_record("metadata", exc)
        else:
            records[path.stem] = record

    for video_id in state.completed_ids.intersection(selected_ids):
        record = records.get(video_id)
        if record is None:
            raise ValueError(
                f"metadata for completed video {video_id} is now missing or invalid; "
                "pass --overwrite to rebuild"
            )
        if state.completed_text_hashes[video_id] != record.text_sha256:
            raise ValueError(
                f"metadata text changed for completed video {video_id}; "
                "pass --overwrite to rebuild"
            )

    completed_ids = set(state.completed_ids)
    pending_records = [
        records[path.stem] for path in selected if path.stem in records and path.stem not in completed_ids
    ]
    for record in pending_records:
        errors.pop(record.video_id, None)

    LOGGER.info(
        "discovered %d videos; selected %d; matched %d metadata sidecars "
        "(%d complete, %d pending)",
        len(videos),
        len(selected),
        len(records),
        len(selected_ids.intersection(completed_ids)),
        len(selected_ids - completed_ids),
    )

    started = time.perf_counter()
    run_started_at = utc_now()
    new_successes = 0
    shard_index = state.next_shard_index

    buffered_ids: list[str] = []
    buffered_embeddings: list[np.ndarray] = []
    buffered_token_counts: list[int] = []
    buffered_truncated: list[bool] = []
    buffered_hashes: list[str] = []

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
            "metadata_matched": len(records),
            "succeeded": len(relevant_completed),
            "failed": len(relevant_errors),
            "pending": len(selected_ids - relevant_completed - relevant_errors),
        }

    def checkpoint() -> None:
        update_stats()
        checkpoint_manifest(options.output_dir, state.manifest)

    def flush_buffer() -> None:
        nonlocal shard_index
        while buffered_ids:
            count = min(len(buffered_ids), options.shard_size)
            shard_ids = buffered_ids[:count]
            shard_hashes = buffered_hashes[:count]
            shard = write_shard(
                options.output_dir,
                shard_index,
                shard_ids,
                np.stack(buffered_embeddings[:count]),
                buffered_token_counts[:count],
                buffered_truncated[:count],
                shard_hashes,
            )
            add_shard(state.manifest, shard)
            completed_ids.update(shard_ids)
            state.completed_text_hashes.update(zip(shard_ids, shard_hashes, strict=True))
            for video_id in shard_ids:
                errors.pop(video_id, None)
            del buffered_ids[:count]
            del buffered_embeddings[:count]
            del buffered_token_counts[:count]
            del buffered_truncated[:count]
            del buffered_hashes[:count]
            shard_index += 1
            checkpoint()

    # Persist metadata failures before model loading, which can fail independently.
    checkpoint()
    sampler: _GpuUtilizationSampler | None = None
    gpu_mean: float | None = None
    gpu_max: float | None = None
    peak_memory: int | None = None

    if pending_records:
        factory = embedder_factory or (
            lambda: F2LLMEmbedder.from_pretrained(max_tokens=options.max_tokens)
        )
        embedder = factory()
        if torch.cuda.is_available():
            device = torch.device(DEVICE)
            torch.cuda.reset_peak_memory_stats(device)
            sampler = _GpuUtilizationSampler(device)
            sampler.start()

        def process_batch(batch: Sequence[MetadataRecord]) -> None:
            nonlocal new_successes
            try:
                result = _validate_batch_result(
                    embedder.embed([record.text for record in batch]), len(batch)
                )
            except torch.cuda.OutOfMemoryError as exc:
                flush_buffer()
                for record in batch:
                    errors[record.video_id] = _error_record("inference", exc)
                checkpoint()
                raise
            except Exception as exc:
                if len(batch) > 1:
                    LOGGER.warning(
                        "F2LLM batch inference failed; retrying %d records individually: %s",
                        len(batch),
                        exc,
                    )
                    for record in batch:
                        process_batch([record])
                    return
                record = batch[0]
                LOGGER.exception("F2LLM inference failed for %s", record.video_id)
                errors[record.video_id] = _error_record("inference", exc)
                checkpoint()
                return

            for index, record in enumerate(batch):
                buffered_ids.append(record.video_id)
                buffered_embeddings.append(result.embeddings[index])
                buffered_token_counts.append(int(result.token_counts[index]))
                buffered_truncated.append(bool(result.truncated[index]))
                buffered_hashes.append(record.text_sha256)
                errors.pop(record.video_id, None)
                new_successes += 1
                if len(buffered_ids) >= options.shard_size:
                    flush_buffer()

        try:
            for start in range(0, len(pending_records), options.batch_size):
                process_batch(pending_records[start : start + options.batch_size])
            flush_buffer()
        finally:
            if sampler is not None:
                gpu_mean, gpu_max = sampler.stop()
            if torch.cuda.is_available():
                peak_memory = int(torch.cuda.max_memory_allocated(torch.device(DEVICE)))

    elapsed = time.perf_counter() - started
    relevant_completed = completed_ids.intersection(selected_ids)
    run_record = {
        "started_at": run_started_at,
        "elapsed_seconds": elapsed,
        "limit": options.limit,
        "selected": len(selected),
        "metadata_matched": len(records),
        "pending_at_start": len(selected_ids - state.completed_ids),
        "new_successes": new_successes,
        "skipped": len(selected_ids.intersection(state.completed_ids)),
        "videos_per_second": new_successes / elapsed if elapsed else 0.0,
        "peak_cuda_memory_bytes": peak_memory,
        "gpu_utilization_mean_percent": gpu_mean,
        "gpu_utilization_max_percent": gpu_max,
        "device": DEVICE,
        "batch_size": options.batch_size,
        "max_tokens": options.max_tokens,
        "shard_size": options.shard_size,
    }
    state.manifest.setdefault("runs", []).append(run_record)
    state.manifest["last_run"] = run_record
    update_stats()
    checkpoint_manifest(options.output_dir, state.manifest)
    if not relevant_completed:
        raise RuntimeError("no videos were embedded successfully")
    return state.manifest
