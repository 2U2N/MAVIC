from __future__ import annotations

import logging
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .decoding import VideoDataset, collate_prepared
from .embedding import aggregate_video_batch
from .persistence import (
    MANIFEST_NAME,
    atomic_json,
    load_state,
    reset_generated_output,
    utc_now,
    write_shard,
)
from .sampling import discover_videos

LOGGER = logging.getLogger(__name__)


class _GpuUtilizationSampler:
    def __init__(self, device: torch.device, interval_seconds: float = 0.5) -> None:
        self.device = device
        self.interval_seconds = interval_seconds
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.device.type != "cuda" or not hasattr(torch.cuda, "utilization"):
            return
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.samples.append(float(torch.cuda.utilization(self.device)))
            except Exception:
                # CUDA inference should not fail because NVML utilization is unavailable.
                return

    def stop(self) -> tuple[float | None, float | None]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds * 2)
        if not self.samples:
            return None, None
        return statistics.fmean(self.samples), max(self.samples)


@dataclass(frozen=True)
class ProcessingConfig:
    # OpenAI's original ViT-B/32 checkpoint was trained with QuickGELU.
    model: str = "ViT-B-32-quickgelu"
    pretrained: str = "openai"
    samples: int = 16
    duplicate_threshold: float = 0.97
    precision: str = "bf16"


@dataclass
class RunOptions:
    input_dir: Path
    output_dir: Path
    videos_per_batch: int = 32
    workers: int = 8
    device: str = "cuda"
    shard_size: int = 1000
    overwrite: bool = False


def _validate_options(config: ProcessingConfig, options: RunOptions) -> torch.device:
    if config.samples <= 0:
        raise ValueError("--samples must be positive")
    if not -1.0 <= config.duplicate_threshold <= 1.0:
        raise ValueError("--duplicate-threshold must be in [-1, 1]")
    if options.videos_per_batch <= 0 or options.shard_size <= 0:
        raise ValueError("batch and shard sizes must be positive")
    if options.workers < 0:
        raise ValueError("--workers cannot be negative")
    if config.precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError("--precision must be bf16, fp16, or fp32")

    device = torch.device(options.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available; use --device cpu for debugging")
        if config.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("this CUDA device does not support BF16; select --precision fp16")
    elif config.precision != "fp32":
        raise ValueError("CPU execution requires --precision fp32")
    return device


def _autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _flush(
    output_dir: Path,
    shard_index: int,
    ids: list[str],
    embeddings: list[np.ndarray],
    counts: list[int],
) -> dict[str, Any]:
    return write_shard(
        output_dir,
        shard_index,
        ids,
        np.stack(embeddings),
        np.asarray(counts, dtype=np.uint8),
    )


def run_pipeline(config: ProcessingConfig, options: RunOptions) -> dict[str, Any]:
    device = _validate_options(config, options)
    videos = discover_videos(options.input_dir)
    if not videos:
        raise ValueError(f"no MP4 videos found in {options.input_dir}")

    options.output_dir.mkdir(parents=True, exist_ok=True)
    if options.overwrite:
        reset_generated_output(options.output_dir)
    config_dict = asdict(config)
    state = load_state(options.output_dir, config_dict)
    pending = [path for path in videos if path.stem not in state.completed_ids]

    LOGGER.info(
        "discovered %d videos (%d complete, %d pending)",
        len(videos),
        len(videos) - len(pending),
        len(pending),
    )
    started = time.perf_counter()
    run_started_at = utc_now()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if not pending:
        elapsed = time.perf_counter() - started
        current_ids = {path.stem for path in videos}
        relevant_completed = state.completed_ids.intersection(current_ids)
        run_record = {
            "started_at": run_started_at,
            "elapsed_seconds": elapsed,
            "pending": 0,
            "new_successes": 0,
            "skipped": len(videos),
            "videos_per_second": 0.0,
            "peak_cuda_memory_bytes": 0 if device.type == "cuda" else None,
            "gpu_utilization_mean_percent": None,
            "gpu_utilization_max_percent": None,
            "device": str(device),
            "videos_per_batch": options.videos_per_batch,
            "workers": options.workers,
        }
        state.manifest.setdefault("runs", []).append(run_record)
        state.manifest.update(
            {
                "updated_at": utc_now(),
                "errors": {},
                "stats": {
                    "discovered": len(videos),
                    "succeeded": len(relevant_completed),
                    "failed": 0,
                },
                "last_run": run_record,
            }
        )
        atomic_json(options.output_dir / MANIFEST_NAME, state.manifest)
        if not relevant_completed:
            raise RuntimeError("no videos were embedded successfully")
        return state.manifest

    # Import lazily so discovery, consolidation, and unit tests do not load OpenCLIP.
    import open_clip

    gpu_sampler = _GpuUtilizationSampler(device)
    gpu_sampler.start()

    model, _, preprocess = open_clip.create_model_and_transforms(
        config.model, pretrained=config.pretrained, device=device
    )
    model.eval()

    dataset = VideoDataset(pending, preprocess=preprocess, samples=config.samples)
    loader_options: dict[str, Any] = {
        "batch_size": options.videos_per_batch,
        "shuffle": False,
        "num_workers": options.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": options.workers > 0,
        "collate_fn": collate_prepared,
    }
    if options.workers > 0:
        # Never fork worker processes after CUDA has been initialized.
        loader_options["multiprocessing_context"] = "spawn"
    loader = torch.utils.data.DataLoader(dataset, **loader_options)

    buffered_ids: list[str] = []
    buffered_embeddings: list[np.ndarray] = []
    buffered_counts: list[int] = []
    new_successes = 0
    errors = dict(state.manifest.get("errors", {}))
    shard_index = state.next_shard_index

    def flush_buffer(limit: int | None = None) -> None:
        nonlocal shard_index
        take = len(buffered_ids) if limit is None else min(limit, len(buffered_ids))
        if take == 0:
            return
        shard = _flush(
            options.output_dir,
            shard_index,
            buffered_ids[:take],
            buffered_embeddings[:take],
            buffered_counts[:take],
        )
        state.manifest["shards"].append(shard)
        shard_index += 1
        del buffered_ids[:take]
        del buffered_embeddings[:take]
        del buffered_counts[:take]

    with torch.inference_mode():
        for batch in loader:
            batch_errors = batch["errors"]
            errors.update(batch_errors)
            frames = batch["frames"]
            ids = batch["video_ids"]
            if frames is None:
                continue
            try:
                video_count, frame_count = frames.shape[:2]
                flat_frames = frames.flatten(0, 1).to(device, non_blocking=device.type == "cuda")
                with _autocast(device, config.precision):
                    encoded = model.encode_image(flat_frames)
                encoded = encoded.reshape(video_count, frame_count, -1).float()
                pooled, retained = aggregate_video_batch(encoded, config.duplicate_threshold)
                pooled_np = pooled.cpu().numpy().astype(np.float32, copy=False)
                retained_np = retained.cpu().numpy()
                for index, video_id in enumerate(ids):
                    buffered_ids.append(video_id)
                    buffered_embeddings.append(pooled_np[index])
                    buffered_counts.append(int(retained_np[index]))
                    errors.pop(video_id, None)
                    new_successes += 1
                while len(buffered_ids) >= options.shard_size:
                    flush_buffer(options.shard_size)
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                LOGGER.exception("embedding batch failed for %d videos", len(ids))
                errors.update({video_id: message for video_id in ids})
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    flush_buffer()
    elapsed = time.perf_counter() - started
    current_ids = {path.stem for path in videos}
    completed_ids = state.completed_ids.union(
        video_id
        for shard in state.manifest["shards"]
        for video_id in _ids_in_shard(options.output_dir / shard["file"])
    )
    relevant_completed = completed_ids.intersection(current_ids)
    relevant_errors = {key: value for key, value in errors.items() if key in current_ids - relevant_completed}
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    gpu_mean, gpu_max = gpu_sampler.stop()
    run_record = {
        "started_at": run_started_at,
        "elapsed_seconds": elapsed,
        "pending": len(pending),
        "new_successes": new_successes,
        "skipped": len(videos) - len(pending),
        "videos_per_second": new_successes / elapsed if elapsed else 0.0,
        "peak_cuda_memory_bytes": peak_memory,
        "gpu_utilization_mean_percent": gpu_mean,
        "gpu_utilization_max_percent": gpu_max,
        "device": str(device),
        "videos_per_batch": options.videos_per_batch,
        "workers": options.workers,
    }
    state.manifest.setdefault("runs", []).append(run_record)
    state.manifest.update(
        {
            "updated_at": utc_now(),
            "errors": relevant_errors,
            "stats": {
                "discovered": len(videos),
                "succeeded": len(relevant_completed),
                "failed": len(relevant_errors),
            },
            "last_run": run_record,
        }
    )
    atomic_json(options.output_dir / MANIFEST_NAME, state.manifest)
    if not relevant_completed:
        raise RuntimeError("no videos were embedded successfully")
    return state.manifest


def _ids_in_shard(path: Path) -> list[str]:
    with np.load(path, allow_pickle=False) as data:
        return [str(value) for value in data["video_ids"].tolist()]
