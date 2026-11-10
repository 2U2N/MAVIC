from __future__ import annotations

import torch
import torch.nn.functional as F


def aggregate_frame_embeddings(
    frame_embeddings: torch.Tensor, duplicate_threshold: float = 0.97
) -> tuple[torch.Tensor, int]:
    """Greedily remove near duplicates, then mean-pool and normalize."""
    if frame_embeddings.ndim != 2 or frame_embeddings.shape[0] == 0:
        raise ValueError("frame_embeddings must have shape [frames, dimensions]")
    if not -1.0 <= duplicate_threshold <= 1.0:
        raise ValueError("duplicate_threshold must be in [-1, 1]")

    normalized = F.normalize(frame_embeddings.float(), dim=-1)
    kept: list[torch.Tensor] = []
    for embedding in normalized:
        if not kept:
            kept.append(embedding)
            continue
        similarities = torch.stack(kept) @ embedding
        if bool(torch.all(similarities < duplicate_threshold)):
            kept.append(embedding)
    pooled = F.normalize(torch.stack(kept).mean(dim=0), dim=0)
    return pooled, len(kept)


def aggregate_video_batch(
    frame_embeddings: torch.Tensor, duplicate_threshold: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if frame_embeddings.ndim != 3:
        raise ValueError("frame_embeddings must have shape [videos, frames, dimensions]")
    pooled: list[torch.Tensor] = []
    counts: list[int] = []
    for video_embeddings in frame_embeddings:
        video_embedding, count = aggregate_frame_embeddings(video_embeddings, duplicate_threshold)
        pooled.append(video_embedding)
        counts.append(count)
    return torch.stack(pooled), torch.tensor(counts, dtype=torch.uint8)
