import numpy as np
import pytest
import torch

from mavic.representations.openclip.embedding import aggregate_frame_embeddings, aggregate_video_batch


def test_greedy_dedup_compares_every_retained_frame() -> None:
    frames = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.999, 0.001, 0.0],
        ]
    )
    pooled, count = aggregate_frame_embeddings(frames, duplicate_threshold=0.97)
    assert count == 2
    assert torch.isclose(torch.linalg.vector_norm(pooled), torch.tensor(1.0))


def test_similarity_equal_to_threshold_is_removed() -> None:
    # Orthogonal vectors have exact similarity 0; equality must count as duplicate.
    frames = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    _, count = aggregate_frame_embeddings(frames, duplicate_threshold=0.0)
    assert count == 1


def test_pooling_is_float32_and_normalized() -> None:
    frames = torch.tensor([[4.0, 0.0], [0.0, 2.0]], dtype=torch.float16)
    pooled, count = aggregate_frame_embeddings(frames, duplicate_threshold=0.97)
    assert count == 2
    assert pooled.dtype == torch.float32
    np.testing.assert_allclose(pooled.numpy(), [2**-0.5, 2**-0.5], rtol=1e-5)


def test_video_batch_returns_counts() -> None:
    frames = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )
    pooled, counts = aggregate_video_batch(frames, duplicate_threshold=0.97)
    assert pooled.shape == (2, 2)
    assert counts.tolist() == [1, 2]
    assert counts.dtype == torch.uint8


def test_embedding_validation() -> None:
    with pytest.raises(ValueError):
        aggregate_frame_embeddings(torch.empty(0, 512))
    with pytest.raises(ValueError):
        aggregate_frame_embeddings(torch.ones(1, 2), duplicate_threshold=1.1)
