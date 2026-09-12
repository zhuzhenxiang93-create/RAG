from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Sampler


def load_token_lengths(path: str | Path, expected_rows: int) -> list[int]:
    values = np.load(Path(path), allow_pickle=False)
    if values.ndim != 1 or len(values) != expected_rows:
        raise ValueError(
            f"Token-length cache has shape {values.shape}; expected ({expected_rows},)"
        )
    if np.any(values <= 0):
        raise ValueError("Token-length cache contains non-positive values")
    return [int(value) for value in values]



def partial_batch_loss_scale(actual_batch_size: int, target_batch_size: int) -> float:
    if not 0 < actual_batch_size <= target_batch_size:
        raise ValueError("actual_batch_size must be in [1, target_batch_size]")
    return actual_batch_size / target_batch_size


class TokenBucketSampler(Sampler[int]):
    """Shuffle fixed-size batches within explicit token-length buckets."""

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        boundaries: Sequence[int],
        seed: int = 42,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not lengths:
            raise ValueError("lengths must not be empty")
        if any(length <= 0 for length in lengths):
            raise ValueError("all token lengths must be positive")
        normalized = [int(value) for value in boundaries]
        if normalized != sorted(set(normalized)):
            raise ValueError("boundaries must be unique and sorted")
        self.lengths = [int(value) for value in lengths]
        self.batch_size = int(batch_size)
        self.boundaries = normalized
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.lengths)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        buckets: list[list[int]] = [[] for _ in range(len(self.boundaries) + 1)]
        for index, length in enumerate(self.lengths):
            buckets[bisect_right(self.boundaries, length)].append(index)

        batches: list[list[int]] = []
        leftovers: list[int] = []
        for bucket in buckets:
            if not bucket:
                continue
            order = torch.randperm(len(bucket), generator=generator).tolist()
            shuffled = [bucket[position] for position in order]
            full_end = len(shuffled) - len(shuffled) % self.batch_size
            batches.extend(
                shuffled[start : start + self.batch_size]
                for start in range(0, full_end, self.batch_size)
            )
            leftovers.extend(shuffled[full_end:])

        if leftovers:
            order = torch.randperm(len(leftovers), generator=generator).tolist()
            leftovers = [leftovers[position] for position in order]
            batches.extend(
                leftovers[start : start + self.batch_size]
                for start in range(0, len(leftovers), self.batch_size)
            )

        batch_order = torch.randperm(len(batches), generator=generator).tolist()
        self.epoch += 1
        return iter(index for position in batch_order for index in batches[position])
