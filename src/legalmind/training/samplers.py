from __future__ import annotations

import random
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Shuffle coarse buckets, then batch examples with similar token lengths."""

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        seed: int = 42,
        bucket_size_multiplier: int = 50,
        drop_last: bool = False,
    ) -> None:
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.seed = seed
        self.bucket_size = max(batch_size, batch_size * bucket_size_multiplier)
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        rng.shuffle(indices)
        batches: list[list[int]] = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            bucket.sort(key=self.lengths.__getitem__)
            for batch_start in range(0, len(bucket), self.batch_size):
                batch = bucket[batch_start : batch_start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        quotient, remainder = divmod(len(self.lengths), self.batch_size)
        return quotient if self.drop_last or not remainder else quotient + 1
