from __future__ import annotations

import numpy as np
import pytest

from legalmind.training.token_bucket import (
    TokenBucketSampler,
    load_token_lengths,
    partial_batch_loss_scale,
)


def test_token_bucket_sampler_emits_each_row_once() -> None:
    lengths = [20, 200, 300, 500, 700, 900, 1100, 1800, 2500] * 7
    sampler = TokenBucketSampler(lengths, batch_size=8, boundaries=[256, 512, 1024, 2048])
    indices = list(sampler)

    assert len(indices) == len(lengths)
    assert sorted(indices) == list(range(len(lengths)))


def test_token_bucket_sampler_is_seeded_and_changes_by_epoch() -> None:
    lengths = list(range(1, 101))
    first = TokenBucketSampler(lengths, batch_size=8, boundaries=[25, 50, 75], seed=9)
    second = TokenBucketSampler(lengths, batch_size=8, boundaries=[25, 50, 75], seed=9)

    first_epoch = list(first)
    assert first_epoch == list(second)
    assert first_epoch != list(first)


def test_load_token_lengths_validates_alignment(tmp_path) -> None:
    cache = tmp_path / "lengths.npy"
    np.save(cache, np.asarray([12, 24, 48], dtype=np.int32))

    assert load_token_lengths(cache, 3) == [12, 24, 48]
    with pytest.raises(ValueError, match="expected"):
        load_token_lengths(cache, 2)



def test_partial_batch_loss_scale_preserves_per_case_weight() -> None:
    assert partial_batch_loss_scale(8, 8) == 1.0
    assert partial_batch_loss_scale(4, 8) == 0.5
