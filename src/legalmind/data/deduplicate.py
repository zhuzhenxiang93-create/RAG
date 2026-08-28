from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable

from legalmind.data.normalize import normalize_text


def exact_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_fingerprint(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def character_ngrams(text: str, size: int = 3) -> list[str]:
    compact = re.sub(r"\s+", "", normalize_text(text))
    if len(compact) <= size:
        return [compact] if compact else []
    return [compact[index : index + size] for index in range(len(compact) - size + 1)]


def simhash64(text: str) -> int:
    vector = [0] * 64
    for token in character_ngrams(text):
        value = int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")
        for bit in range(64):
            vector[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, score in enumerate(vector):
        if score >= 0:
            result |= 1 << bit
    return result


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def near_duplicate_candidates(
    rows: Iterable[tuple[str, str]], threshold: int = 3, max_pairs: int = 10_000
) -> list[dict[str, int | str]]:
    """Find candidates with four 16-bit LSH bands; intended for review, not deletion."""
    buckets: dict[tuple[int, int], list[tuple[str, int]]] = defaultdict(list)
    pairs: list[dict[str, int | str]] = []
    seen: set[tuple[str, str]] = set()
    for row_id, text in rows:
        fingerprint = simhash64(text)
        candidates: dict[str, int] = {}
        for band in range(4):
            key = (band, (fingerprint >> (band * 16)) & 0xFFFF)
            for candidate_id, candidate_hash in buckets[key]:
                candidates[candidate_id] = candidate_hash
        for candidate_id, candidate_hash in candidates.items():
            pair = tuple(sorted((row_id, candidate_id)))
            if pair in seen:
                continue
            distance = hamming_distance(fingerprint, candidate_hash)
            if distance <= threshold:
                pairs.append({"left_id": pair[0], "right_id": pair[1], "hamming_distance": distance})
                seen.add(pair)
                if len(pairs) >= max_pairs:
                    return pairs
        for band in range(4):
            key = (band, (fingerprint >> (band * 16)) & 0xFFFF)
            buckets[key].append((row_id, fingerprint))
    return pairs
