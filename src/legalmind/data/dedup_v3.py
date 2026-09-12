from __future__ import annotations

import hashlib
import re
import unicodedata


def normalized_duplicate_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[\s\W_]+", "", value)


def duplicate_group_id(text: str) -> str:
    value = normalized_duplicate_text(text)
    return "DG-" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def simhash64(text: str) -> int:
    value = normalized_duplicate_text(text)
    grams = [value[i : i + 3] for i in range(max(1, len(value) - 2))]
    vector = [0] * 64
    for gram in grams:
        digest = int.from_bytes(hashlib.blake2b(gram.encode(), digest_size=8).digest(), "big")
        for bit in range(64):
            vector[bit] += 1 if digest & (1 << bit) else -1
    return sum((1 << bit) for bit, score in enumerate(vector) if score >= 0)


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()
