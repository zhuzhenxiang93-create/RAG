"""Deterministic tokenizer for mixed Chinese, English and identifiers."""

import re
from typing import List

_segments = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/@+-]*|[\u4e00-\u9fff]+")


def tokenize(text: str) -> List[str]:
    """Return lowercase words plus Chinese unigrams and bigrams."""
    tokens: List[str] = []
    for segment in _segments.findall(text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", segment):
            characters = list(segment)
            tokens.extend(characters)
            tokens.extend(
                characters[index] + characters[index + 1]
                for index in range(len(characters) - 1)
            )
        else:
            tokens.append(segment)
    return tokens
