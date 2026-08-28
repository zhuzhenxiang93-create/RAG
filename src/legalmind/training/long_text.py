from __future__ import annotations

import re
from typing import Any, Literal

_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？；!?;])")
_FACT_KEYWORDS = (
    "持刀",
    "殴打",
    "盗窃",
    "骗取",
    "抢劫",
    "死亡",
    "重伤",
    "轻伤",
    "金额",
    "价值",
    "自首",
    "投案",
    "累犯",
    "赔偿",
    "谅解",
    "故意",
    "过失",
)


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=True))


def head_tail_truncate(tokenizer: Any, text: str, max_length: int) -> str:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_length:
        return text
    head_size = max_length * 2 // 3
    tail_size = max_length - head_size
    selected = token_ids[:head_size] + token_ids[-tail_size:]
    return tokenizer.decode(selected, skip_special_tokens=True)


def sentence_fact_select(tokenizer: Any, text: str, max_length: int) -> str:
    sentences = [value.strip() for value in _SENTENCE_BOUNDARY.split(text) if value.strip()]
    if token_count(tokenizer, text) <= max_length or len(sentences) <= 1:
        return text
    scored = []
    for index, sentence in enumerate(sentences):
        keyword_score = sum(keyword in sentence for keyword in _FACT_KEYWORDS)
        boundary_score = 1 if index in {0, len(sentences) - 1} else 0
        scored.append((keyword_score * 10 + boundary_score, -index, index, sentence))
    selected: set[int] = set()
    budget = max_length - 8
    used = 0
    for _, _, index, sentence in sorted(scored, reverse=True):
        length = token_count(tokenizer, sentence)
        if selected and used + length > budget:
            continue
        if length > budget:
            sentence = head_tail_truncate(tokenizer, sentence, budget)
            length = token_count(tokenizer, sentence)
            sentences[index] = sentence
        selected.add(index)
        used += length
        if used >= budget:
            break
    return "".join(sentences[index] for index in sorted(selected))


def truncate_fact(
    tokenizer: Any,
    text: str,
    max_length: int,
    strategy: Literal["head", "head_tail", "sentence"] = "head_tail",
) -> str:
    if strategy == "head":
        ids = tokenizer.encode(text, add_special_tokens=False)[:max_length]
        return tokenizer.decode(ids, skip_special_tokens=True)
    if strategy == "head_tail":
        return head_tail_truncate(tokenizer, text, max_length)
    if strategy == "sentence":
        return sentence_fact_select(tokenizer, text, max_length)
    raise ValueError(f"Unsupported truncation strategy: {strategy}")
