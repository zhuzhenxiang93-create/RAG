from __future__ import annotations

from typing import Any


def configure_padding(tokenizer: Any, model: Any) -> None:
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id or eos_token_id")
    model.config.pad_token_id = tokenizer.pad_token_id
