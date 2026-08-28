from __future__ import annotations

import os
from typing import Any


class OpenAICompatibleGenerator:
    """Client for a separately deployed Qwen3.5 or compatible generation service."""

    def __init__(self, config: dict[str, Any]):
        from openai import OpenAI

        base_url = os.getenv(config.get("base_url_env", "GENERATION_BASE_URL"))
        api_key = os.getenv(config.get("api_key_env", "GENERATION_API_KEY"), "EMPTY")
        if not base_url:
            raise ValueError(
                "Generation base URL is missing from the configured environment variable"
            )
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = config["model"]
        self.max_tokens = int(config.get("max_tokens", 4096))
        self.temperature = float(config.get("temperature", 0.7))
        self.top_p = float(config.get("top_p", 0.8))
        self.top_k = int(config.get("top_k", 20))
        self.presence_penalty = float(config.get("presence_penalty", 1.5))

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            presence_penalty=self.presence_penalty,
            extra_body={"top_k": self.top_k},
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Generation service returned an empty response")
        return content
