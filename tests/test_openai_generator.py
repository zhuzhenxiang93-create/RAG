from __future__ import annotations

import builtins
from unittest.mock import Mock

from legalmind.generation.openai_generator import OpenAICompatibleGenerator


def test_requests_fallback_calls_openai_compatible_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("GENERATION_BASE_URL", "https://example.test/v1/")
    monkeypatch.setenv("GENERATION_API_KEY", "test-secret")
    original_import = builtins.__import__

    def import_without_openai(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("openai unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_openai)
    response = Mock()
    response.json.return_value = {"choices": [{"message": {"content": '{"ok":true}'}}]}
    monkeypatch.setattr("requests.post", Mock(return_value=response))

    generator = OpenAICompatibleGenerator({"model": "qwen-plus"})

    assert generator.generate("prompt") == '{"ok":true}'
    response.raise_for_status.assert_called_once_with()
    requests_post = __import__("requests").post
    _, kwargs = requests_post.call_args
    assert kwargs["json"]["model"] == "qwen-plus"
    assert kwargs["headers"]["Authorization"] == "Bearer test-secret"
    assert requests_post.call_args.args[0] == "https://example.test/v1/chat/completions"
