from types import SimpleNamespace

import pytest

from legalmind.models.loading import configure_padding


def test_configure_padding_copies_tokenizer_id_to_model() -> None:
    tokenizer = SimpleNamespace(
        pad_token=None,
        eos_token="<eos>",
        pad_token_id=151643,
    )
    model = SimpleNamespace(config=SimpleNamespace(pad_token_id=None))

    configure_padding(tokenizer, model)

    assert tokenizer.pad_token == "<eos>"
    assert model.config.pad_token_id == 151643


def test_configure_padding_rejects_tokenizer_without_id() -> None:
    tokenizer = SimpleNamespace(pad_token=None, eos_token=None, pad_token_id=None)
    model = SimpleNamespace(config=SimpleNamespace(pad_token_id=None))

    with pytest.raises(ValueError, match="pad_token_id"):
        configure_padding(tokenizer, model)
