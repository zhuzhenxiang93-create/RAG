from legalmind.models.peft_classifier import adapter_method


def test_school_config_mode_is_lora() -> None:
    config = {"quantization": {"load_in_4bit": False}}
    assert adapter_method(config) == "lora"


def test_historical_quantized_mode_remains_qlora() -> None:
    config = {"quantization": {"load_in_4bit": True}}
    assert adapter_method(config) == "qlora"


def test_missing_quantization_defaults_to_lora() -> None:
    assert adapter_method({}) == "lora"
