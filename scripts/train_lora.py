"""Canonical classifier training entrypoint for the school BF16 LoRA run.

The implementation remains shared with the historical PEFT trainer so old QLoRA
experiments stay reproducible. The selected method is controlled by
quantization.load_in_4bit in the model config.
"""

from train_qlora import main


if __name__ == "__main__":
    main()
