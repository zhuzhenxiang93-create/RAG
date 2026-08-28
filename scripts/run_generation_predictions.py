from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from legalmind.evaluation.evaluate_generation import evaluate


def _extract_json(text: str) -> str:
    value = text.strip()
    if "</think>" in value:
        value = value.split("</think>", 1)[1].strip()
    if value.startswith("```json"):
        value = value[7:]
    elif value.startswith("```"):
        value = value[3:]
    value = value.removesuffix("```")
    return value.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--adapter")
    parser.add_argument("--input", default="data/sft_v2/validation.jsonl")
    parser.add_argument("--label-mapping", default="data/processed_v2/label_mapping.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-rows", type=int, default=20)
    parser.add_argument("--max-input-tokens", type=int, default=1664)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite experiment: {output}")
    output.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    started = time.perf_counter()
    predictions_path = output / "predictions.jsonl"
    rows = 0
    with (
        Path(args.input).open(encoding="utf-8") as source,
        predictions_path.open("w", encoding="utf-8", newline="\n") as target,
    ):
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            encoded = tokenizer.apply_chat_template(
                row["messages"][:-1],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            if encoded["input_ids"].shape[1] > args.max_input_tokens:
                encoded = {
                    key: value[:, -args.max_input_tokens :] for key, value in encoded.items()
                }
            input_length = encoded["input_ids"].shape[1]
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            text = tokenizer.decode(generated[0, input_length:], skip_special_tokens=True)
            target.write(
                json.dumps(
                    {
                        "case_id": row["case_id"],
                        "output": _extract_json(text),
                        "raw_output": text,
                        "reference_accusations": row["target"]["predicted_accusations"],
                        "retrieved_case_ids": [],
                        "retrieved_articles": [],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            rows += 1
            if rows >= args.max_rows:
                break
    label_mapping = json.loads(Path(args.label_mapping).read_text(encoding="utf-8"))
    metrics = evaluate(predictions_path, label_mapping)
    manifest = {
        "status": "completed",
        "scope": "smoke",
        "model": args.model,
        "adapter": args.adapter,
        "rows": rows,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "metrics": metrics,
        "limitations": [
            "Smoke subset only; not a full generation evaluation.",
            "No retrieved statute or case evidence is supplied in this smoke run.",
            "Validation targets are deterministic and unreviewed.",
        ],
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
