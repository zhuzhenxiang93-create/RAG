"""Evaluate either the base model or a LoRA adapter on held-out router data."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.plugins.enterprise_router.data import read_jsonl
from app.plugins.enterprise_router.metrics import routing_metrics
from app.plugins.enterprise_router.parsing import parse_route
from app.plugins.enterprise_router.schema import RouteDecision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the base model in NF4 for evaluation on limited VRAM.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use only the local Hugging Face cache and perform no network checks.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=Path("artifacts/router-evaluation.json"))
    args = parser.parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    try:
        import torch
        from huggingface_hub import snapshot_download
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise SystemExit(
            "Install full dependencies first: pip install -r requirements-full.txt"
        ) from exc

    device = (
        "cuda:0"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    model_source = (
        snapshot_download(args.base_model, local_files_only=True)
        if args.offline
        else args.base_model
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=True,
        local_files_only=args.offline,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        if args.load_in_4bit
        else None
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map="auto" if args.load_in_4bit else None,
        local_files_only=args.offline,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, str(args.adapter))
    if not args.load_in_4bit:
        model = model.to(device)
    model = model.eval()

    rows = list(read_jsonl(args.data))
    if args.limit:
        rows = rows[: args.limit]
    expected = []
    valid_expected = []
    predicted = []
    predictions = []
    latencies = []
    for row in rows:
        gold = RouteDecision.model_validate(row["route"])
        expected.append(gold)
        encoded = tokenizer(
            row["prompt"],
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        latency = (time.perf_counter() - started) * 1000.0
        latencies.append(latency)
        completion = tokenizer.decode(
            generated[0][encoded["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        try:
            route, canonical = parse_route(completion)
            valid_expected.append(gold)
            predicted.append(route)
            error = None
        except ValueError as exc:
            canonical = None
            error = str(exc)
        predictions.append(
            {
                "id": row["id"],
                "question": row["question"],
                "gold": gold.model_dump(),
                "prediction": canonical,
                "raw_completion": completion,
                "parse_error": error,
                "latency_ms": round(latency, 3),
            }
        )
    result = routing_metrics(
        valid_expected,
        predicted,
        attempted_count=len(expected),
    )
    result.update(
        {
            "method": "lora" if args.adapter else "base_zero_shot",
            "base_model": args.base_model,
            "adapter": str(args.adapter) if args.adapter else None,
            "load_in_4bit": args.load_in_4bit,
            "offline": args.offline,
            "split": str(args.data),
            "limited": args.limit is not None,
            "latency_ms_mean": round(sum(latencies) / len(latencies), 3),
            "predictions": predictions,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary_keys = (
        "method",
        "sample_count",
        "json_valid_rate",
        "exact_route_accuracy",
        "question_type_accuracy",
        "source_micro_f1",
    )
    print(json.dumps({key: result[key] for key in summary_keys}, indent=2))


if __name__ == "__main__":
    main()
