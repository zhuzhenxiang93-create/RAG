from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from legalmind.generation.contracts_v2 import EvidencePacketV1
from legalmind.generation.grounding_validator import validate_generated_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    rows = []
    with args.data.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= args.max_rows:
                break

    details = []
    for row in rows:
        encoded = tokenizer.apply_chat_template(
            row["messages"][:-1],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        text = tokenizer.decode(generated[0, inputs.shape[1] :], skip_special_tokens=True)
        packet = EvidencePacketV1.model_validate(row["evidence_packet"])
        report = validate_generated_text(text, packet)
        predicted_disposition = None
        if report.get("schema_valid"):
            predicted_disposition = report["analysis"].disposition
        details.append(
            {
                "record_id": row["record_id"],
                "target_disposition": row["target"]["disposition"],
                "predicted_disposition": predicted_disposition,
                "schema_valid": bool(report.get("schema_valid")),
                "grounding_valid": bool(report.get("valid")),
                "unknown_evidence_ids": report.get("unknown_evidence_ids", []),
                "statute_validity_errors": report.get("statute_validity_errors", {}),
                "privacy_hits": report.get("privacy_hits", {}),
                "output_sha256": __import__("hashlib").sha256(text.encode()).hexdigest(),
            }
        )
    count = len(details)
    refusals = [row for row in details if row["target_disposition"] == "insufficient_evidence"]
    metrics = {
        "rows": count,
        "json_schema_rate": sum(row["schema_valid"] for row in details) / count,
        "grounding_valid_rate": sum(row["grounding_valid"] for row in details) / count,
        "unknown_citation_count": sum(len(row["unknown_evidence_ids"]) for row in details),
        "expired_or_invalid_statute_count": sum(
            len(row["statute_validity_errors"]) for row in details
        ),
        "privacy_hit_count": sum(sum(row["privacy_hits"].values()) for row in details),
        "refusal_recall": (
            sum(row["predicted_disposition"] == "insufficient_evidence" for row in refusals)
            / len(refusals)
            if refusals
            else None
        ),
    }
    thresholds = {
        "json_schema_rate": 1.0,
        "grounding_valid_rate": 0.95,
        "unknown_citation_count": 0,
        "expired_or_invalid_statute_count": 0,
        "privacy_hit_count": 0,
        "refusal_recall": 0.95,
    }
    passed = all(
        metrics[key] >= threshold
        if key.endswith("rate") or key == "refusal_recall"
        else metrics[key] <= threshold
        for key, threshold in thresholds.items()
    )
    result = {
        "evaluation_type": "synthetic_contract_smoke",
        "test_data_used_for_model_selection": False,
        "metrics": metrics,
        "thresholds": thresholds,
        "quality_gate_passed": passed,
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "details"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
