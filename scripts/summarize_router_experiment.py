"""Create a machine-readable and Markdown summary from real experiment artifacts."""

import argparse
from importlib import metadata
import json
from pathlib import Path
import platform
import sys


def load(path: Path):
    if not path.is_file():
        raise SystemExit(f"Required experiment artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def delta(after, before):
    return round(float(after) - float(before), 6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--base-router", type=Path, required=True)
    parser.add_argument("--lora-router", type=Path, required=True)
    parser.add_argument("--retrieval-all", type=Path, required=True)
    parser.add_argument("--retrieval-lora", type=Path, required=True)
    parser.add_argument("--retrieval-oracle", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    training = load(args.training)
    base = load(args.base_router)
    lora = load(args.lora_router)
    retrieval_all = load(args.retrieval_all)
    retrieval_lora = load(args.retrieval_lora)
    retrieval_oracle = load(args.retrieval_oracle)
    data_manifest = load(args.data_manifest)
    corpus_manifest = load(args.corpus_manifest)
    top_k_key = next(key for key in retrieval_all if key.startswith("recall@"))
    ndcg_key = next(key for key in retrieval_all if key.startswith("ndcg@"))

    try:
        import torch

        cuda = {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        cuda = {"torch": None, "cuda_available": False, "cuda_runtime": None, "gpu": None}
    packages = {}
    for name in ("transformers", "peft", "bitsandbytes", "datasets"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None

    summary = {
        "claim_policy": "Only values loaded from the listed artifacts are reported.",
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            **cuda,
            "packages": packages,
        },
        "data": {
            "train_samples": training["train_samples"],
            "validation_samples": training["validation_samples"],
            "benchmark_samples": data_manifest["counts"]["benchmark"],
            "leakage_check": data_manifest["leakage_check"],
            "retrieval_documents": retrieval_all["document_count"],
            "gold_document_coverage": corpus_manifest["gold_document_coverage"],
        },
        "lora": {
            "base_model": training["base_model"],
            "rank": training["rank"],
            "alpha": training["alpha"],
            "target_modules": training["target_modules"],
            "trainable_parameters": training["trainable_parameters"],
            "all_parameters": training["all_parameters"],
            "trainable_parameter_fraction_percent": training["trainable_fraction_percent"],
            "train_runtime_seconds": training.get("train_runtime"),
            "training_loss": training.get("train_loss"),
            "validation_loss": training.get("eval_loss"),
        },
        "router": {
            "base": {
                key: base[key]
                for key in (
                    "json_valid_rate",
                    "exact_route_accuracy",
                    "question_type_accuracy",
                    "question_type_macro_f1",
                    "source_micro_f1",
                    "source_macro_f1",
                )
            },
            "lora": {
                key: lora[key]
                for key in (
                    "json_valid_rate",
                    "exact_route_accuracy",
                    "question_type_accuracy",
                    "question_type_macro_f1",
                    "source_micro_f1",
                    "source_macro_f1",
                )
            },
        },
        "retrieval": {
            item["routing"]: {
                top_k_key: item[top_k_key],
                "mrr": item["mrr"],
                ndcg_key: item[ndcg_key],
                "latency_ms_mean": item["latency_ms_mean"],
            }
            for item in (retrieval_all, retrieval_lora, retrieval_oracle)
        },
    }
    summary["router"]["lora_minus_base"] = {
        key: delta(summary["router"]["lora"][key], summary["router"]["base"][key])
        for key in summary["router"]["base"]
    }
    summary["retrieval"]["lora_minus_all"] = {
        key: delta(summary["retrieval"]["lora"][key], summary["retrieval"]["all"][key])
        for key in (top_k_key, "mrr", ndcg_key)
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    router_rows = "\n".join(
        f"| {name} | {values['json_valid_rate']:.4f} | "
        f"{values['question_type_macro_f1']:.4f} | {values['source_micro_f1']:.4f} |"
        for name, values in (("Base zero-shot", summary["router"]["base"]), ("LoRA", summary["router"]["lora"]))
    )
    retrieval_rows = "\n".join(
        f"| {name} | {values[top_k_key]:.4f} | {values['mrr']:.4f} | "
        f"{values[ndcg_key]:.4f} | {values['latency_ms_mean']:.2f} |"
        for name, values in (
            ("All sources", summary["retrieval"]["all"]),
            ("LoRA", summary["retrieval"]["lora"]),
            ("Oracle", summary["retrieval"]["oracle"]),
        )
    )
    markdown = f"""# Enterprise Router 实验结果

本页由 `scripts/summarize_router_experiment.py` 从原始 JSON 产物生成。

## 数据与训练

- 训练 / 验证 / 官方测试：{summary['data']['train_samples']} / {summary['data']['validation_samples']} / {summary['data']['benchmark_samples']}
- 检索语料：{summary['data']['retrieval_documents']} 份；金标准文档覆盖率 {summary['data']['gold_document_coverage']:.4f}
- QLoRA：r={summary['lora']['rank']}，alpha={summary['lora']['alpha']}，可训练参数 {summary['lora']['trainable_parameters']:,}（{summary['lora']['trainable_parameter_fraction_percent']:.4f}%）
- 训练耗时：{summary['lora']['train_runtime_seconds']} 秒；训练 Loss={summary['lora']['training_loss']}；验证 Loss={summary['lora']['validation_loss']}

## 路由指标

| 方法 | JSON 合法率 | 问题类型 Macro-F1 | 来源 Micro-F1 |
|---|---:|---:|---:|
{router_rows}

## 下游 BM25 检索

| 路由 | {top_k_key} | MRR | {ndcg_key} | 平均延迟 ms |
|---|---:|---:|---:|---:|
{retrieval_rows}

所有数值均来自正式实验 JSON；建议把审计副本提交到
`results/enterprise_router_v2/`。未运行的实验不会出现在本页。
"""
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
