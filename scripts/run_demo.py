from __future__ import annotations

import argparse
import json

from legalmind.config import load_yaml
from legalmind.data.labels import load_label_mapping
from legalmind.generation.openai_generator import OpenAICompatibleGenerator
from legalmind.generation.prompts import build_grounded_prompt
from legalmind.models.inference import ChargeClassifier
from legalmind.retrieval.index import HybridIndex
from legalmind.retrieval.reranker import APIReranker, CrossEncoderReranker
from legalmind.retrieval.retriever import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Run classification and traceable case retrieval.")
    parser.add_argument("--fact", required=True)
    parser.add_argument("--model-config", default="configs/model_qwen3_4b.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--retrieval-config", default="configs/retrieval.yaml")
    parser.add_argument("--generation-config", default="configs/generation.yaml")
    parser.add_argument("--label-mapping", default="data/manifests/label_mapping.json")
    parser.add_argument("--adapter")
    parser.add_argument("--thresholds", default="artifacts/results/classifier/thresholds.json")
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    model_config = load_yaml(args.model_config)
    training = load_yaml(args.training_config)
    retrieval = load_yaml(args.retrieval_config)
    label_to_id = load_label_mapping(args.label_mapping)
    id_to_label = {label_id: label for label, label_id in label_to_id.items()}
    classifier = ChargeClassifier(
        base_model=model_config["name_or_path"],
        adapter_path=args.adapter or training["output_dir"],
        id_to_label=id_to_label,
        thresholds_path=args.thresholds,
        max_length=int(training["max_length"]),
    )
    index = HybridIndex.load(retrieval["output_dir"])
    if retrieval.get("reranker_provider", "local") == "api":
        reranker = APIReranker(
            retrieval["reranker_model"],
            retrieval.get("reranker_url_env", "DASHSCOPE_RERANK_URL"),
            retrieval.get("reranker_api_key_env", "DASHSCOPE_API_KEY"),
            instruction=retrieval.get("reranker_instruction", ""),
            timeout=float(retrieval.get("reranker_timeout", 60)),
        )
    else:
        reranker = CrossEncoderReranker(
            retrieval["reranker_model"],
            max_length=int(retrieval.get("reranker_max_length", 2048)),
            batch_size=int(retrieval.get("reranker_batch_size", 8)),
        )
    retriever = HybridRetriever(
        index,
        reranker=reranker,
        rrf_k=int(retrieval.get("rrf_k", 60)),
        label_boost=float(retrieval.get("label_boost", 0.15)),
        fusion_top_k=int(retrieval.get("fusion_top_k", 50)),
        rerank_top_k=int(retrieval.get("rerank_top_k", 20)),
    )
    classification = classifier.predict(args.fact)
    predicted_labels = {item.label for item in classification.labels}
    evidence = retriever.search(
        args.fact,
        predicted_labels=predicted_labels,
        candidate_k=max(int(retrieval["bm25_top_k"]), int(retrieval["vector_top_k"])),
        final_k=int(retrieval["final_top_k"]),
    )
    result = {
        "classification": classification.model_dump(),
        "evidence": [item.model_dump() for item in evidence],
    }
    if args.generate:
        generator = OpenAICompatibleGenerator(load_yaml(args.generation_config))
        result["report"] = generator.generate(build_grounded_prompt(args.fact, evidence))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
