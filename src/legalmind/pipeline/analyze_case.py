from __future__ import annotations

import argparse
import json
from pathlib import Path

from legalmind.config import load_yaml
from legalmind.data.labels import load_label_mapping
from legalmind.models.inference import ChargeClassifier
from legalmind.pipeline.core import LegalMindPipeline
from legalmind.retrieval.index import HybridIndex
from legalmind.retrieval.lexical import LexicalBM25Index
from legalmind.retrieval.partitioned import ChargePartitionedRetriever
from legalmind.retrieval.reranker import APIReranker, CrossEncoderReranker
from legalmind.retrieval.retriever import HybridRetriever


def build_pipeline(config: dict) -> LegalMindPipeline:
    initialization_warnings: list[str] = []
    classifier = None
    classifier_config = config.get("classifier", {})
    adapter = Path(classifier_config.get("adapter", ""))
    if classifier_config.get("enabled", True) and adapter.exists():
        mapping = load_label_mapping(classifier_config["label_mapping"])
        try:
            classifier = ChargeClassifier(
                classifier_config["base_model"],
                str(adapter),
                {value: key for key, value in mapping.items()},
                classifier_config.get("thresholds"),
                int(classifier_config.get("max_length", 2048)),
                classifier_config.get("truncation_strategy", "head_tail"),
            )
        except ImportError as error:
            initialization_warnings.append(
                f"charge_classifier_disabled_missing_dependency:{error.name or 'unknown'}"
            )
    retriever = None
    retrieval_config = config.get("retrieval", {})
    retrieval_mode = retrieval_config.get("mode", "legacy")
    hybrid_path = Path(retrieval_config.get("hybrid_index", ""))
    partitioned_path = Path(retrieval_config.get("partitioned_index", ""))
    index_path = Path(retrieval_config.get("bm25_index", ""))

    if retrieval_mode == "hybrid_rrf_rerank" and hybrid_path.exists():
        try:
            hybrid_index = HybridIndex.load(hybrid_path)
            reranker = None
            reranker_provider = retrieval_config.get("reranker_provider", "local")
            if reranker_provider == "local":
                reranker = CrossEncoderReranker(
                    retrieval_config.get("reranker_model", "Qwen/Qwen3-Reranker-0.6B"),
                    max_length=int(retrieval_config.get("reranker_max_length", 2048)),
                    batch_size=int(retrieval_config.get("reranker_batch_size", 8)),
                )
            elif reranker_provider == "api":
                reranker = APIReranker(
                    retrieval_config.get("reranker_model", "qwen3-rerank"),
                    retrieval_config.get("reranker_url_env", "DASHSCOPE_RERANK_URL"),
                    retrieval_config.get("reranker_api_key_env", "DASHSCOPE_API_KEY"),
                    instruction=retrieval_config.get("reranker_instruction", ""),
                    timeout=float(retrieval_config.get("reranker_timeout", 60)),
                )
            elif reranker_provider not in {None, "none", "disabled"}:
                raise ValueError(f"Unsupported reranker provider: {reranker_provider}")
            retriever = HybridRetriever(
                hybrid_index,
                reranker=reranker,
                rrf_k=int(retrieval_config.get("rrf_k", 60)),
                fusion_top_k=int(retrieval_config.get("fusion_top_k", 50)),
                rerank_top_k=int(retrieval_config.get("rerank_top_k", 20)),
                candidate_k=int(retrieval_config.get("candidate_k", 100)),
                final_k=int(retrieval_config.get("top_k", 3)),
            )
        except (ImportError, ValueError, FileNotFoundError) as error:
            initialization_warnings.append(
                f"hybrid_retrieval_disabled:{type(error).__name__}:{error}"
            )
    elif retrieval_mode == "hybrid_rrf_rerank":
        initialization_warnings.append(
            f"hybrid_retrieval_index_missing:{hybrid_path}"
        )

    # Keep the previous sparse routes only as explicit degradation paths.
    if retriever is None and partitioned_path.exists():
        retriever = ChargePartitionedRetriever.load(partitioned_path)
        if retrieval_mode == "hybrid_rrf_rerank":
            initialization_warnings.append("retrieval_fallback:partitioned_bm25")
    elif retriever is None and index_path.exists():
        retriever = LexicalBM25Index.load(index_path)
        if retrieval_mode == "hybrid_rrf_rerank":
            initialization_warnings.append("retrieval_fallback:bm25_only")
    statute_retriever = None
    statute_index_path = Path(config.get("retrieval", {}).get("statute_bm25_index", ""))
    if statute_index_path.exists():
        statute_retriever = LexicalBM25Index.load(statute_index_path)
    sentencing_service = None
    sentencing_config = config.get("sentencing", {})
    sentencing_model_dir = Path(sentencing_config.get("model_dir", ""))
    if sentencing_config.get("enabled", False) and sentencing_model_dir.exists():
        from legalmind.sentencing.evidence import SentencingEvidenceIndex
        from legalmind.sentencing.model import SentencingBaseline
        from legalmind.sentencing.service import SentencingService

        sentencing_model = SentencingBaseline.load(sentencing_model_dir)
        evidence_dir = Path(sentencing_config.get("evidence_dir", ""))
        sentencing_evidence = (
            SentencingEvidenceIndex.load(evidence_dir, sentencing_model)
            if evidence_dir.exists()
            else None
        )
        sentencing_service = SentencingService(
            sentencing_model,
            evidence_index=sentencing_evidence,
            charge_predictor=classifier,
        )
    grounded_analysis_service = None
    generation_config = config.get("generation", {})
    if generation_config.get("mode") == "openai_compatible_grounded":
        try:
            from legalmind.generation.analysis_service import GroundedAnalysisService
            from legalmind.generation.openai_generator import OpenAICompatibleGenerator

            grounded_analysis_service = GroundedAnalysisService(
                OpenAICompatibleGenerator(generation_config),
                max_repairs=int(generation_config.get("max_repairs", 1)),
            )
        except (ImportError, ValueError) as error:
            initialization_warnings.append(
                f"grounded_generation_disabled:{type(error).__name__}:{error}"
            )
    return LegalMindPipeline(
        classifier=classifier,
        retriever=retriever,
        statute_retriever=statute_retriever,
        sentencing_service=sentencing_service,
        grounded_analysis_service=grounded_analysis_service,
        initialization_warnings=initialization_warnings,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fact-file")
    parser.add_argument("--fact")
    parser.add_argument("--input-jsonl")
    parser.add_argument("--output-jsonl")
    parser.add_argument("--config", default="configs/pipeline/default.yaml")
    parser.add_argument("--as-of-date", help="法规适用日期，格式 YYYY-MM-DD")
    parser.add_argument("--accusation", action="append", default=[])
    parser.add_argument("--top-k", type=int, choices=range(1, 4), default=3)
    args = parser.parse_args()
    pipeline = build_pipeline(load_yaml(args.config))
    if args.input_jsonl:
        if not args.output_jsonl:
            parser.error("--output-jsonl is required with --input-jsonl")
        with (
            Path(args.input_jsonl).open(encoding="utf-8") as source,
            Path(args.output_jsonl).open("w", encoding="utf-8", newline="\n") as target,
        ):
            for line in source:
                if line.strip():
                    row = json.loads(line)
                    result = pipeline.analyze(
                        row["fact"],
                        accusations=row.get("accusations", args.accusation),
                        top_k=int(row.get("top_k", args.top_k)),
                        as_of_date=row.get("as_of_date", args.as_of_date),
                    )
                    target.write(result.model_dump_json() + "\n")
        return
    fact = args.fact or (
        Path(args.fact_file).read_text(encoding="utf-8") if args.fact_file else None
    )
    if not fact:
        parser.error("provide --fact, --fact-file, or --input-jsonl")
    print(
        pipeline.analyze(
            fact,
            accusations=args.accusation,
            top_k=args.top_k,
            as_of_date=args.as_of_date,
        ).model_dump_json(indent=2)
    )


if __name__ == "__main__":
    main()
