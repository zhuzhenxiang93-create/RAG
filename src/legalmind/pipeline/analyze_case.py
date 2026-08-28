from __future__ import annotations

import argparse
import json
from pathlib import Path

from legalmind.config import load_yaml
from legalmind.data.labels import load_label_mapping
from legalmind.models.inference import ChargeClassifier
from legalmind.pipeline.core import LegalMindPipeline
from legalmind.retrieval.lexical import LexicalBM25Index


def build_pipeline(config: dict) -> LegalMindPipeline:
    classifier = None
    classifier_config = config.get("classifier", {})
    adapter = Path(classifier_config.get("adapter", ""))
    if classifier_config.get("enabled", True) and adapter.exists():
        mapping = load_label_mapping(classifier_config["label_mapping"])
        classifier = ChargeClassifier(
            classifier_config["base_model"],
            str(adapter),
            {value: key for key, value in mapping.items()},
            classifier_config.get("thresholds"),
            int(classifier_config.get("max_length", 2048)),
        )
    retriever = None
    index_path = Path(config.get("retrieval", {}).get("bm25_index", ""))
    if index_path.exists():
        retriever = LexicalBM25Index.load(index_path)
    statute_retriever = None
    statute_index_path = Path(config.get("retrieval", {}).get("statute_bm25_index", ""))
    if statute_index_path.exists():
        statute_retriever = LexicalBM25Index.load(statute_index_path)
    return LegalMindPipeline(classifier, retriever, statute_retriever)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fact-file")
    parser.add_argument("--fact")
    parser.add_argument("--input-jsonl")
    parser.add_argument("--output-jsonl")
    parser.add_argument("--config", default="configs/pipeline/default.yaml")
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
                    target.write(
                        json.dumps(pipeline.analyze(row["fact"]), ensure_ascii=False) + "\n"
                    )
        return
    fact = args.fact or (
        Path(args.fact_file).read_text(encoding="utf-8") if args.fact_file else None
    )
    if not fact:
        parser.error("provide --fact, --fact-file, or --input-jsonl")
    print(json.dumps(pipeline.analyze(fact), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
