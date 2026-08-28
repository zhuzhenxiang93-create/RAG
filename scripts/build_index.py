from __future__ import annotations

import argparse
import json
from pathlib import Path

from legalmind.config import load_yaml
from legalmind.data.normalize import normalize_record
from legalmind.retrieval.chunker import ChineseCaseChunker
from legalmind.retrieval.index import HybridIndex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/retrieval.yaml")
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()
    config = load_yaml(args.config)
    from transformers import AutoTokenizer

    embedding_tokenizer = AutoTokenizer.from_pretrained(
        config["chunk_tokenizer_model"],
        use_fast=True,
    )
    chunker = ChineseCaseChunker(
        int(config["chunk_size_tokens"]),
        int(config["chunk_overlap_tokens"]),
        tokenizer=embedding_tokenizer,
    )
    chunks = []
    case_count = 0
    with Path(config["source"]).open("r", encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle):
            if args.max_cases and index >= args.max_cases:
                break
            case = normalize_record(json.loads(line), source_split="train")
            chunks.extend(chunker.split(case))
            case_count += 1
    hybrid_index = HybridIndex(
        config["embedding_model"],
        embedding_provider=config.get("embedding_provider", "local"),
        query_instruction=config.get("query_instruction", ""),
        max_length=int(config.get("embedding_max_length", 2048)),
        embedding_dimension=config.get("embedding_dimension"),
        base_url_env=config.get("embedding_base_url_env", "DASHSCOPE_BASE_URL"),
        api_key_env=config.get("embedding_api_key_env", "DASHSCOPE_API_KEY"),
        api_batch_size=int(config.get("embedding_api_batch_size", 20)),
    )
    hybrid_index.build(chunks, int(config.get("batch_size", 64)))
    hybrid_index.save(config["output_dir"])
    print(f"indexed cases={case_count} chunks={len(chunks)}")


if __name__ == "__main__":
    main()
