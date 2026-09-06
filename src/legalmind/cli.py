from __future__ import annotations

import argparse
import json
import platform
import time

from legalmind.config import load_yaml
from legalmind.pipeline.analyze_case import build_pipeline
from legalmind.sentencing.evaluate import evaluate, write_evaluation
from legalmind.sentencing.evidence import SentencingEvidenceIndex
from legalmind.sentencing.io import read_jsonl
from legalmind.sentencing.model import SentencingBaseline


def train(args: argparse.Namespace) -> None:
    started = time.time()
    rows = read_jsonl(args.train_file, args.limit)
    model = SentencingBaseline(random_state=args.seed)
    metadata = model.fit(rows)
    metadata.update(
        {
            "dataset_file": args.train_file,
            "training_seconds": round(time.time() - started, 3),
            "python": platform.python_version(),
        }
    )
    model.save(args.output_dir)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def build_index(args: argparse.Namespace) -> None:
    model = SentencingBaseline.load(args.model_dir)
    rows = read_jsonl(args.train_file, args.limit)
    index = SentencingEvidenceIndex(model)
    index.build(rows)
    index.save(args.output_dir)
    print(json.dumps({"rows": len(rows), "output_dir": args.output_dir}, ensure_ascii=False))


def run_evaluation(args: argparse.Namespace) -> None:
    model = SentencingBaseline.load(args.model_dir)
    rows = read_jsonl(args.test_file, args.limit)
    report = evaluate(model, rows)
    report["model_manifest"] = model.metadata
    write_evaluation(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def analyze(args: argparse.Namespace) -> None:
    pipeline = build_pipeline(load_yaml(args.config))
    response = pipeline.analyze(
        args.fact,
        accusations=args.accusation,
        top_k=args.top_k,
        as_of_date=args.as_of_date,
    )
    print(response.model_dump_json(indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="legalmind")
    commands = root.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train-sentencing")
    train_parser.add_argument("--train-file", default="data/processed_v2_1_1/train.jsonl")
    train_parser.add_argument("--output-dir", default="artifacts/models/sentencing_baseline_v1_2")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--limit", type=int)
    train_parser.set_defaults(function=train)

    index_parser = commands.add_parser("build-sentencing-index")
    index_parser.add_argument("--train-file", default="data/processed_v2_1_1/train.jsonl")
    index_parser.add_argument("--model-dir", default="artifacts/models/sentencing_baseline_v1_2")
    index_parser.add_argument("--output-dir", default="artifacts/indexes/sentencing_evidence_v1_2")
    index_parser.add_argument("--limit", type=int)
    index_parser.set_defaults(function=build_index)

    evaluation_parser = commands.add_parser("evaluate-sentencing")
    evaluation_parser.add_argument("--test-file", default="data/processed_v2_1_1/test.jsonl")
    evaluation_parser.add_argument(
        "--model-dir", default="artifacts/models/sentencing_baseline_v1_2"
    )
    evaluation_parser.add_argument("--output", default="reports/sentencing/evaluation_v1_1.json")
    evaluation_parser.add_argument("--limit", type=int)
    evaluation_parser.set_defaults(function=run_evaluation)

    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("--fact", required=True)
    analyze_parser.add_argument("--accusation", action="append", default=[])
    analyze_parser.add_argument("--top-k", type=int, choices=range(1, 4), default=3)
    analyze_parser.add_argument("--as-of-date")
    analyze_parser.add_argument("--config", default="configs/pipeline/default.yaml")
    analyze_parser.set_defaults(function=analyze)
    return root


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
