from __future__ import annotations

import argparse
import json
from pathlib import Path

from legalmind.generation.pilot_data import build_pilot_dataset, build_review_queue


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--review-queue", type=Path)
    args = parser.parse_args()
    manifest = build_pilot_dataset(args.output, args.rows, args.seed)
    if args.review_queue:
        manifest["review_queue_rows"] = build_review_queue(
            args.output, args.review_queue, min(200, args.rows), args.seed
        )
        manifest["review_queue"] = str(args.review_queue)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
