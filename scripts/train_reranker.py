from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

from legalmind.config import load_yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/retrieval/reranker_hard_negative_pilot.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    output = Path(config["output_dir"])
    existing_non_log_files = (
        [path for path in output.iterdir() if path.name != "logs"] if output.exists() else []
    )
    if existing_non_log_files:
        raise FileExistsError(f"Refusing to overwrite experiment: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    started = time.time()

    triplet_path = Path(config["triplets"])
    triplets = read_jsonl(triplet_path)
    rng = random.Random(int(config["seed"]))
    rng.shuffle(triplets)
    triplets = triplets[: int(config["max_triplets"])]

    query_by_id = {row["case_id"]: row["fact"] for row in read_jsonl(Path(config["train_cases"]))}
    candidate_by_id: dict[str, str] = {}
    for row in read_jsonl(Path(config["case_chunks"])):
        candidate_by_id.setdefault(row["case_id"], row["text"])

    examples: list[InputExample] = []
    missing = 0
    for row in triplets:
        query = query_by_id.get(row["query_id"])
        positive = candidate_by_id.get(row["positive_id"])
        negative = candidate_by_id.get(row["negative_id"])
        if not query or not positive or not negative:
            missing += 1
            continue
        examples.append(InputExample(texts=[query, positive], label=1.0))
        examples.append(InputExample(texts=[query, negative], label=0.0))
    if not examples:
        raise ValueError("No valid training pairs were resolved from the triplets")

    model = CrossEncoder(
        config["model_name"],
        max_length=int(config["max_length"]),
        trust_remote_code=True,
        prompts={"legal": config["instruction"]},
        default_prompt_name="legal",
        num_labels=1,
    )
    loader = DataLoader(
        examples,
        shuffle=True,
        batch_size=int(config["batch_size"]),
        generator=torch.Generator().manual_seed(int(config["seed"])),
    )
    warmup_steps = math.ceil(len(loader) * int(config["epochs"]) * float(config["warmup_ratio"]))
    model.fit(
        train_dataloader=loader,
        epochs=int(config["epochs"]),
        warmup_steps=warmup_steps,
        optimizer_params={"lr": float(config["learning_rate"])},
        output_path=str(output / "model"),
        save_best_model=False,
        # Qwen3-Reranker loads BF16 weights; FP16 GradScaler cannot unscale BF16 grads.
        use_amp=False,
    )
    elapsed = time.time() - started
    manifest = {
        "status": "completed",
        "scope": config["scope"],
        "model": config["model_name"],
        "model_license": config["model_license"],
        "train_triplets": len(triplets),
        "training_pairs": len(examples),
        "unresolved_triplets": missing,
        "source_split": "train_only",
        "test_used": False,
        "review_status": "unreviewed",
        "seed": int(config["seed"]),
        "max_length": int(config["max_length"]),
        "batch_size": int(config["batch_size"]),
        "epochs": int(config["epochs"]),
        "learning_rate": float(config["learning_rate"]),
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024**3
        if torch.cuda.is_available()
        else 0.0,
        "data_sha256": {
            "triplets": sha256_file(triplet_path),
            "train_cases": sha256_file(Path(config["train_cases"])),
            "case_chunks": sha256_file(Path(config["case_chunks"])),
        },
    }
    (output / "config.yaml").write_text(
        Path(args.config).read_text(encoding="utf-8"), encoding="utf-8"
    )
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
