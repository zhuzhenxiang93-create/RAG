"""Generate a reproducible, benchmark-independent enterprise router dataset.

The output follows EnterpriseRAG-Bench's question schema but uses the fictional
company Northstar Labs. No official Redwood question text or document is read.
"""

import argparse
import json
from pathlib import Path
import random


SOURCES = {
    "slack": ("Slack", "discussion", "channel"),
    "gmail": ("Gmail", "email", "mailbox"),
    "linear": ("Linear", "issue", "project board"),
    "google_drive": ("Google Drive", "document", "shared drive"),
    "hubspot": ("HubSpot", "customer record", "CRM"),
    "fireflies": ("Fireflies", "meeting transcript", "call archive"),
    "github": ("GitHub", "pull request", "repository"),
    "jira": ("Jira", "ticket", "sprint board"),
    "confluence": ("Confluence", "knowledge page", "wiki"),
}
QUESTION_TYPES = (
    "basic",
    "semantic",
    "intra_document_reasoning",
    "project_related",
    "constrained",
    "conflicting_info",
    "completeness",
    "miscellaneous",
    "high_level",
    "info_not_found",
)
PRODUCTION_SHAPED_COUNTS = {
    "basic": 800,
    "semantic": 500,
    "intra_document_reasoning": 140,
    "project_related": 140,
    "constrained": 120,
    "conflicting_info": 60,
    "completeness": 60,
    "miscellaneous": 80,
    "high_level": 40,
    "info_not_found": 60,
}
TOPICS = (
    "Atlas checkout migration",
    "Beacon observability rollout",
    "Cedar enterprise renewal",
    "Drift mobile release",
    "Echo data-retention policy",
    "Falcon inference gateway",
    "Harbor onboarding program",
    "Iris incident response",
    "Juniper pricing experiment",
    "Kepler model evaluation",
    "Lumen security review",
    "Mosaic support escalation",
)
TEAMS = (
    "platform",
    "applied AI",
    "security",
    "customer success",
    "sales engineering",
    "product",
    "site reliability",
    "finance",
)
TIME_WINDOWS = (
    "during Q1 2026",
    "before the April launch",
    "in the last two sprints",
    "between January and March 2026",
    "after the production incident",
)
ACCOUNT_ADJECTIVES = (
    "Amber", "Blue", "Cobalt", "Delta", "Evergreen", "Frost", "Golden", "Helix",
    "Indigo", "Jade", "Kinetic", "Lunar", "Metro", "Nova", "Orchid", "Pacific",
    "Quartz", "River", "Solar", "Titan",
)
ACCOUNT_NOUNS = (
    "Analytics", "Bank", "Cloud", "Dynamics", "Energy", "Foods", "Group", "Health",
    "Industries", "Junction", "Labs", "Markets", "Networks", "Operations", "Partners",
    "Retail", "Systems", "Telecom", "University", "Ventures",
)

TEMPLATES = {
    "basic": (
        "According to the {artifact} in {product}, who owns {topic}?",
        "What deadline is recorded for {topic} in the {product} {artifact}?",
        "Which threshold does the {artifact} specify for {topic}?",
        "What status is currently listed for {topic} in {product}?",
    ),
    "semantic": (
        "Where did the {team} team explain why {topic} was delayed?",
        "Find the {product} material describing the rationale behind {topic}.",
        "What does the {artifact} imply is the main risk to {topic}?",
        "Locate the discussion that means {topic} cannot ship as originally planned.",
    ),
    "intra_document_reasoning": (
        "Using the milestones and dependencies in one {product} {artifact}, when can {topic} safely launch?",
        "Based on the figures in the {artifact}, what is the estimated impact of {topic}?",
        "Which option best satisfies all requirements listed for {topic} in the same {artifact}?",
        "What conclusion follows from the timeline and acceptance criteria for {topic}?",
    ),
    "project_related": (
        "Summarize the decisions, open work, owners, and risks for {topic} across {products}.",
        "Trace {topic} from the original request through implementation and rollout using {products}.",
        "What is the complete current state of {topic} across {products}?",
        "Combine the planning, execution, and stakeholder updates for {topic} from {products}.",
    ),
    "constrained": (
        "For {topic}, list only items owned by {team} {window} in {product}.",
        "Which {topic} actions in {product} are still open, exclude completed work, and include owners?",
        "Find {topic} records {window} with severity high or critical in {product}.",
        "Return only customer-facing decisions about {topic} approved by {team} in {product}.",
    ),
    "conflicting_info": (
        "{product1} and {product2} appear to give different dates for {topic}; identify the conflict and the latest authoritative value.",
        "Compare the owner recorded for {topic} in {product1} versus {product2} and resolve the discrepancy.",
        "Which source is current when {product1} and {product2} disagree about the status of {topic}?",
        "Explain the inconsistent requirements for {topic} across {product1} and {product2}.",
    ),
    "completeness": (
        "Find every recorded decision and follow-up for {topic} across {products}; do not omit closed items.",
        "List all customer commitments connected to {topic} across {products}.",
        "Collect the complete set of owners, deadlines, blockers, and outcomes for {topic} from {products}.",
        "Which records across {products} mention {topic}, including superseded or archived items?",
    ),
    "miscellaneous": (
        "Who should I contact about {topic}, and where is the supporting context in {product}?",
        "Give me the most useful {product} record to understand {topic} before a stakeholder meeting.",
        "Is there a recent update about {topic} in {product}, and what should I pay attention to?",
        "Help me orient myself to {topic} using the available {product} material.",
    ),
    "high_level": (
        "What are Northstar Labs' major strategic risks and priorities {window}?",
        "Give an executive view of how the company is performing across product, engineering, and customers {window}.",
        "What recurring themes appear across company work and stakeholder communication {window}?",
        "Summarize the organization's overall progress, trade-offs, and unresolved concerns {window}.",
    ),
    "info_not_found": (
        "What is the approved acquisition price for Northstar Labs' purchase of Zephyr Robotics?",
        "Which office will Northstar Labs open on Mars next quarter?",
        "What final dividend was authorized for shareholders in 2032?",
        "Who won the internal quantum teleportation patent dispute?",
    ),
}


def _source_count(question_type: str, index: int) -> int:
    if question_type in {"project_related", "completeness"}:
        return 2 + index % 3
    if question_type == "conflicting_info":
        return 2
    if question_type in {"high_level", "info_not_found"}:
        return 0
    return 1


def build_rows_from_counts(counts, seed: int):
    rng = random.Random(seed)
    source_names = list(SOURCES)
    rows = []
    for type_index, question_type in enumerate(QUESTION_TYPES):
        for index in range(counts[question_type]):
            count = _source_count(question_type, index)
            selected = rng.sample(source_names, count) if count else []
            primary = selected[0] if selected else source_names[index % len(source_names)]
            secondary = selected[1] if len(selected) > 1 else source_names[(index + 1) % len(source_names)]
            product, artifact, _ = SOURCES[primary]
            product2 = SOURCES[secondary][0]
            products = ", ".join(SOURCES[source][0] for source in selected)
            topic = TOPICS[(index * 5 + type_index) % len(TOPICS)]
            team = TEAMS[(index * 3 + type_index) % len(TEAMS)]
            window = TIME_WINDOWS[(index + type_index) % len(TIME_WINDOWS)]
            account = (
                f"{ACCOUNT_ADJECTIVES[index % len(ACCOUNT_ADJECTIVES)]} "
                f"{ACCOUNT_NOUNS[(index // len(ACCOUNT_ADJECTIVES)) % len(ACCOUNT_NOUNS)]}"
            )
            if index >= len(ACCOUNT_ADJECTIVES) * len(ACCOUNT_NOUNS):
                account += f" Division {1 + index // (len(ACCOUNT_ADJECTIVES) * len(ACCOUNT_NOUNS))}"
            template = TEMPLATES[question_type][index % len(TEMPLATES[question_type])]
            question = template.format(
                product=product,
                product1=product,
                product2=product2,
                products=products or "all enterprise sources",
                artifact=artifact,
                topic=topic,
                team=team,
                window=window,
            )
            # Deterministic context qualifiers create lexical diversity without
            # copying or paraphrasing the held-out benchmark.
            if index % 4 == 1 and question_type not in {"high_level", "info_not_found"}:
                question += " Use the latest updated record."
            elif index % 4 == 2 and question_type not in {"high_level", "info_not_found"}:
                question += " Include the supporting owner and date."
            elif index % 4 == 3 and question_type not in {"high_level", "info_not_found"}:
                question += " Prefer authoritative records over informal mentions."
            question += f" Business context: {account}."
            doc_count = (
                0
                if question_type == "info_not_found"
                else max(1, count)
                if question_type not in {"project_related", "conflicting_info", "completeness", "high_level"}
                else max(2, count)
            )
            rows.append(
                {
                    "question_id": f"northstar_{type_index:02d}_{index:04d}",
                    "question_type": question_type,
                    "source_types": selected,
                    "question": question,
                    "expected_doc_ids": [
                        f"northstar_doc_{type_index:02d}_{index:04d}_{doc_index}"
                        for doc_index in range(doc_count)
                    ],
                    "generation": {
                        "company": "Northstar Labs",
                        "method": "deterministic_templates_v1",
                        "seed": seed,
                    },
                }
            )
    rng.shuffle(rows)
    return rows


def build_rows(per_type: int, seed: int):
    return build_rows_from_counts(
        {question_type: per_type for question_type in QUESTION_TYPES}, seed
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-type", type=int, default=180)
    parser.add_argument(
        "--profile",
        choices=("balanced", "production-shaped"),
        default="balanced",
        help="production-shaped uses a documented synthetic traffic prior.",
    )
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()
    if args.per_type < 4:
        raise SystemExit("--per-type must be at least 4")
    counts = (
        PRODUCTION_SHAPED_COUNTS
        if args.profile == "production-shaped"
        else {question_type: args.per_type for question_type in QUESTION_TYPES}
    )
    rows = build_rows_from_counts(counts, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "count": len(rows),
                "profile": args.profile,
                "question_type_counts": counts,
                "company": "Northstar Labs",
                "benchmark_text_used": False,
                "seed": args.seed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
