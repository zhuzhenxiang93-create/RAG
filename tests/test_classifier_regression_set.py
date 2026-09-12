from __future__ import annotations

import json
from pathlib import Path

from legalmind.data.leakage_v3 import detect_explicit_leakage
from legalmind.data.privacy import scan_privacy

FIXTURE = Path(__file__).parent / "fixtures/classifier_regression_30.jsonl"


def test_classifier_regression_fixture_is_synthetic_safe_and_complete() -> None:
    rows = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 30
    assert len({row["case_id"] for row in rows}) == 30
    assert all(row["case_id"].startswith("synthetic-") for row in rows)
    assert all(row["fact"].strip() for row in rows)
    assert all(not any(scan_privacy(row["fact"]).values()) for row in rows)
    assert all(not any(detect_explicit_leakage(row["fact"]).values()) for row in rows)
    assert any(row["expect_abstain"] for row in rows)
    assert any(len(row["expected_accusations"]) > 1 for row in rows)
    assert any(row["forbidden_accusations"] for row in rows)
