from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from legalmind.data.contracts_v3 import FineLabel, ProcessedCaseV3
from legalmind.data.dedup_v3 import duplicate_group_id, hamming_distance, simhash64
from legalmind.data.leakage_v3 import clean_input_fact, detect_explicit_leakage
from legalmind.data.money import parse_rmb
from legalmind.data.privacy import redact_privacy, scan_privacy
from legalmind.data.sources.cail import CailAdapter
from legalmind.data.sources.hrn import HrnAdapter
from legalmind.data.sources.judge import JudgeAdapter
from legalmind.data.sources.multijustice import MultiJusticeAdapter
from legalmind.data.temporal_split import temporal_group_split


@pytest.mark.parametrize(
    ("raw", "status", "amount"),
    [
        (None, "missing", None),
        (0, "zero", 0),
        ("一万元", "positive", 10000),
        ("1万", "positive", 10000),
        ("10000元", "positive", 10000),
        (-1, "invalid", None),
        ("币种不明", "invalid", None),
    ],
)
def test_chinese_money_semantics(raw, status, amount):
    parsed = parse_rmb(raw)
    assert (parsed.status.value, parsed.amount) == (status, amount)


def test_fine_contract_rejects_silent_invalid_values():
    with pytest.raises(ValidationError):
        FineLabel(status="positive", amount=0)
    with pytest.raises(ValidationError):
        FineLabel(status="missing", amount=0)


def base_case(**changes):
    value = {
        "case_id": "x",
        "source": "s",
        "source_case_id": "1",
        "source_schema_version": "1",
        "fact": "人工构造事实",
        "accusations": ["盗窃罪"],
        "relevant_articles": [],
        "sentence_type": "fixed_term",
        "imprisonment_months": 8,
        "probation": {"imposed": False, "months": None},
        "fine": {"status": "zero", "amount": 0},
        "death_penalty": False,
        "life_imprisonment": False,
        "label_quality": "test",
        "fine_quality": "test",
        "sentence_type_quality": "test",
        "privacy_status": "clear",
        "leakage_status": "clean",
        "duplicate_group_id": "dg",
        "source_case_group_id": "cg",
        "task_masks": {},
        "data_quality": {
            "date_available": False,
            "fine_available": True,
            "sentence_type_reliable": True,
            "target_leakage_detected": False,
        },
    }
    value.update(changes)
    return value


def test_death_and_life_never_get_fake_months():
    for kind in ("death", "life"):
        with pytest.raises(ValidationError):
            ProcessedCaseV3(
                **base_case(
                    sentence_type=kind,
                    imprisonment_months=999,
                    death_penalty=kind == "death",
                    life_imprisonment=kind == "life",
                )
            )


@pytest.mark.parametrize("kind", ["fixed_term", "detention", "control", "exempt", "unknown"])
def test_sentence_types_are_distinct(kind):
    months = (
        6 if kind in {"fixed_term", "detention", "control"} else 0 if kind == "exempt" else None
    )
    assert (
        ProcessedCaseV3(
            **base_case(sentence_type=kind, imprisonment_months=months)
        ).sentence_type.value
        == kind
    )


def test_probation_is_separate_from_principal_term():
    row = ProcessedCaseV3(**base_case(probation={"imposed": True, "months": 12}))
    assert row.imprisonment_months == 8 and row.probation.months == 12


def test_privacy_detection_and_redaction():
    text = "被告人张三，出生于1980年1月2日，住址北京市某区某路，手机号13812345678。"
    result = redact_privacy(text)
    assert sum(result.counts.values()) >= 3
    assert sum(scan_privacy(result.text).values()) == 0
    assert "13812345678" not in result.text


def test_leakage_removed_but_crime_features_retained():
    text = "被告人盗窃人民币一万元，三次作案，后自首并退赃。判处有期徒刑八个月，并处罚金一千元。"
    result = clean_input_fact(text)
    assert (
        "一万元" in result.text
        and "三次" in result.text
        and "自首" in result.text
        and "退赃" in result.text
    )
    assert sum(detect_explicit_leakage(result.text).values()) == 0


def test_duplicate_and_near_duplicate():
    assert duplicate_group_id("甲，乙。") == duplicate_group_id("甲 乙")
    assert (
        hamming_distance(simhash64("被告人实施盗窃并退赃"), simhash64("被告人实施盗窃并退赃。"))
        <= 3
    )


def test_temporal_split_keeps_case_defendants_together():
    rows = []
    for year in range(2010, 2020):
        for defendant in ("a", "b"):
            rows.append(
                {
                    "source_case_group_id": f"case-{year}",
                    "judgment_date": f"{year}-01-01",
                    "defendant": defendant,
                }
            )
    split = temporal_group_split(rows, 0.2, 0.2)
    locations = {}
    for name, values in split.items():
        for row in values:
            locations.setdefault(row["source_case_group_id"], set()).add(name)
    assert all(len(value) == 1 for value in locations.values())


def test_source_adapters(tmp_path):
    judge = tmp_path / "judge.json"
    judge.write_text(
        json.dumps([{"CaseId": "1", "Fact": "事实", "Fine": "一千元"}]), encoding="utf-8"
    )
    assert next(JudgeAdapter(judge).records())["fine_raw"] == "一千元"
    cail = tmp_path / "cail.jsonl"
    cail.write_text(
        json.dumps({"fact": "事实", "meta": {"accusation": ["盗窃"], "punish_of_money": 0}}) + "\n",
        encoding="utf-8",
    )
    assert next(CailAdapter(cail).records())["fine_raw"] == 0
    csv_path = tmp_path / "multi.csv"
    csv_path.write_text(
        "index,date,fact,accusation,relevant_article,imprisonment\n1,2020-01-01,事实,盗窃罪,264,8\n",
        encoding="utf-8",
    )
    assert next(MultiJusticeAdapter(csv_path).records())["judgment_date"] == "2020-01-01"


def test_hrn_per_defendant_alignment(tmp_path):
    import zipfile

    path = tmp_path / "hrn.zip"
    item = {
        "id": "c",
        "fact": "[被告A]与[被告B]共同作案",
        "criminals_info": {
            "[被告A]": {"accusations": ["甲罪"], "laws": [1], "term": "拘役三个月"},
            "[被告B]": {"accusations": ["乙罪"], "laws": [2], "term": "有期徒刑六个月"},
        },
    }
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("train.jsonl", json.dumps(item, ensure_ascii=False) + "\n")
    rows = list(HrnAdapter(path).records())
    assert len(rows) == 2 and all(row["alignment_reliable"] for row in rows)


def test_manifest_has_no_sensitive_fixture_data():
    assert "身份证" not in json.dumps(base_case(), ensure_ascii=False)
