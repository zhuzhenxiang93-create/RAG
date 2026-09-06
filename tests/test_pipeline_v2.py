from __future__ import annotations

from legalmind.pipeline.contracts import LegalCaseAnalysisResponse
from legalmind.pipeline.core import LegalMindPipeline
from legalmind.schemas import ClassificationResult, LabelScore, SearchHit
from legalmind.sentencing.schemas import SentencingResponse


class FakeRetriever:
    def __init__(self, hits: list[SearchHit]):
        self.hits = hits

    def search(self, _query: str, top_k: int = 5) -> list[SearchHit]:
        return self.hits[:top_k]


class FakeSentencingService:
    received: dict | None = None

    def predict(self, value: dict) -> SentencingResponse:
        assert value["fact"]
        self.received = value
        return SentencingResponse(status="ok", confidence=0.7)


class FakeClassifier:
    def __init__(self, *, fallback: bool = False):
        self.fallback = fallback

    def predict(self, _fact: str) -> ClassificationResult:
        return ClassificationResult(
            labels=[
                LabelScore(label_id=1, label="盗窃", probability=0.31),
                LabelScore(label_id=2, label="诈骗", probability=0.22),
            ],
            used_fallback=self.fallback,
            max_probability=0.31,
        )


class FakePartitionedRetriever:
    def search_by_accusations(self, _query, accusations, top_k=3):
        assert accusations[0].probability == 0.31
        return [
            SearchHit(
                chunk_id="THEFT#0",
                case_id="THEFT",
                text="匿名盗窃事实",
                score=0.8,
                accusations=["盗窃"],
            )
        ][:top_k]


def test_pipeline_keeps_case_and_statute_evidence_separate() -> None:
    case = SearchHit(
        chunk_id="CASE-1#0",
        case_id="CASE-1",
        text="持刀伤人并造成轻伤",
        score=1.0,
        accusations=["故意伤害"],
    )
    statute = SearchHit(
        chunk_id="PRC-CRIMINAL-LAW-234#article",
        case_id="PRC-CRIMINAL-LAW-234",
        text="刑法第二百三十四条",
        score=1.0,
        relevant_articles=[234],
        evidence_type="statute",
        source_url="https://www.gov.cn/official-law",
        effective_date="2021-03-01",
        legal_status="effective",
        source_status="human_verified_official",
    )
    result = LegalMindPipeline(
        retriever=FakeRetriever([case]),
        statute_retriever=FakeRetriever([statute]),
    ).analyze("被告人持刀将他人刺成轻伤。", as_of_date="2025-01-01")
    assert isinstance(result, LegalCaseAnalysisResponse)
    assert result.retrieval.evidence[0].case_id == "CASE-1"
    assert result.retrieval.statutes[0].relevant_articles == [234]
    assert result.analysis.relevant_articles == [234]
    assert result.citation_validation.valid is True
    assert result.evidence_firewall.passed is True
    assert result.legal_as_of_date == "2025-01-01"


def test_pipeline_has_explicit_cpu_degradation_status() -> None:
    result = LegalMindPipeline().analyze("仅用于测试的匿名案件事实。")
    assert result.classification.status == "degraded_no_classifier"
    assert result.retrieval.status == "degraded_no_index"
    assert result.retrieval.statute_status == "degraded_no_statute_index"
    assert result.analysis.requires_manual_review is True
    assert result.requires_manual_review is True
    assert "grounded_analysis" not in LegalCaseAnalysisResponse.model_fields


def test_pipeline_rejects_unverified_or_future_statute() -> None:
    future_statute = SearchHit(
        chunk_id="LAW#1",
        case_id="LAW-1",
        text="尚未生效的规则",
        score=1.0,
        relevant_articles=[1],
        evidence_type="statute",
        source_url="https://www.gov.cn/law",
        effective_date="2030-01-01",
        legal_status="effective",
        source_status="human_verified_official",
    )
    result = LegalMindPipeline(statute_retriever=FakeRetriever([future_statute])).analyze(
        "仅用于测试的匿名案件事实。", as_of_date="2029-12-31"
    )
    assert result.retrieval.statutes == []
    assert result.retrieval.statute_status == "no_verified_applicable_statute"
    assert "applicable_statute_present" in result.evidence_firewall.blocking_reasons
    assert any(
        "not_yet_effective" in item.reasons for item in result.evidence_firewall.rejected_statutes
    )


def test_pipeline_exposes_structured_sentencing_result() -> None:
    result = LegalMindPipeline(sentencing_service=FakeSentencingService()).analyze(
        "人工构造且长度足够的匿名案件事实，用于验证结构化量刑服务已经进入主流水线。"
    )
    assert result.sentencing.status == "ok"
    assert result.sentencing.confidence == 0.7
    assert "sentencing_seconds" in result.timings


def test_pipeline_exposes_initialization_degradation() -> None:
    result = LegalMindPipeline(
        initialization_warnings=["charge_classifier_disabled_missing_dependency:peft"]
    ).analyze("人工构造且长度足够的匿名案件事实，仅用于测试依赖缺失时的显式降级。")
    assert result.classification.status == "degraded_no_classifier"
    assert result.initialization_warnings == ["charge_classifier_disabled_missing_dependency:peft"]


def test_pipeline_never_returns_direct_identifiers_in_case_evidence() -> None:
    case = SearchHit(
        chunk_id="SAFE#1",
        case_id="SAFE",
        text=(
            "被告人张三，手机号13812345678，住址：测试省测试市测试路88号。"
            "身份证号110101199001011234。"
        ),
        score=1.0,
    )
    result = LegalMindPipeline(retriever=FakeRetriever([case])).analyze("人工匿名测试案情。")
    returned = result.retrieval.evidence[0]
    assert "张三" not in returned.text
    assert "13812345678" not in returned.text
    assert "110101199001011234" not in returned.text
    assert returned.presentation == "deidentified_summary"
    assert returned.privacy_redaction_counts["identity_card"] == 1


def test_fallback_probability_is_preserved_and_only_top_label_flows_downstream() -> None:
    sentencing = FakeSentencingService()
    result = LegalMindPipeline(
        classifier=FakeClassifier(fallback=True),
        retriever=FakePartitionedRetriever(),
        sentencing_service=sentencing,
    ).analyze("人工构造且长度足够的匿名盗窃案件事实，用于测试概率传递。")
    assert result.classification.labels[0].probability == 0.31
    assert len(result.classification.labels) == 1
    assert sentencing.received is not None
    assert sentencing.received["predicted_accusations"][0].probability == 0.31
    assert result.retrieval.evidence[0].accusations == ["盗窃"]


def test_user_provided_accusation_overrides_classifier() -> None:
    result = LegalMindPipeline(classifier=FakeClassifier()).analyze(
        "人工构造且长度足够的匿名案件事实，用于测试用户覆盖罪名。",
        accusations=["故意伤害"],
    )
    assert result.classification.status == "provided"
    assert result.classification.labels[0].label == "故意伤害"
    assert result.classification.labels[0].probability == 1.0
