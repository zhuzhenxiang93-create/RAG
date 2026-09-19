from __future__ import annotations

from legalmind.generation.contracts_v2 import GroundedClaimV1, LegalAnalysisV1
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


class FakeGroundedAnalysisService:
    received = None

    def analyze(self, packet):
        self.received = packet
        case_id = next(item.evidence_id for item in packet.evidence if item.evidence_type == "case")
        statute_id = next(
            item.evidence_id for item in packet.evidence if item.evidence_type == "statute"
        )
        return (
            LegalAnalysisV1(
                disposition="analyzed",
                candidate_accusations=packet.predicted_accusations,
                key_facts=[packet.fact],
                legal_basis=[GroundedClaimV1(claim="适用检索法条", evidence_ids=[statute_id])],
                analogous_cases=[
                    GroundedClaimV1(claim="类案包含可比量刑标签", evidence_ids=[case_id])
                ],
                sentencing_assessment=[
                    GroundedClaimV1(claim="结合类案与基线分析量刑", evidence_ids=[case_id])
                ],
                confidence="medium",
                requires_manual_review=False,
            ),
            {"valid": True, "fallback_used": False, "attempts": 1},
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


def test_pipeline_sends_retrieval_and_sentencing_context_to_grounded_llm() -> None:
    case = SearchHit(
        chunk_id="CASE-1#0",
        case_id="CASE-1",
        text="盗窃后退赔并取得谅解",
        score=0.9,
        accusations=["盗窃"],
        penalty={"sentence_type": "fixed_term", "imprisonment_months": 8, "fine": 1000},
    )
    statute = SearchHit(
        chunk_id="LAW-264#0",
        case_id="LAW-264",
        text="刑法第二百六十四条",
        score=1.0,
        relevant_articles=[264],
        evidence_type="statute",
        source_url="https://www.gov.cn/law",
        effective_date="2021-03-01",
        legal_status="effective",
        source_status="human_verified_official",
    )
    grounded = FakeGroundedAnalysisService()
    result = LegalMindPipeline(
        retriever=FakeRetriever([case]),
        statute_retriever=FakeRetriever([statute]),
        sentencing_service=FakeSentencingService(),
        grounded_analysis_service=grounded,
    ).analyze("盗窃财物后退赔。", accusations=["盗窃"], as_of_date="2025-01-01")

    assert result.grounded_generation.status == "ok"
    assert result.grounded_generation.analysis is not None
    assert grounded.received.sentencing_baseline["status"] == "ok"
    case_evidence = next(
        item for item in grounded.received.evidence if item.evidence_type == "case"
    )
    assert case_evidence.penalty["imprisonment_months"] == 8
    assert "grounded_generation_seconds" in result.timings


def test_pipeline_has_explicit_cpu_degradation_status() -> None:
    result = LegalMindPipeline().analyze("仅用于测试的匿名案件事实。")
    assert result.classification.status == "degraded_no_classifier"
    assert result.retrieval.status == "degraded_no_index"
    assert result.retrieval.statute_status == "degraded_no_statute_index"
    assert result.analysis.requires_manual_review is True
    assert result.requires_manual_review is True
    assert result.grounded_generation.status == "disabled"


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


def test_fallback_probabilities_are_preserved_for_multi_partition_retrieval() -> None:
    sentencing = FakeSentencingService()
    result = LegalMindPipeline(
        classifier=FakeClassifier(fallback=True),
        retriever=FakePartitionedRetriever(),
        sentencing_service=sentencing,
    ).analyze("人工构造且长度足够的匿名盗窃案件事实，用于测试概率传递。")
    assert result.classification.labels[0].probability == 0.31
    assert len(result.classification.labels) == 2
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
