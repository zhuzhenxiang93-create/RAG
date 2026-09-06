from pathlib import Path

from legalmind.retrieval.partitioned import ChargePartitionedRetriever
from legalmind.schemas import CaseChunk, LabelScore


def chunk(case_id: str, text: str, accusations: list[str]) -> CaseChunk:
    return CaseChunk(
        chunk_id=f"{case_id}#0",
        case_id=case_id,
        text=text,
        chunk_index=0,
        start_char=0,
        end_char=len(text),
        accusations=accusations,
    )


def label(name: str, probability: float) -> LabelScore:
    return LabelScore(label_id=0, label=name, probability=probability)


def test_partition_blocks_other_accusations() -> None:
    retriever = ChargePartitionedRetriever(min_score=0.0)
    retriever.build(
        [
            chunk("THEFT", "秘密窃取手机价值五千元", ["盗窃"]),
            chunk("FRAUD", "虚构事实骗取手机价值五千元", ["诈骗"]),
        ]
    )
    hits = retriever.search_by_accusations("窃取手机价值5000元", [label("盗窃罪", 0.8)])
    assert hits
    assert {hit.case_id for hit in hits} == {"THEFT"}


def test_amount_and_factor_rerank() -> None:
    retriever = ChargePartitionedRetriever(min_score=0.0)
    retriever.build(
        [
            chunk("NEAR", "秘密窃取财物价值一万元并退赔", ["盗窃"]),
            chunk("FAR", "秘密窃取财物价值一百万元", ["盗窃"]),
        ]
    )
    hits = retriever.search_by_accusations(
        "秘密窃取财物价值1万元，案发后退赔", [label("盗窃", 0.9)], top_k=2
    )
    assert hits[0].case_id == "NEAR"


def test_normalized_duplicates_are_collapsed_and_labels_merged() -> None:
    retriever = ChargePartitionedRetriever(min_score=0.0)
    counts = retriever.build(
        [
            chunk("A", "窃取 手机。", ["盗窃"]),
            chunk("B", "窃取手机", ["诈骗"]),
        ]
    )
    assert counts == {"盗窃": 1, "诈骗": 1}
    assert set(retriever.partitions["盗窃"].chunks[0].accusations) == {"盗窃", "诈骗"}


def test_probability_weighted_quota_and_top_three() -> None:
    retriever = ChargePartitionedRetriever(min_score=0.0)
    retriever.build(
        [
            chunk("T1", "窃取甲手机价值一万元", ["盗窃"]),
            chunk("T2", "窃取乙手机价值一万元", ["盗窃"]),
            chunk("F1", "虚构事实骗取手机价值一万元", ["诈骗"]),
        ]
    )
    hits = retriever.search_by_accusations(
        "取得手机价值一万元", [label("盗窃", 0.8), label("诈骗", 0.2)], top_k=3
    )
    assert len(hits) == 3
    assert sum("盗窃" in hit.accusations for hit in hits) == 2
    assert sum("诈骗" in hit.accusations for hit in hits) == 1


def test_save_load_and_minimum_threshold(tmp_path: Path) -> None:
    retriever = ChargePartitionedRetriever(min_score=1.1)
    retriever.build([chunk("A", "秘密窃取手机价值五千元", ["盗窃"])])
    retriever.save(tmp_path / "index")
    loaded = ChargePartitionedRetriever.load(tmp_path / "index")
    assert loaded.search_by_accusations("窃取手机", [label("盗窃", 0.9)]) == []
