from legalmind.data.prepare import (
    canonicalize_label,
    is_low_information_fact,
    merge_case_labels,
)
from legalmind.schemas import CaseRecord


def test_canonicalize_legacy_composite_label() -> None:
    assert canonicalize_label("[走私、贩卖、运输、制造]毒品") == "走私、贩卖、运输、制造毒品"
    assert canonicalize_label("非法[持有、私藏][枪支、弹药]") == "非法持有、私藏枪支、弹药"


def test_low_information_filter_only_applies_to_short_boilerplate() -> None:
    boilerplate = "公诉机关指控的事实与庭审查明的事实基本一致。"
    assert is_low_information_fact(boilerplate)
    assert is_low_information_fact("经审理查明，原判认定的事实清楚，证据确实、充分，本院予以确认。")
    assert is_low_information_fact("公诉机关指控的事实与本院认定的事实基本一致。")
    assert not is_low_information_fact(boilerplate + "被告人盗窃手机。" * 20)


def test_merge_duplicate_annotations_unions_labels_and_articles() -> None:
    first = CaseRecord(
        case_id="a",
        fact="被告人实施了盗窃行为。",
        accusations=["盗窃"],
        relevant_articles=[264],
    )
    second = CaseRecord(
        case_id="b",
        fact=first.fact,
        accusations=["掩饰、隐瞒犯罪所得、犯罪所得收益"],
        relevant_articles=[312],
    )

    assert merge_case_labels(first, second)
    assert first.accusations == ["掩饰、隐瞒犯罪所得、犯罪所得收益", "盗窃"]
    assert first.relevant_articles == [264, 312]
