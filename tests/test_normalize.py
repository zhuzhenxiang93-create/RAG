from legalmind.data.normalize import clean_fact, normalize_record


def test_normalize_official_cail_shape():
    case = normalize_record(
        {
            "fact": "  被告人\n盗窃手机。 ",
            "meta": {
                "accusation": ["盗窃"],
                "relevant_articles": [264],
                "term_of_imprisonment": {"imprisonment": 8},
            },
        },
        source_split="train",
        label_to_id={"盗窃": 1},
    )
    assert case.fact == "被告人 盗窃手机。"
    assert case.accusation_ids == [1]
    assert case.relevant_articles == [264]
    assert case.case_id.startswith("train-")


def test_clean_fact_full_width_spaces():
    assert clean_fact("甲\u3000乙\n丙") == "甲 乙 丙"


def test_normalize_legacy_input_output_shape():
    case = normalize_record(
        {
            "input": "被告人盗窃手机。",
            "output": {
                "是否死刑": False,
                "有期徒刑": 8,
                "是否无期": False,
                "罪名": ["盗窃"],
                "罚金": 1000,
            },
        }
    )
    assert case.accusations == ["盗窃"]
    assert case.penalty["imprisonment_months"] == 8
    assert case.penalty["fine"] == 1000


def test_normalize_processed_shape_preserves_fine():
    case = normalize_record(
        {
            "case_id": "processed-1",
            "fact": "被告人盗窃手机。",
            "labels": {
                "accusations": ["盗窃"],
                "imprisonment_months": 8,
                "fine": 1000,
                "life_imprisonment": False,
                "death_penalty": False,
            },
            "accusation_ids": [99],
        },
        source_split="train",
    )
    assert case.penalty["fine"] == 1000
