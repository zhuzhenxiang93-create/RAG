from legalmind.retrieval.chunker import ChineseCaseChunker
from legalmind.schemas import CaseRecord


def test_chunker_preserves_offsets_and_overlap():
    case = CaseRecord(
        case_id="x",
        fact="第一句话。第二句话很长。第三句话也很长。第四句话结束。",
        accusations=["盗窃"],
    )
    chunks = ChineseCaseChunker(chunk_size=16, overlap=4).split(case)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.text == case.fact[chunk.start_char : chunk.end_char]
        assert chunk.end_char - chunk.start_char <= 16
    assert chunks[1].start_char <= chunks[0].end_char


def test_short_case_is_single_chunk():
    case = CaseRecord(case_id="x", fact="短案情。")
    chunks = ChineseCaseChunker().split(case)
    assert len(chunks) == 1
    assert chunks[0].text == case.fact


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)


def test_tokenizer_budget_is_respected():
    case = CaseRecord(case_id="x", fact="甲乙丙丁戊己庚辛。壬癸子丑寅卯辰巳。")
    chunks = ChineseCaseChunker(8, 2, tokenizer=CharacterTokenizer()).split(case)
    assert all(len(chunk.text) <= 8 for chunk in chunks)
    assert all(chunk.text == case.fact[chunk.start_char : chunk.end_char] for chunk in chunks)
