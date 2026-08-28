from legalmind.data.token_stats import summarize_token_lengths


def test_summarize_token_lengths_reports_tail_fraction() -> None:
    summary = summarize_token_lengths([10, 100, 1024, 2049], thresholds=(1024, 2048))

    assert summary["rows"] == 4
    assert summary["max"] == 2049
    assert summary["thresholds"]["1024"]["rows_exceeding"] == 1
    assert summary["thresholds"]["2048"]["fraction_exceeding"] == 0.25


def test_summarize_token_lengths_handles_empty_input() -> None:
    assert summarize_token_lengths([]) == {"rows": 0}
