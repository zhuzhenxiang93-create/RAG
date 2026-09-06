from legalmind.cli import parser


def test_cli_exposes_unified_analyze_only() -> None:
    choices = parser()._subparsers._group_actions[0].choices
    assert "analyze" in choices
    assert "predict-sentence" not in choices


def test_analyze_limits_case_results_to_three() -> None:
    args = parser().parse_args(["analyze", "--fact", "人工构造测试事实", "--top-k", "3"])
    assert args.top_k == 3
