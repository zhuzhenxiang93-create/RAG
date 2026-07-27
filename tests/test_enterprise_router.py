import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from app.plugins.enterprise_router.data import (
    assert_no_benchmark_leakage,
    format_instruction,
    normalized_training_row,
    question_fingerprint,
    route_from_question,
)
from app.plugins.enterprise_router.metrics import routing_metrics
from app.plugins.enterprise_router.parsing import parse_route
from app.plugins.enterprise_router.schema import SOURCE_TYPES, RouteDecision
from app.plugins.enterprise_router.training import RouterTrainingConfig


class EnterpriseRouterTest(unittest.TestCase):
    def test_project_question_becomes_deep_multi_source_route(self) -> None:
        route = route_from_question(
            {
                "question_type": "Project Related",
                "source_types": ["linear", "slack", "gmail"],
                "question": "Why did the launch move?",
                "expected_doc_ids": ["d1", "d2", "d3"],
            }
        )
        self.assertEqual(route.question_type, "project_related")
        self.assertEqual(route.sources, ["linear", "slack", "gmail"])
        self.assertTrue(route.multi_document)
        self.assertTrue(route.completeness_required)
        self.assertEqual(route.retrieval_depth, "deep")

    def test_info_not_found_uses_broad_unanswerable_route(self) -> None:
        route = route_from_question(
            {
                "question_type": "Info Not Found",
                "source_types": [],
                "question": "Is there an office in Wellington?",
                "expected_doc_ids": [],
            }
        )
        self.assertEqual(route.answerability, "unanswerable")
        self.assertEqual(route.sources, list(SOURCE_TYPES))

    def test_parser_accepts_fenced_or_prefixed_json_but_validates_schema(self) -> None:
        text = (
            "```json\n"
            '{"question_type":"basic","sources":["github"],'
            '"multi_document":false,"conflict_check":false,'
            '"completeness_required":false,"retrieval_depth":"shallow",'
            '"answerability":"answerable"}\n```'
        )
        route, canonical = parse_route(text)
        self.assertEqual(route.sources, ["github"])
        self.assertEqual(json.loads(canonical)["question_type"], "basic")
        with self.assertRaises(ValueError):
            parse_route('{"question_type":"made_up"}')

    def test_leakage_guard_catches_id_and_normalized_text(self) -> None:
        benchmark = [
            {
                "question_id": "qst_0001",
                "question": "What is the release date?",
            }
        ]
        with self.assertRaisesRegex(ValueError, "contamination"):
            assert_no_benchmark_leakage(
                [
                    {
                        "question_id": "custom",
                        "question": "What is the release date!!!",
                    }
                ],
                benchmark,
            )
        with self.assertRaisesRegex(ValueError, "qst_0001"):
            assert_no_benchmark_leakage(
                [{"question_id": "qst_0001", "question": "A different question"}],
                benchmark,
            )

    def test_normalized_row_is_deterministic_and_prompt_does_not_include_answer(self) -> None:
        row = {
            "question_id": "train_1",
            "question_type": "basic",
            "source_types": ["jira"],
            "question": "Which ticket is open?",
            "expected_doc_ids": ["doc_1"],
            "gold_answer": "SECRET",
        }
        normalized = normalized_training_row(row)
        self.assertEqual(
            normalized["question_fingerprint"],
            question_fingerprint("Which ticket is open?"),
        )
        self.assertNotIn("SECRET", normalized["prompt"])
        self.assertEqual(normalized["prompt"], format_instruction(row["question"]))
        self.assertIn('"sources":["jira"]', normalized["target"])

    def test_metrics_penalize_invalid_outputs_via_attempted_count(self) -> None:
        gold = RouteDecision(
            question_type="basic",
            sources=["github"],
            multi_document=False,
            conflict_check=False,
            completeness_required=False,
            retrieval_depth="shallow",
            answerability="answerable",
        )
        result = routing_metrics([gold, gold], [gold, None])
        self.assertEqual(result["json_valid_rate"], 0.5)
        self.assertEqual(result["exact_route_accuracy"], 0.5)
        self.assertLess(result["source_micro_f1"], 1.0)

    def test_training_config_targets_attention_projections(self) -> None:
        config = RouterTrainingConfig(
            base_model="Qwen/Qwen2.5-1.5B-Instruct",
            output_dir="artifacts/router",
        )
        self.assertEqual(
            config.target_modules,
            ["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        with self.assertRaises(ValueError):
            RouterTrainingConfig(
                base_model="test",
                output_dir="test",
                max_length=32,
            )


if __name__ == "__main__":
    unittest.main()
