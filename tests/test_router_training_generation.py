import unittest

from scripts.generate_router_training_questions import QUESTION_TYPES, build_rows


class RouterTrainingGenerationTest(unittest.TestCase):
    def test_generated_router_data_is_balanced_unique_and_independent(self):
        rows = build_rows(per_type=8, seed=7)
        self.assertEqual(len(rows), 8 * len(QUESTION_TYPES))
        self.assertEqual(len({row["question"] for row in rows}), len(rows))
        self.assertEqual(
            {
                question_type: sum(
                    row["question_type"] == question_type for row in rows
                )
                for question_type in QUESTION_TYPES
            },
            {question_type: 8 for question_type in QUESTION_TYPES},
        )
        self.assertTrue(
            all(row["generation"]["company"] == "Northstar Labs" for row in rows)
        )
        self.assertTrue(
            all(
                row["generation"]["method"] == "deterministic_templates_v1"
                for row in rows
            )
        )
