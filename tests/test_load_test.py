import unittest

from scripts.load_test import percentile


class LoadTestMetricTest(unittest.TestCase):
    def test_nearest_rank_percentile(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(percentile(values, 0.50), 2.0)
        self.assertEqual(percentile(values, 0.95), 4.0)
        self.assertEqual(percentile([], 0.95), 0.0)


if __name__ == "__main__":
    unittest.main()
