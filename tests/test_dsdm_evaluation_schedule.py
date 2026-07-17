import unittest

from DSDM.evaluation_schedule import (
    DEFAULT_EVALUATION_ITERATIONS,
    resolve_evaluation_iterations,
)


class EvaluationScheduleTest(unittest.TestCase):
    def test_default_sparse_schedule(self):
        self.assertEqual(
            resolve_evaluation_iterations(10000, DEFAULT_EVALUATION_ITERATIONS),
            list(DEFAULT_EVALUATION_ITERATIONS),
        )

    def test_final_iteration_is_always_evaluated(self):
        self.assertEqual(
            resolve_evaluation_iterations(6000, DEFAULT_EVALUATION_ITERATIONS),
            [100, 500, 1000, 2000, 3000, 5000, 6000],
        )

    def test_custom_string_is_sorted_and_deduplicated(self):
        self.assertEqual(
            resolve_evaluation_iterations(1000, "500,100,500"),
            [100, 500, 1000],
        )

    def test_empty_schedule_uses_legacy_interval(self):
        self.assertEqual(
            resolve_evaluation_iterations(550, [], evaluate_iter=200),
            [200, 400, 550],
        )

    def test_invalid_iteration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            resolve_evaluation_iterations(1000, [0, 500])


if __name__ == "__main__":
    unittest.main()
