import unittest

import numpy as np

from conditioning import compare, make_problem
from coursework.bracketed_roots import bisection, regula_falsi


class ConditioningTests(unittest.TestCase):
    def test_problem_has_requested_condition_number(self):
        design, target, coefficients = make_problem(1e6)
        self.assertAlmostEqual(np.linalg.cond(design) / 1e6, 1.0, places=8)
        np.testing.assert_array_equal(design @ coefficients, target)

    def test_solvers_agree_on_well_conditioned_problem(self):
        _, rank, results = compare(100)
        self.assertEqual(rank, 8)
        self.assertEqual(set(results), {"normal", "qr", "lstsq"})
        self.assertTrue(all(error < 1e-10 for error, _ in results.values()))

    def test_qr_preserves_coefficients_better_than_normal_equations(self):
        _, rank, results = compare(1e10)
        self.assertEqual(rank, 8)
        self.assertLess(results["qr"][0], 1e-5)
        self.assertLess(results["lstsq"][0], 1e-5)
        if "normal" in results:
            self.assertLess(results["qr"][0], results["normal"][0] / 1_000)

    def test_bracketed_solvers_accept_a_root_at_either_endpoint(self):
        for solve in (bisection, regula_falsi):
            self.assertEqual(solve(lambda x: x - 2, 2, 3), (0, 2))
            self.assertEqual(solve(lambda x: x - 2, 1, 2), (0, 2))


if __name__ == "__main__":
    unittest.main()
