import unittest

import torch

from DSDM.numerical_guard import clip_and_validate_gradients, ensure_finite_tensor


class DSDMNumericalGuardTest(unittest.TestCase):
    def test_finite_tensor_passes(self):
        ensure_finite_tensor(torch.tensor([0.0, 1.0]), "value", iteration=1)

    def test_nonfinite_tensor_reports_context(self):
        with self.assertRaisesRegex(FloatingPointError, "iteration=4.*class_id=7.*guide_idx=2"):
            ensure_finite_tensor(
                torch.tensor([float("nan")]),
                "loss",
                iteration=4,
                class_id=7,
                guide_idx=2,
            )

    def test_gradient_clipping_limits_norm(self):
        parameter = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
        parameter.grad = torch.tensor([30.0, 40.0])
        total_norm, clipped = clip_and_validate_gradients(
            [parameter],
            10.0,
            iteration=1,
            class_id=2,
            guide_idx=3,
        )
        self.assertAlmostEqual(total_norm, 50.0, places=4)
        self.assertTrue(clipped)
        self.assertLessEqual(float(parameter.grad.norm().item()), 10.0001)

    def test_nonfinite_gradient_fails_before_step(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        parameter.grad = torch.tensor([float("inf")])
        with self.assertRaisesRegex(FloatingPointError, "synthetic gradient"):
            clip_and_validate_gradients(
                [parameter],
                100.0,
                iteration=5,
                class_id=6,
                guide_idx=7,
            )


if __name__ == "__main__":
    unittest.main()
