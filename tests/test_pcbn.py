import unittest

import torch

from DSDM.models.resnet import ResNet
from DSDM.pcbn import PCBNRegularizer


class PCBNRegularizerTest(unittest.TestCase):
    @staticmethod
    def _args(enabled=True, weight=1.0):
        return type(
            "Args",
            (),
            {
                "pcbn_enabled": enabled,
                "pcbn_weight": weight,
                "pcbn_layers": "all",
                "pcbn_normalize_layers": True,
            },
        )()

    def test_resnet18_hooks_all_batchnorm_layers(self):
        model = ResNet("tinyimagenet", 18, 200, norm_type="batch", size=64).eval()
        regularizer = PCBNRegularizer(self._args())
        hooked = regularizer.attach(model)
        expected = sum(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules())
        self.assertEqual(hooked, expected)
        self.assertGreater(hooked, 0)
        regularizer.close()
        self.assertEqual(len(regularizer.handles), 0)

    def test_identical_inputs_have_zero_loss_and_perturbation_backpropagates(self):
        torch.manual_seed(0)
        model = ResNet("tinyimagenet", 18, 200, norm_type="batch", size=64).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        regularizer = PCBNRegularizer(self._args(weight=2.0))
        regularizer.attach(model)
        real = torch.randn(2, 3, 64, 64)
        identical = real.clone().requires_grad_(True)
        self.assertLess(float(regularizer.loss(model, real, identical).item()), 1e-10)

        synthetic = (real + 0.25 * torch.randn_like(real)).requires_grad_(True)
        loss = regularizer.loss(model, real, synthetic)
        self.assertGreater(float(loss.item()), 0.0)
        loss.backward()
        self.assertIsNotNone(synthetic.grad)
        self.assertGreater(float(synthetic.grad.abs().sum().item()), 0.0)
        regularizer.close()

    def test_non_positive_weight_disables_regularizer(self):
        regularizer = PCBNRegularizer(self._args(enabled=True, weight=0.0))
        self.assertFalse(regularizer.enabled)


if __name__ == "__main__":
    unittest.main()
