import unittest

import torch

from scripts.run_tiny_backbone_validation import build_model


class TinyImageNetBackboneTest(unittest.TestCase):
    def test_backbones_accept_64px_and_output_200_classes(self):
        expected_parameters = {
            "convnet4": 1_617_416,
            "resnet18": 11_271_432,
            "alexnet": 7_076_616,
            "mobilenetv2": 2_480_072,
        }
        inputs = torch.randn(2, 3, 64, 64)
        for name, parameter_count in expected_parameters.items():
            with self.subTest(model=name):
                model = build_model(name).eval()
                with torch.no_grad():
                    outputs = model(inputs)
                self.assertEqual(tuple(outputs.shape), (2, 200))
                self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), parameter_count)

    def test_alexnet_remains_compatible_with_32px_inputs(self):
        model = build_model("alexnet").eval()
        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(outputs.shape), (2, 200))


if __name__ == "__main__":
    unittest.main()
