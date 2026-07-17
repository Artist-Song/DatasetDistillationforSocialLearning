import unittest

import torch

from DSDM.models.resnet import ResNet
from social_trainer import _augment_cifar_batch, _freeze_batchnorm_stats


class ResNetVariantTest(unittest.TestCase):
    def test_compact_resnet18_remains_backward_compatible(self):
        model = ResNet("cifar100", 18, 100, norm_type="batch", size=32)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 2_820_740)
        self.assertEqual(model.fc.in_features, 256)

    def test_standard_cifar_resnet18_uses_full_width(self):
        model = ResNet(
            "cifar100",
            18,
            100,
            norm_type="batch",
            size=32,
            cifar_base_width=64,
        )
        self.assertEqual(sum(p.numel() for p in model.parameters()), 11_220_132)
        self.assertEqual(model.fc.in_features, 512)
        self.assertEqual(tuple(model(torch.randn(2, 3, 32, 32)).shape), (2, 100))

    def test_receiver_augmentation_and_bn_freeze_are_opt_in(self):
        args = type("Args", (), {"receiver_augment": True, "dataset": "cifar100"})()
        images = torch.zeros(4, 3, 32, 32)
        self.assertEqual(tuple(_augment_cifar_batch(images, args).shape), tuple(images.shape))

        model = ResNet("cifar100", 10, 100, norm_type="batch", size=32)
        model.train()
        _freeze_batchnorm_stats(model)
        batch_norms = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        self.assertTrue(batch_norms)
        self.assertTrue(all(not layer.training for layer in batch_norms))
        self.assertTrue(all(layer.weight.requires_grad for layer in batch_norms))


if __name__ == "__main__":
    unittest.main()
