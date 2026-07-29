import copy
import unittest

import torch
import torch.nn as nn

from agent_data import build_agent_args
from agent_trainer import _build_sgd_optimizer, _ensure_dsdm_path, mask_inactive_class_logits
from config_adapter import load_config, normalize_receiver_checkpoint_retention

_ensure_dsdm_path()

from models.alexnet_cifar import AlexNetCIFAR
from models.convnet import ConvNet
from models.cosine_classifier import (
    CosineClassifier,
    get_classifier_weight,
    get_cosine_classifier,
    get_normalized_classifier_weight,
    set_classifier_weight_rows,
)
from models.resnet import ResNet


class CosineExpertProtocolTest(unittest.TestCase):
    def test_receiver_checkpoint_retention_is_explicit_and_validated(self):
        self.assertEqual(normalize_receiver_checkpoint_retention("all"), "all")
        self.assertEqual(normalize_receiver_checkpoint_retention(" FINAL_ONLY "), "final_only")
        with self.assertRaisesRegex(ValueError, "checkpoint_retention"):
            normalize_receiver_checkpoint_retention("minimal")

    def test_cosine_geometry_and_positive_scale(self):
        torch.manual_seed(3)
        head = CosineClassifier(4, 3, scale_init=10.0)
        features = torch.randn(5, 4)
        logits = head(features)

        self.assertIsNone(head.bias)
        self.assertAlmostEqual(float(head.scale.detach()), 10.0, places=5)
        self.assertGreater(float(head.scale.detach()), 0.0)
        self.assertTrue(torch.allclose(logits, head(features * 7.0), atol=1e-6, rtol=1e-5))
        self.assertLessEqual(float(logits.abs().max()), float(head.scale.detach()) + 1e-5)

    def test_prototype_rows_are_normalized_without_overwriting_local_rows(self):
        torch.manual_seed(5)
        model = nn.Sequential(CosineClassifier(4, 6))
        head = get_cosine_classifier(model)
        before = head.weight.detach().clone()
        self.assertIs(get_classifier_weight(model), head.weight)

        set_classifier_weight_rows(
            model,
            [1, 4],
            torch.tensor([[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, -2.0, 0.0]]),
        )
        normalized = get_normalized_classifier_weight(model)
        self.assertTrue(torch.allclose(head.weight[[1, 4]].norm(dim=1), torch.ones(2)))
        self.assertTrue(torch.allclose(normalized[[1, 4]], head.weight[[1, 4]]))
        self.assertTrue(torch.equal(head.weight[[0, 2, 3, 5]], before[[0, 2, 3, 5]]))

    def test_cosine_scale_has_zero_weight_decay(self):
        model = nn.Sequential(nn.Linear(4, 4), CosineClassifier(4, 3))
        optimizer = _build_sgd_optimizer(model, lr=0.1, momentum=0.9, weight_decay=5e-4)
        head = get_cosine_classifier(model)

        scale_groups = [
            group
            for group in optimizer.param_groups
            if any(parameter is head.scale_parameter for parameter in group["params"])
        ]
        weight_groups = [
            group
            for group in optimizer.param_groups
            if any(parameter is head.weight for parameter in group["params"])
        ]
        self.assertEqual(len(scale_groups), 1)
        self.assertEqual(scale_groups[0]["weight_decay"], 0.0)
        self.assertEqual(len(weight_groups), 1)
        self.assertEqual(weight_groups[0]["weight_decay"], 5e-4)

    def test_masked_ce_keeps_global_labels_and_zeroes_inactive_gradients(self):
        logits = torch.zeros(2, 100, requires_grad=True)
        with torch.no_grad():
            logits[:, 99] = 1000.0
            logits[0, 17] = 2.0
            logits[1, 73] = 2.0
        labels = torch.tensor([17, 73], dtype=torch.long)
        masked = mask_inactive_class_logits(logits, labels, [17, 73])
        loss = nn.CrossEntropyLoss()(masked, labels)
        loss.backward()

        self.assertEqual(masked.argmax(dim=1).tolist(), labels.tolist())
        self.assertEqual(tuple(masked.shape), (2, 100))
        self.assertEqual(float(logits.grad[:, 99].abs().sum()), 0.0)
        self.assertGreater(float(logits.grad[:, [17, 73]].abs().sum()), 0.0)

    def test_five_backbones_keep_output_and_feature_contracts(self):
        models_and_features = [
            (
                ConvNet(100, net_depth=3, net_width=128, classifier_type="cosine"),
                2,
                (2, 128, 4, 4),
            ),
            (
                ConvNet(100, net_depth=4, net_width=192, classifier_type="cosine"),
                3,
                (2, 192, 2, 2),
            ),
            (AlexNetCIFAR(100, classifier_type="cosine"), 7, (2, 512)),
            (
                ResNet(
                    "cifar100",
                    10,
                    100,
                    size=32,
                    cifar_base_width=64,
                    classifier_type="cosine",
                ),
                5,
                (2, 512),
            ),
            (
                ResNet(
                    "cifar100",
                    18,
                    100,
                    size=32,
                    cifar_base_width=64,
                    classifier_type="cosine",
                ),
                5,
                (2, 512),
            ),
        ]
        images = torch.randn(2, 3, 32, 32)
        for model, feature_index, expected_feature_shape in models_and_features:
            model.eval()
            with torch.no_grad():
                self.assertEqual(tuple(model(images).shape), (2, 100))
                result = model.get_feature(images, feature_index, feature_index)
            feature_list = result[0] if isinstance(result, tuple) else result
            self.assertEqual(tuple(feature_list[0].shape), expected_feature_shape)
            self.assertEqual(get_cosine_classifier(model).out_features, 100)

    def test_classifier_protocol_is_opt_in_and_guide_stays_linear(self):
        path = "configs/teacher_quality/train_conv3_seed0.yaml"
        historical = load_config(path)
        historical_args = build_agent_args(historical, path, 0)
        self.assertEqual(historical_args.classifier_type, "linear")
        self.assertFalse(historical_args.expert_mask_nonlocal_classes)

        protocol = copy.deepcopy(historical)
        model_cfg = protocol["model_pool"]["models"]["convnet3w1"]
        model_cfg["classifier"] = {
            "type": "cosine",
            "scale_init": 10.0,
            "positive": "softplus",
            "bias": False,
            "scale_weight_decay": 0.0,
        }
        model_cfg["expert_training"]["masked_local_ce"] = True
        args = build_agent_args(protocol, "diagnostic.yaml", 0)
        self.assertEqual(args.classifier_type, "cosine")
        self.assertEqual(args.cosine_scale_init, 10.0)
        self.assertTrue(args.expert_mask_nonlocal_classes)
        self.assertEqual(args.guide_classifier_type, "linear")

        linear_protocol = copy.deepcopy(historical)
        linear_model_cfg = linear_protocol["model_pool"]["models"]["convnet3w1"]
        linear_model_cfg["classifier"] = {"type": "linear"}
        linear_model_cfg["expert_training"]["masked_local_ce"] = True
        linear_args = build_agent_args(linear_protocol, "linear_diagnostic.yaml", 0)
        self.assertEqual(linear_args.classifier_type, "linear")
        self.assertTrue(linear_args.expert_mask_nonlocal_classes)
        self.assertEqual(linear_args.nclass, 100)


if __name__ == "__main__":
    unittest.main()
