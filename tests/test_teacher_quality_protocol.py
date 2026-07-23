import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from agent_data import build_agent_args, get_num_classes
from agent_trainer import _guide_paths, _stratified_split_indices, _train_epoch
from config_adapter import load_config
from DSDM.models.resnet import ResNet
from scripts.prepare_teacher_quality_protocol import MODELS


class _TargetsOnlyDataset:
    def __init__(self):
        self.targets = [class_id for class_id in range(4) for _ in range(10)]


class TeacherQualityProtocolTest(unittest.TestCase):
    def test_local_validation_split_is_stratified_and_deterministic(self):
        dataset = _TargetsOnlyDataset()
        train_a, validation_a = _stratified_split_indices(dataset, 0.2, seed=7)
        train_b, validation_b = _stratified_split_indices(dataset, 0.2, seed=7)
        self.assertEqual((train_a, validation_a), (train_b, validation_b))
        self.assertEqual(len(train_a), 32)
        self.assertEqual(len(validation_a), 8)
        self.assertEqual(
            {class_id: sum(dataset.targets[idx] == class_id for idx in validation_a) for class_id in range(4)},
            {class_id: 2 for class_id in range(4)},
        )

    def test_generated_training_configs_separate_guides_and_logit_teachers(self):
        for name, spec in MODELS.items():
            path = f"configs/teacher_quality/train_{name}_seed0.yaml"
            config = load_config(path)
            args = build_agent_args(config, path, spec["agent"])
            self.assertTrue(args.separate_expert)
            self.assertEqual(args.guide_snapshot_epochs, spec["guide_epochs"])
            self.assertEqual(args.guide_model_number, 10)
            self.assertEqual(args.expert_epochs, spec["expert_epochs"])
            self.assertEqual(args.guide_augment, spec["guide_augment"])
            self.assertEqual(args.expert_augment, spec["expert_augment"])
            self.assertEqual(args.expert_use_dsdm_train, spec["expert_use_dsdm_train"])
            self.assertEqual(args.expert_validation_fraction, 0.1)

    def test_standard_resnets_have_expected_capacity_and_feature_index(self):
        expected = {
            "resnet10_standard": 4_949_412,
            "resnet18_standard": 11_220_132,
        }
        for name, parameter_count in expected.items():
            spec = MODELS[name]
            path = f"configs/teacher_quality/train_{name}_seed0.yaml"
            config = load_config(path)
            args = build_agent_args(config, path, spec["agent"])
            model = ResNet(
                args.dataset,
                args.depth,
                get_num_classes(args),
                norm_type=args.norm_type,
                size=args.size,
                cifar_base_width=64,
            )
            model.eval()
            self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), parameter_count)
            self.assertEqual(args.f_idx, "5")
            self.assertEqual((args.idx_from, args.idx_to), (5, -1))
            self.assertEqual(tuple(model(torch.randn(2, 3, 32, 32)).shape), (2, 100))

    def test_candidate_guide_paths_are_epoch_scoped(self):
        spec = MODELS["conv3"]
        path = "configs/teacher_quality/packet_conv3_guidee0100_seed0_ipc10.yaml"
        config = load_config(path)
        args = build_agent_args(config, path, spec["agent"])
        paths = _guide_paths(args, Path("source") / "checkpoints")
        self.assertEqual(len(paths), 10)
        self.assertTrue(all("guide_pools/e0100" in str(item) for item in paths))

    def test_validated_convnet_expert_recipe_executes_cutmix(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 4))
        loader = DataLoader(
            TensorDataset(torch.randn(8, 3, 4, 4), torch.arange(8) % 4),
            batch_size=8,
        )
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        loss = _train_epoch(
            model,
            loader,
            nn.CrossEntropyLoss(),
            optimizer,
            torch.device("cpu"),
            args=SimpleNamespace(mixup="cut", mix_p=1.0, beta=1.0),
            use_dsdm_train=True,
        )
        self.assertTrue(torch.isfinite(torch.tensor(loss)))


if __name__ == "__main__":
    unittest.main()
