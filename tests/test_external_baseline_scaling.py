import unittest
from pathlib import Path

import torch

from agent_data import get_agent_class_split, get_agent_model_split
from baseline_adapters.prepare_fedre_reproduction import project_dataset_name
from baseline_adapters.run_desa_cil import validate_agent_protocol
from baseline_adapters.run_fedre_reproduction import (
    fedre_communication_accounting,
    fedre_model_expressions,
    fedre_model_names,
)
from baseline_adapters.run_masc_complete import (
    class_membership_mask,
    validate_class_split,
)
from config_adapter import load_config


ROOT = Path(__file__).resolve().parents[1]


class ExternalBaselineScalingTests(unittest.TestCase):
    def _config(self, agent_count: int, seed: int = 1):
        classes_per_agent = 100 // agent_count
        path = ROOT / (
            f"configs/iclr2027/scaling/cifar100_{agent_count}agent{classes_per_agent}cls_"
            f"dkp_cosine_experts_seed{seed}_v1.yaml"
        )
        return path, load_config(path)

    def test_all_current_splits_pass_desa_and_masc_contracts(self):
        names = set()
        for agent_count in (5, 10, 20):
            for seed in (1, 2, 3):
                with self.subTest(agent_count=agent_count, seed=seed):
                    _path, config = self._config(agent_count, seed)
                    classes = get_agent_class_split(config)
                    models = get_agent_model_split(config)
                    validate_class_split(classes)
                    validate_agent_protocol(classes, models)
                    name = project_dataset_name(config)
                    self.assertNotIn(name, names)
                    names.add(name)

    def test_masc_membership_uses_explicit_random_classes(self):
        labels = torch.tensor([0, 2, 7, 30, 80, 99])
        mask = class_membership_mask(labels, [80, 2, 30])
        self.assertEqual(mask.tolist(), [False, True, False, True, True, False])

    def test_fedre_model_assignment_rule(self):
        first_five = fedre_model_names(5)
        official_ten = fedre_model_names(10)
        repeated_twenty = fedre_model_names(20)
        self.assertEqual(first_five, official_ten[:5])
        self.assertEqual(repeated_twenty[:10], official_ten)
        self.assertEqual(repeated_twenty[10:], official_ten)
        expressions = fedre_model_expressions(20)
        self.assertEqual(expressions[:10], expressions[10:])

    def test_fedre_communication_accounts_for_head_and_representations(self):
        report = fedre_communication_accounting(
            num_clients=5,
            rounds=100,
            feature_dim=512,
            num_classes=100,
            class_counts=[20] * 5,
        )
        self.assertEqual(report["official_loop_updates"], 101)
        self.assertEqual(report["shared_head_bytes_per_broadcast"], 205_200)
        self.assertEqual(report["shared_head_broadcast_bytes_all_clients"], 103_626_000)
        self.assertEqual(report["entangled_representation_bytes_all_clients"], 1_034_240)
        self.assertEqual(report["raw_image_communication"], 0)
        self.assertEqual(report["total_logical_bytes"], 104_781_440)

    def test_invalid_fedre_class_count_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "cover"):
            fedre_communication_accounting(5, 100, 512, 100, [20] * 4 + [19])


if __name__ == "__main__":
    unittest.main()
