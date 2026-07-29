import copy
import unittest

from config_adapter import load_config
from scripts.validate_iclr2027_dkp_communication import (
    CE_CONFIG,
    EXPERT_CONFIG,
    FULL_CONFIG,
    CommunicationPreflightError,
    validate_config_contract,
)


class ICLR2027DKPCommunicationPreflightTest(unittest.TestCase):
    def setUp(self):
        self.expert = load_config(EXPERT_CONFIG)
        self.ce_only = load_config(CE_CONFIG)
        self.full = load_config(FULL_CONFIG)

    def test_current_three_variant_contract(self):
        class_split, model_split = validate_config_contract(self.expert, self.ce_only, self.full)
        self.assertEqual(sorted(class_split), list(range(5)))
        self.assertEqual(sorted(model_split), list(range(5)))

    def test_rejects_variant_or_partition_drift(self):
        wrong_variant = copy.deepcopy(self.full)
        wrong_variant["social_learning"]["receiver"]["dkp_variant"] = "ce_only"
        with self.assertRaisesRegex(CommunicationPreflightError, "full variant"):
            validate_config_contract(self.expert, self.ce_only, wrong_variant)

        wrong_partition = copy.deepcopy(self.ce_only)
        wrong_partition["agents"]["class_split"]["agent_0"][0] = 99
        with self.assertRaisesRegex(CommunicationPreflightError, "class split"):
            validate_config_contract(self.expert, wrong_partition, self.full)


if __name__ == "__main__":
    unittest.main()
