import copy
import math
import unittest

from config_adapter import build_dsdm_args_from_config
from scripts.prepare_iclr2027_dkp_domain_mix import (
    EXPECTED_LOSSES,
    EXPECTED_STEP_MILESTONES,
    EXPECTED_STEPS,
    MIX_CONDITIONS,
    build_configs,
    run_name,
    validate_config,
)


class ICLR2027DKPDomainMixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configs = build_configs()

    def test_materializes_exact_predeclared_fraction_curve(self):
        self.assertEqual(len(self.configs), 4)
        observed = {}
        for condition, fraction in MIX_CONDITIONS.items():
            filename = f"{run_name(condition)}.yaml"
            self.assertIn(filename, self.configs)
            config = self.configs[filename]
            receiver = config["social_learning"]["receiver"]
            self.assertEqual(config["project"]["paper_eligible"], False)
            self.assertEqual(config["project"]["protocol_status"], "planned_diagnostic")
            self.assertEqual(receiver["local_ce_source"], "real_packet_mix")
            self.assertTrue(math.isclose(receiver["local_ce_real_fraction"], fraction))
            self.assertEqual(receiver["optimizer_steps"], EXPECTED_STEPS)
            self.assertEqual(receiver["scheduler_step_milestones"], EXPECTED_STEP_MILESTONES)
            self.assertEqual(receiver["loss_switches"], EXPECTED_LOSSES)
            args = build_dsdm_args_from_config(config)
            self.assertEqual(args.receiver_local_ce_source, "real_packet_mix")
            self.assertTrue(math.isclose(args.receiver_local_ce_real_fraction, fraction))
            observed[condition] = args.receiver_local_ce_real_fraction
        self.assertEqual(observed, MIX_CONDITIONS)

    def test_validator_rejects_fraction_or_protocol_drift(self):
        condition = "r10"
        filename = f"{run_name(condition)}.yaml"
        for mutation in ("fraction", "steps", "logits", "paper"):
            with self.subTest(mutation=mutation):
                config = copy.deepcopy(self.configs[filename])
                if mutation == "fraction":
                    config["social_learning"]["receiver"]["local_ce_real_fraction"] = 0.2
                elif mutation == "steps":
                    config["social_learning"]["receiver"]["optimizer_steps"] = 3779
                elif mutation == "logits":
                    config["communication"]["use_sender_logits"] = True
                else:
                    config["project"]["paper_eligible"] = True
                with self.assertRaises(Exception):
                    validate_config(config, condition)


if __name__ == "__main__":
    unittest.main()
