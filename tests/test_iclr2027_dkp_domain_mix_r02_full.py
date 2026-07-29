import copy
import unittest

from scripts.prepare_iclr2027_dkp_domain_mix_r02_full import (
    EXPECTED_LOSSES,
    RUN_NAME,
    build_config,
    validate_config,
)


class ICLR2027DKPDomainMixR02FullTests(unittest.TestCase):
    def test_exact_config_enables_only_the_paired_kd_condition(self):
        config = build_config()
        args = validate_config(config)
        self.assertEqual(config["project"]["run_name"], RUN_NAME)
        self.assertIs(config["project"]["paper_eligible"], False)
        self.assertEqual(args.dkp_variant, "full")
        self.assertEqual(args.dkp_loss_switches, EXPECTED_LOSSES)
        self.assertTrue(args.use_sender_logits)
        self.assertEqual(args.receiver_local_ce_real_fraction, 0.02)
        self.assertEqual(config["loss_ablation"]["variant_id"], "fr1_kd1_sc1")
        self.assertEqual(config["loss_ablation"]["switches"], EXPECTED_LOSSES)
        self.assertEqual(config["domain_mix_diagnostic"]["paired_difference"], "sender_class_kd_enabled")

    def test_exact_config_rejects_kd_or_adaptive_provenance_drift(self):
        for mutation in ("kd", "logits", "adaptive", "ablation_variant", "ablation_switch"):
            with self.subTest(mutation=mutation):
                config = copy.deepcopy(build_config())
                if mutation == "kd":
                    config["logits"]["lambda_kd"] = 0.5
                elif mutation == "logits":
                    config["communication"]["use_sender_logits"] = False
                elif mutation == "ablation_variant":
                    config["loss_ablation"]["variant_id"] = "fr1_kd0_sc1"
                elif mutation == "ablation_switch":
                    config["loss_ablation"]["switches"]["kd"] = False
                else:
                    config["domain_mix_diagnostic"]["formal_hyperparameter_selection"] = True
                with self.assertRaises(Exception):
                    validate_config(config)


if __name__ == "__main__":
    unittest.main()
