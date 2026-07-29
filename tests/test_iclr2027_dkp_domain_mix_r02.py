import copy
import unittest

from scripts.prepare_iclr2027_dkp_domain_mix_r02 import (
    FRACTION,
    RUN_NAME,
    build_config,
    validate_config,
)


class ICLR2027DKPDomainMixR02Tests(unittest.TestCase):
    def test_exact_postcurve_config_is_adaptive_and_nonformal(self):
        config = build_config()
        args = validate_config(config)
        self.assertEqual(config["project"]["run_name"], RUN_NAME)
        self.assertIs(config["project"]["paper_eligible"], False)
        self.assertEqual(args.receiver_local_ce_real_fraction, FRACTION)
        diagnostic = config["domain_mix_diagnostic"]
        self.assertIs(diagnostic["adaptive_after_completed_fraction_curve"], True)
        self.assertIs(diagnostic["formal_hyperparameter_selection"], False)

    def test_exact_validator_rejects_any_fraction_or_provenance_drift(self):
        for mutation in ("fraction", "adaptive", "steps"):
            with self.subTest(mutation=mutation):
                config = copy.deepcopy(build_config())
                if mutation == "fraction":
                    config["social_learning"]["receiver"]["local_ce_real_fraction"] = 0.03
                elif mutation == "adaptive":
                    config["domain_mix_diagnostic"]["adaptive_after_completed_fraction_curve"] = False
                else:
                    config["social_learning"]["receiver"]["optimizer_steps"] = 3779
                with self.assertRaises(Exception):
                    validate_config(config)


if __name__ == "__main__":
    unittest.main()
