import unittest

from packet_integrity import resolve_strict_dkp_contract
from scripts.prepare_iclr2027_dkp_scaling import (
    BACKBONES,
    build_dkp_config,
    build_expert_config,
    build_hard_label_config,
    nested_class_split,
    nested_model_split,
    validate_config,
)
from scripts.validate_iclr2027_cosine_experts import validate_protocol_config
from scripts.validate_iclr2027_dkp_scaling import _validate_config_pair
from scripts.run_iclr2027_dkp_scaling_queue import _receiver_checkpoint_path
from run_social_pipeline import build_receiver_args


class Iclr2027DkpScalingConfigTests(unittest.TestCase):
    def test_nested_partitions_preserve_five_macro_agents(self):
        seed = 1
        split5 = nested_class_split(seed, 5)
        split10 = nested_class_split(seed, 10)
        split20 = nested_class_split(seed, 20)
        for macro_id in range(5):
            macro = split5[f"agent_{macro_id}"]
            children10 = sum(
                (split10[f"agent_{macro_id * 2 + child}"] for child in range(2)),
                [],
            )
            children20 = sum(
                (split20[f"agent_{macro_id * 4 + child}"] for child in range(4)),
                [],
            )
            self.assertEqual(children10, macro)
            self.assertEqual(children20, macro)

    def test_nested_models_repeat_within_each_macro_agent(self):
        for agent_count, repeats in ((5, 1), (10, 2), (20, 4)):
            models = nested_model_split(agent_count)
            expected = [backbone for backbone in BACKBONES for _ in range(repeats)]
            self.assertEqual(
                [models[f"agent_{agent_id}"] for agent_id in range(agent_count)],
                expected,
            )

    def test_main_configs_resolve_dynamic_strict_contracts(self):
        for agent_count, per_agent in ((5, 20), (10, 10), (20, 5)):
            with self.subTest(agent_count=agent_count):
                expert = build_expert_config(1, agent_count)
                full = build_dkp_config(1, agent_count)
                validate_config(expert, "expert")
                validate_config(full, "dkp_full")
                contract = resolve_strict_dkp_contract(full)
                self.assertEqual(contract.agent_count, agent_count)
                self.assertEqual(contract.classes_per_agent, per_agent)
                self.assertEqual(full["runtime"]["seed"], 1)
                self.assertEqual(
                    full["expert_reuse"]["source_run"],
                    expert["project"]["run_name"],
                )
                args_by_agent = validate_protocol_config(
                    expert,
                    f"seed1_{agent_count}agent.yaml",
                )
                self.assertEqual(len(args_by_agent), agent_count)
                self.assertEqual(_validate_config_pair(expert, full), contract)

    def test_receiver_args_resolve_main_and_hard_label_loss_contracts(self):
        full = build_dkp_config(1, 20)
        full_args = build_receiver_args(full, "full.yaml", 0)
        self.assertTrue(full_args.use_logits)
        self.assertTrue(full_args.use_fr)
        self.assertEqual(full_args.dkp_variant, "full")
        self.assertEqual(full_args.receiver_packet_raw_per_class, 10)

        baseline = build_hard_label_config(1, 20, "fast")
        baseline_args = build_receiver_args(
            baseline,
            "fast.yaml",
            0,
            packet_method="fast",
        )
        self.assertFalse(baseline_args.use_logits)
        self.assertTrue(baseline_args.use_fr)
        self.assertEqual(baseline_args.lambda_kd, 0.0)
        self.assertEqual(baseline_args.dkp_variant, "ablation_fr1_kd0_sc1")

    def test_hard_label_baselines_disable_logits_and_set_packet_counts(self):
        for method, per_class in (("heuristic", 10), ("fast", 10), ("full_real", 500)):
            with self.subTest(method=method):
                config = build_hard_label_config(2, 10, method)
                validate_config(config, method)
                receiver = config["social_learning"]["receiver"]
                self.assertFalse(config["communication"]["use_sender_logits"])
                self.assertFalse(config["logits"]["enabled"])
                self.assertEqual(receiver["packet_raw_per_class"], per_class)
                self.assertEqual(receiver["prototype_decoded_per_class"], per_class)
                self.assertEqual(
                    receiver["loss_switches"],
                    {"fr": True, "kd": False, "supcon": True},
                )

    def test_queue_resume_uses_canonical_receiver_agent_directory(self):
        path = _receiver_checkpoint_path(
            "configs/iclr2027/scaling/"
            "cifar100_5agent20cls_dkp_r02_full_steps3780_ipc10_seed1_v1.yaml",
            3,
        )
        self.assertEqual(
            path.parts[-5:],
            (
                "social_learning",
                "receiver_agent_3",
                "checkpoints",
                "dkp_sl_v1_full",
                "after_social.pt",
            ),
        )


if __name__ == "__main__":
    unittest.main()
