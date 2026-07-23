import unittest

from scripts.prepare_pat_class_split_experiments import (
    MODEL_ORDER,
    SPARSE_EVALUATION_ITERATIONS,
    build_config,
    build_class_split,
    build_model_split,
)
from config_adapter import load_config
from agent_data import build_agent_args


class PATClassSplitConfigTest(unittest.TestCase):
    def test_seeded_class_splits_are_disjoint_complete_and_deterministic(self):
        for num_agents in (5, 10):
            split = build_class_split(num_agents, seed=0)
            self.assertEqual(split, build_class_split(num_agents, seed=0))
            groups = [split[f"agent_{agent_id}"] for agent_id in range(num_agents)]
            self.assertTrue(all(len(group) == 100 // num_agents for group in groups))
            self.assertEqual(sorted(class_id for group in groups for class_id in group), list(range(100)))
            self.assertNotEqual(split, build_class_split(num_agents, seed=1))

    def test_pat5_groups_are_pairs_of_pat10_groups(self):
        split5 = build_class_split(5, seed=0)
        split10 = build_class_split(10, seed=0)
        for agent_id in range(5):
            expected = split10[f"agent_{2 * agent_id}"] + split10[f"agent_{2 * agent_id + 1}"]
            self.assertEqual(split5[f"agent_{agent_id}"], expected)

    def test_five_models_occur_once_or_twice(self):
        for num_agents, repetitions in ((5, 1), (10, 2)):
            split = build_model_split(num_agents)
            counts = {model: list(split.values()).count(model) for model in MODEL_ORDER}
            self.assertEqual(counts, {model: repetitions for model in MODEL_ORDER})

    def test_generated_configs_enable_sparse_dsdm_evaluation(self):
        base = load_config("configs/main_cifar100_one_resnet_seed0_ipc10.yaml")
        for num_agents in (5, 10):
            config = build_config(base, num_agents, seed=0)
            self.assertEqual(
                config["distillation"]["evaluate_iterations"],
                SPARSE_EVALUATION_ITERATIONS,
            )
            expected_clip = 100.0 if num_agents == 10 else None
            self.assertEqual(
                config["model_pool"]["models"]["alexnet"]["distillation"].get("grad_clip_norm"),
                expected_clip,
            )

    def test_pat10_alexnet_uses_stabilized_distillation_recipe(self):
        path = "configs/pat_class_split/main_cifar100_pat10agent_seed0_ipc10.yaml"
        config = load_config(path)
        alexnet_args = build_agent_args(config, path, 2)
        convnet_args = build_agent_args(config, path, 1)
        self.assertEqual(alexnet_args.net_type, "alexnet")
        self.assertEqual(alexnet_args.f_idx, "7")
        self.assertEqual(alexnet_args.lr_img, 0.005)
        self.assertEqual(alexnet_args.grad_clip_norm, 100.0)
        self.assertEqual(convnet_args.grad_clip_norm, 0.0)


if __name__ == "__main__":
    unittest.main()
