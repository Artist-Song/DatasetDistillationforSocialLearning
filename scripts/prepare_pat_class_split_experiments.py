from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = ROOT / "configs/main_cifar100_one_resnet_seed0_ipc10.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "configs/pat_class_split"
MODEL_ORDER = ["convnet3w1", "convnet4w15", "alexnet", "resnet10", "resnet18"]
SPARSE_EVALUATION_ITERATIONS = [100, 500, 1000, 2000, 3000, 5000, 7500, 10000]


def build_class_split(num_agents: int, seed: int) -> dict[str, list[int]]:
    if num_agents not in {5, 10}:
        raise ValueError("PAT class-split validation only supports 5 or 10 agents")
    permutation = np.random.RandomState(seed).permutation(100).tolist()
    classes_per_agent = 100 // num_agents
    return {
        f"agent_{agent_id}": permutation[
            agent_id * classes_per_agent : (agent_id + 1) * classes_per_agent
        ]
        for agent_id in range(num_agents)
    }


def build_model_split(num_agents: int) -> dict[str, str]:
    return {
        f"agent_{agent_id}": MODEL_ORDER[agent_id % len(MODEL_ORDER)]
        for agent_id in range(num_agents)
    }


def validate_protocol_config(config: dict, num_agents: int, seed: int) -> None:
    dataset = config["dataset"]
    agents = config["agents"]
    class_split = agents["class_split"]
    model_split = agents["model_split"]

    if dataset.get("partition") != "pat_class_disjoint":
        raise ValueError("dataset.partition must be pat_class_disjoint")
    if dataset.get("source_split") != "original_cifar100_train_test":
        raise ValueError("PAT validation must preserve the official CIFAR-100 train/test split")
    if int(dataset.get("class_assignment_seed")) != seed:
        raise ValueError("class_assignment_seed does not match runtime seed")
    if int(agents["num_agents"]) != num_agents:
        raise ValueError("agents.num_agents does not match class split")

    expected_size = 100 // num_agents
    groups = [class_split[f"agent_{agent_id}"] for agent_id in range(num_agents)]
    if any(len(group) != expected_size for group in groups):
        raise ValueError("every agent must receive the same number of classes")
    flattened = [int(class_id) for group in groups for class_id in group]
    if sorted(flattened) != list(range(100)):
        raise ValueError("agent classes must be a disjoint cover of global labels 0-99")

    expected_models = build_model_split(num_agents)
    if model_split != expected_models:
        raise ValueError("model assignment does not follow the fixed five-model cycle")
    counts = {model: list(model_split.values()).count(model) for model in MODEL_ORDER}
    expected_count = num_agents // len(MODEL_ORDER)
    if any(count != expected_count for count in counts.values()):
        raise ValueError("each heterogeneous model must occur equally often")


def build_config(base: dict, num_agents: int, seed: int) -> dict:
    classes_per_agent = 100 // num_agents
    config = copy.deepcopy(base)
    run_id = f"cifar100_pat{num_agents}agent_{classes_per_agent}cls_ours_seed{seed}_ipc10"

    config["project"]["stage"] = f"pat{num_agents}agent_seed{seed}_ipc10_validation"
    config["project"]["run_name"] = run_id
    config["project"]["comparability_group"] = (
        f"cifar100_pat{num_agents}agent_{classes_per_agent}cls_ours_ipc10"
    )
    config["dataset"]["partition"] = "pat_class_disjoint"
    config["dataset"]["source_split"] = "original_cifar100_train_test"
    config["dataset"]["class_assignment"] = "numpy_randomstate_permutation"
    config["dataset"]["class_assignment_seed"] = seed
    config["agents"]["num_agents"] = num_agents
    config["agents"]["class_split"] = build_class_split(num_agents, seed)
    config["agents"]["model_split"] = build_model_split(num_agents)
    config["distillation"]["evaluate_iterations"] = list(SPARSE_EVALUATION_ITERATIONS)
    if num_agents == 10:
        config["model_pool"]["models"]["alexnet"]["distillation"]["grad_clip_norm"] = 100.0
    config["runtime"]["seed"] = seed
    validate_protocol_config(config, num_agents, seed)
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate our CIFAR-100 pipeline configs with PAT-style class-disjoint agents."
    )
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for num_agents in (5, 10):
        config = build_config(base, num_agents, args.seed)
        path = args.output_dir / f"main_cifar100_pat{num_agents}agent_seed{args.seed}_ipc10.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
