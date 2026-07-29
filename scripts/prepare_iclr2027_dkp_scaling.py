#!/usr/bin/env python3
"""Generate preregistered CIFAR-100 DKP scaling and hard-label baseline configs."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from packet_integrity import build_strict_dkp_protocol, resolve_strict_dkp_contract  # noqa: E402


BASE_CONFIG = ROOT / "configs/iclr2027/cifar100_5agent20cls_dkp_domain_mix_r02_full_steps3780_ipc10_seed0_v2.yaml"
TARGET_DIR = ROOT / "configs/iclr2027/scaling"
POOL_CATALOG = "configs/packet_pools/cifar100_fullclass_ipc10_seed0_dkp_v2.yaml"
BACKBONES = (
    "convnet3w1",
    "convnet4w15",
    "alexnet",
    "resnet10_standard",
    "resnet18_standard",
)
SUPPORTED_AGENT_COUNTS = (5, 10, 20)
DEFAULT_SEEDS = (1, 2, 3)
FAST_COMMIT = "6a218fcfdc93838634921399b0de6a36cdd29756"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def classes_per_agent(agent_count: int) -> int:
    agent_count = int(agent_count)
    if agent_count not in SUPPORTED_AGENT_COUNTS:
        raise ValueError(f"agent_count must be one of {SUPPORTED_AGENT_COUNTS}")
    return 100 // agent_count


def nested_class_split(seed: int, agent_count: int) -> dict[str, list[int]]:
    """Split each of five 20-class macro agents into 1/2/4 nested children."""
    per_agent = classes_per_agent(agent_count)
    children_per_macro = int(agent_count) // len(BACKBONES)
    permutation = np.random.RandomState(int(seed)).permutation(100).tolist()
    split = {}
    agent_id = 0
    for macro_id in range(len(BACKBONES)):
        macro = permutation[macro_id * 20 : (macro_id + 1) * 20]
        for child_id in range(children_per_macro):
            start = child_id * per_agent
            split[f"agent_{agent_id}"] = macro[start : start + per_agent]
            agent_id += 1
    return split


def nested_model_split(agent_count: int) -> dict[str, str]:
    children_per_macro = int(agent_count) // len(BACKBONES)
    result = {}
    agent_id = 0
    for backbone in BACKBONES:
        for _ in range(children_per_macro):
            result[f"agent_{agent_id}"] = backbone
            agent_id += 1
    return result


def run_stem(seed: int, agent_count: int) -> str:
    per_agent = classes_per_agent(agent_count)
    return f"cifar100_{agent_count}agent{per_agent}cls"


def expert_run_name(seed: int, agent_count: int) -> str:
    return f"{run_stem(seed, agent_count)}_dkp_cosine_experts_seed{int(seed)}_v1"


def method_run_name(seed: int, agent_count: int, method: str) -> str:
    stem = run_stem(seed, agent_count)
    names = {
        "dkp_full": f"{stem}_dkp_r02_full_steps3780_ipc10_seed{int(seed)}_v1",
        "heuristic": f"{stem}_heuristic_hard_r02_steps3780_ipc10_seed{int(seed)}_v1",
        "fast": f"{stem}_fast_hard_r02_steps3780_ipc10_seed{int(seed)}_v1",
        "full_real": f"{stem}_full_real_r02_steps3780_seed{int(seed)}_v1",
    }
    try:
        return names[str(method)]
    except KeyError as error:
        raise ValueError(f"unsupported scaling method: {method}") from error


def config_path(seed: int, agent_count: int, role: str) -> Path:
    run_name = (
        expert_run_name(seed, agent_count)
        if role == "expert"
        else method_run_name(seed, agent_count, role)
    )
    return TARGET_DIR / f"{run_name}.yaml"


def _scaling_base(seed: int, agent_count: int) -> dict:
    config = copy.deepcopy(_load(BASE_CONFIG))
    per_agent = classes_per_agent(agent_count)
    protocol = build_strict_dkp_protocol(agent_count, per_agent, 10)
    config["dataset"].update(
        {
            "partition": f"nested_{agent_count}agent_class_disjoint_fullclass_pool_slice",
            "class_assignment_seed": int(seed),
        }
    )
    config["agents"].update(
        {
            "num_agents": int(agent_count),
            "num_classes": 100,
            "class_split": nested_class_split(seed, agent_count),
            "model_split": nested_model_split(agent_count),
        }
    )
    config["communication"].update(
        {
            "protocol": protocol,
            "pool_catalog": POOL_CATALOG,
            "strict_packet_validation": True,
        }
    )
    receiver = config["social_learning"]["receiver"]
    receiver.update(
        {
            "protocol": "dkp_sl_v1",
            "dkp_variant": "full",
            "loss_switches": {"fr": True, "kd": True, "supcon": True},
            "local_ce_source": "real_packet_mix",
            "local_ce_real_fraction": 0.02,
            "optimizer_steps": 3780,
            "scheduler_unit": "optimizer_step",
            "scheduler_step_milestones": [2457, 3213],
            "packet_raw_per_class": 10,
            "prototype_decoded_per_class": 40,
            "checkpoint_retention": "final_only",
        }
    )
    config["runtime"].update(
        {
            "seed": int(seed),
            "workers": 3,
            "device": "cuda",
            "gpu_id": 0,
        }
    )
    config.pop("domain_mix_diagnostic", None)
    config.pop("loss_ablation", None)
    config["scaling_protocol"] = {
        "role": "preregistered_independent_seed_scaling",
        "source_selection_seed": 0,
        "evaluation_seed": int(seed),
        "agent_count": int(agent_count),
        "classes_per_agent": per_agent,
        "nested_from_five_macro_agents": True,
        "fixed_method": "r02_full",
        "fixed_optimizer_steps": 3780,
        "fixed_ipc": 10,
        "test_metric_adaptation_allowed": False,
        "all_receivers_required": True,
    }
    return config


def build_expert_config(seed: int, agent_count: int) -> dict:
    config = _scaling_base(seed, agent_count)
    run_name = expert_run_name(seed, agent_count)
    config["project"].update(
        {
            "stage": "iclr2027_dkp_scaling_cosine_experts",
            "run_name": run_name,
            "comparability_group": f"{run_stem(seed, agent_count)}_dkp_scaling_ipc10",
            "paper_eligible": False,
            "protocol_status": "preregistered_training",
        }
    )
    config["communication"].update(
        {
            "enabled": False,
            "use_sender_logits": False,
            "use_generalist_logits": False,
        }
    )
    config["logits"].update({"enabled": False, "lambda_kd": 0.0})
    config.pop("expert_reuse", None)
    return config


def build_dkp_config(seed: int, agent_count: int) -> dict:
    config = _scaling_base(seed, agent_count)
    run_name = method_run_name(seed, agent_count, "dkp_full")
    config["project"].update(
        {
            "stage": "iclr2027_dkp_scaling_r02_full",
            "run_name": run_name,
            "comparability_group": f"{run_stem(seed, agent_count)}_ipc10_image_budget",
            "paper_eligible": False,
            "protocol_status": "preregistered_validation",
        }
    )
    config["communication"].update(
        {
            "enabled": True,
            "use_sender_logits": True,
            "use_generalist_logits": False,
        }
    )
    config["logits"].update({"enabled": True, "lambda_kd": 0.6, "temperature": 2.0})
    config["expert_reuse"] = {
        "source_run": expert_run_name(seed, agent_count),
        "require_checkpoint_sha_match": True,
    }
    resolve_strict_dkp_contract(config)
    return config


def build_hard_label_config(seed: int, agent_count: int, method: str) -> dict:
    if method not in {"heuristic", "fast", "full_real"}:
        raise ValueError(f"unsupported hard-label method: {method}")
    config = _scaling_base(seed, agent_count)
    per_class = 500 if method == "full_real" else 10
    run_name = method_run_name(seed, agent_count, method)
    comparison_group = (
        f"{run_stem(seed, agent_count)}_full_data_oracle"
        if method == "full_real"
        else f"{run_stem(seed, agent_count)}_ipc10_image_budget"
    )
    config["project"].update(
        {
            "stage": f"iclr2027_{method}_hard_label_scaling",
            "run_name": run_name,
            "comparability_group": comparison_group,
            "paper_eligible": False,
            "protocol_status": "planned_baseline",
        }
    )
    config["communication"].update(
        {
            "enabled": True,
            "protocol": f"hard_label_cifar100_{agent_count}x{classes_per_agent(agent_count)}_{method}_v1",
            "strict_packet_validation": False,
            "packet_source": "sender_local_real_selection",
            "use_sender_logits": False,
            "use_generalist_logits": False,
        }
    )
    receiver = config["social_learning"]["receiver"]
    receiver.update(
        {
            "dkp_variant": "ablation_fr1_kd0_sc1",
            "loss_switches": {"fr": True, "kd": False, "supcon": True},
            "lambda_fr": 0.2,
            "lambda_sc": 0.1,
            "packet_raw_per_class": per_class,
            "prototype_decoded_per_class": per_class,
        }
    )
    config["logits"].update({"enabled": False, "lambda_kd": 0.0, "temperature": 2.0})
    config["expert_reuse"] = {
        "source_run": expert_run_name(seed, agent_count),
        "require_checkpoint_sha_match": True,
    }
    config.pop("pool_reuse", None)
    if method == "heuristic":
        config["selection"] = {
            "enabled": True,
            "methods": ["heuristic"],
            "heuristic": {"strategy": "random_per_class"},
        }
    elif method == "fast":
        config["selection"] = {
            "enabled": True,
            "methods": ["fast"],
            "fast": {
                "strategy": "official_pixels_per_class_minmax",
                "repo_path": "external_baselines/repos/FAST",
                "cache_root": "external_baselines/outputs/fast_cache",
                "commit": FAST_COMMIT,
                "seed": 0,
            },
        }
    else:
        config["selection"] = {"enabled": True, "methods": ["full_real"]}
    config["scaling_protocol"].update(
        {
            "method": method,
            "hard_labels_only": True,
            "sender_logits": False,
            "packet_raw_per_class": per_class,
        }
    )
    return config


def validate_config(config: dict, role: str) -> None:
    agent_count = int(config["agents"]["num_agents"])
    per_agent = classes_per_agent(agent_count)
    split = config["agents"]["class_split"]
    models = config["agents"]["model_split"]
    if len(split) != agent_count or len(models) != agent_count:
        raise ValueError(f"{role}: incomplete agent split")
    flat = [int(class_id) for values in split.values() for class_id in values]
    if sorted(flat) != list(range(100)) or any(len(values) != per_agent for values in split.values()):
        raise ValueError(f"{role}: class split is not a balanced partition of 0-99")
    expected_models = nested_model_split(agent_count)
    if models != expected_models:
        raise ValueError(f"{role}: nested backbone assignment drifted")
    receiver = config["social_learning"]["receiver"]
    if int(receiver.get("optimizer_steps", -1)) != 3780:
        raise ValueError(f"{role}: optimizer step count drifted")
    if float(receiver.get("local_ce_real_fraction", -1)) != 0.02:
        raise ValueError(f"{role}: r02 local CE fraction drifted")
    if role in {"expert", "dkp_full"}:
        contract = resolve_strict_dkp_contract(config)
        if contract.agent_count != agent_count or contract.classes_per_agent != per_agent:
            raise ValueError(f"{role}: strict DKP contract drifted")


def write_config(config: dict, target: Path) -> str:
    validate_config(config, "expert" if "cosine_experts" in target.name else "dkp_full" if "dkp_r02_full" in target.name else "baseline")
    if target.exists():
        if _load(target) != config:
            raise FileExistsError(f"refusing to replace a different scaling config: {target}")
        return "reused_identical"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return "created"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--agent-counts", nargs="+", type=int, default=list(SUPPORTED_AGENT_COUNTS))
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument("--write", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    cli = parse_args(argv)
    outputs = []
    for agent_count in cli.agent_counts:
        for seed in cli.seeds:
            configs = {
                "expert": build_expert_config(seed, agent_count),
                "dkp_full": build_dkp_config(seed, agent_count),
            }
            if cli.include_baselines:
                for method in ("heuristic", "fast", "full_real"):
                    configs[method] = build_hard_label_config(seed, agent_count, method)
            for role, config in configs.items():
                validate_config(config, role)
                target = config_path(seed, agent_count, role)
                status = write_config(config, target) if cli.write else "validated"
                outputs.append(f"{target.relative_to(ROOT)} ({status})")
    print("\n".join(outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
