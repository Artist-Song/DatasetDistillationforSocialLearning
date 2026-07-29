#!/usr/bin/env python3
"""Prepare fixed-step S/U/H receiver diagnostics for DKP domain balance."""

from __future__ import annotations

import copy
import argparse
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_adapter import build_dsdm_args_from_config  # noqa: E402


BASE_CONFIG = (
    ROOT
    / "configs/iclr2027/cifar100_5agent20cls_dkp_ablation_fr1_kd0_sc1_ipc10_seed0_v1.yaml"
)
TARGET_DIR = ROOT / "configs/iclr2027"
CONDITIONS = {
    "s_real": "real",
    "u_packet": "packet",
    "h_real_packet": "real_packet_50_50",
}
EXPECTED_STEPS = 3780
EXPECTED_STEP_MILESTONES = [2457, 3213]
EXPECTED_LOSSES = {"fr": True, "kd": False, "supcon": True}
EXPECTED_MODELS = {
    "agent_0": "convnet3w1",
    "agent_1": "convnet4w15",
    "agent_2": "alexnet",
    "agent_3": "resnet10_standard",
    "agent_4": "resnet18_standard",
}


class DomainBalanceConfigError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise DomainBalanceConfigError(message)


def run_name(condition):
    return f"cifar100_5agent20cls_dkp_domain_{condition}_steps3780_ipc10_seed0_v1"


def _load(path):
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"config is not a mapping: {path}")
    return payload


def validate_config(config, condition, base=None):
    _require(condition in CONDITIONS, f"unknown condition: {condition}")
    source = CONDITIONS[condition]
    project = config.get("project", {})
    receiver = config.get("social_learning", {}).get("receiver", {})
    diagnostic = config.get("domain_balance_diagnostic", {})
    _require(project.get("run_name") == run_name(condition), f"{condition}: run_name mismatch")
    _require(project.get("paper_eligible") is False, f"{condition}: must be paper-ineligible")
    _require(project.get("protocol_status") == "planned_diagnostic", f"{condition}: status drifted")
    _require(receiver.get("local_ce_source") == source, f"{condition}: local CE source mismatch")
    _require(receiver.get("loss_switches") == EXPECTED_LOSSES, f"{condition}: loss switches drifted")
    _require(receiver.get("dkp_variant") == "ablation_fr1_kd0_sc1", f"{condition}: variant drifted")
    _require(receiver.get("checkpoint_retention") == "final_only", f"{condition}: retention drifted")
    _require(receiver.get("optimizer_steps") == EXPECTED_STEPS, f"{condition}: optimizer steps drifted")
    _require(receiver.get("scheduler_unit") == "optimizer_step", f"{condition}: scheduler unit drifted")
    _require(receiver.get("scheduler_step_milestones") == EXPECTED_STEP_MILESTONES, f"{condition}: step milestones drifted")
    _require(receiver.get("scheduler_milestones") == [39, 51], f"{condition}: reference milestones drifted")
    _require(receiver.get("self_packet_batch_size") == 64, f"{condition}: self-packet batch drifted")
    _require(float(receiver.get("lambda_fr", -1)) == 0.2, f"{condition}: lambda_FR drifted")
    _require(float(receiver.get("lambda_sc", -1)) == 0.1, f"{condition}: lambda_SC drifted")
    _require(config.get("communication", {}).get("use_sender_logits") is False, f"{condition}: sender logits must be off")
    _require(config.get("communication", {}).get("strict_packet_validation") is True, f"{condition}: strict packets disabled")
    _require(config.get("logits", {}).get("enabled") is False, f"{condition}: logits.enabled must be off")
    _require(float(config.get("logits", {}).get("lambda_kd", -1)) == 0.0, f"{condition}: lambda_KD must be zero")
    _require(config.get("agents", {}).get("model_split") == EXPECTED_MODELS, f"{condition}: model split drifted")
    class_split = config.get("agents", {}).get("class_split", {})
    groups = [class_split.get(f"agent_{agent_id}", []) for agent_id in range(5)]
    flat = [int(class_id) for group in groups for class_id in group]
    _require(all(len(group) == 20 for group in groups), f"{condition}: expected 20 classes per agent")
    _require(len(flat) == 100 and set(flat) == set(range(100)), f"{condition}: classes do not partition 0-99")
    _require(int(config.get("distillation", {}).get("ipc", -1)) == 10, f"{condition}: IPC drifted")
    _require(int(config.get("runtime", {}).get("seed", -1)) == 0, f"{condition}: seed drifted")
    _require(diagnostic.get("condition") == condition, f"{condition}: diagnostic metadata mismatch")
    _require(diagnostic.get("losses") == EXPECTED_LOSSES, f"{condition}: diagnostic loss metadata drifted")
    _require(diagnostic.get("self_packet_counted_as_external_communication") is False, f"{condition}: self packet communication metadata drifted")
    for model_name in EXPECTED_MODELS.values():
        classifier = config["model_pool"]["models"][model_name].get("classifier", {})
        _require(classifier == {"type": "cosine", "scale_init": 10.0}, f"{condition}/{model_name}: classifier drifted")
    if base is not None:
        for key in ("dataset", "distillation", "pool_reuse"):
            _require(config.get(key) == base.get(key), f"{condition}: {key} differs from base")
        _require(config.get("agents") == base.get("agents"), f"{condition}: agent protocol differs from base")


def build_configs(base_path=BASE_CONFIG):
    base = _load(base_path)
    configs = {}
    for condition, local_ce_source in CONDITIONS.items():
        config = copy.deepcopy(base)
        name = run_name(condition)
        config["project"].update(
            {
                "run_name": name,
                "stage": "iclr2027_dkp_domain_balance_diagnostic",
                "comparability_group": (
                    "cifar100_5agent20cls_iclr2027_dkp_domain_balance_seed0_ipc10"
                ),
                "paper_eligible": False,
                "protocol_status": "planned_diagnostic",
            }
        )
        receiver = config["social_learning"]["receiver"]
        receiver.update(
            {
                "local_ce_source": local_ce_source,
                "optimizer_steps": EXPECTED_STEPS,
                "self_packet_batch_size": 64,
                "scheduler_unit": "optimizer_step",
                "scheduler_step_milestones": list(EXPECTED_STEP_MILESTONES),
                "checkpoint_retention": "final_only",
            }
        )
        config["domain_balance_diagnostic"] = {
            "role": "fixed_seed0_internal_diagnostic_not_hyperparameter_search",
            "condition": condition,
            "local_ce_source": local_ce_source,
            "target_optimizer_steps": EXPECTED_STEPS,
            "historical_pat5_step_match": True,
            "scheduler_unit": "optimizer_step",
            "scheduler_step_milestones": list(EXPECTED_STEP_MILESTONES),
            "losses": dict(EXPECTED_LOSSES),
            "self_packet_counted_as_external_communication": False,
            "selection_gate": "convnet3w1_and_convnet4w15_then_five_receivers",
        }
        args = build_dsdm_args_from_config(config)
        validate_config(config, condition, base)
        _require(args.receiver_local_ce_source == local_ce_source, f"{condition}: parsed local source mismatch")
        _require(args.receiver_optimizer_steps == EXPECTED_STEPS, f"{condition}: parsed steps mismatch")
        _require(args.receiver_scheduler_unit == "optimizer_step", f"{condition}: parsed scheduler mismatch")
        _require(args.receiver_scheduler_step_milestones == EXPECTED_STEP_MILESTONES, f"{condition}: parsed milestones mismatch")
        _require(args.dkp_loss_switches == EXPECTED_LOSSES, f"{condition}: parsed switches mismatch")
        configs[f"{name}.yaml"] = config
    return configs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default=str(BASE_CONFIG))
    parser.add_argument("--target-dir", default=str(TARGET_DIR))
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    cli = parse_args(argv)
    written = []
    for filename, config in build_configs(cli.base_config).items():
        path = Path(cli.target_dir) / filename
        if path.exists():
            existing = _load(path)
            if existing != config:
                raise FileExistsError(f"refusing to replace different config: {path}")
        elif cli.check:
            raise FileNotFoundError(f"expected config is missing: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        written.append(str(path))
    print(json.dumps({"status": "passed", "configs": written}, indent=2))


if __name__ == "__main__":
    main()
