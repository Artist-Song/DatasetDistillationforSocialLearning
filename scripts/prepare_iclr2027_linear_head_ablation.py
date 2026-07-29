#!/usr/bin/env python3
"""Generate immutable matched-linear ICLR 2027 DKP diagnostic configs."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_iclr2027_dkp_protocol import (  # noqa: E402
    CATALOG,
    POOL_PROTOCOL,
    TARGET_DIR,
    _variant as _cosine_variant,
)


EXPERT_RUN = "cifar100_5agent20cls_dkp_linear_experts_seed0_v1"
CE_ONLY_RUN = "cifar100_5agent20cls_dkp_linear_ce_only_ipc10_seed0_v1"
FULL_RUN = "cifar100_5agent20cls_dkp_linear_full_ipc10_seed0_v1"
COMPARABILITY_GROUP = "cifar100_5agent20cls_iclr2027_dkp_linear_head_seed0_ipc10"
TARGETS = {
    "cifar100_5agent20cls_dkp_linear_experts_seed0_v1.yaml": (EXPERT_RUN, "local_expert"),
    "cifar100_5agent20cls_dkp_linear_ce_only_ipc10_seed0_v1.yaml": (CE_ONLY_RUN, "ce_only"),
    "cifar100_5agent20cls_dkp_linear_full_ipc10_seed0_v1.yaml": (FULL_RUN, "full"),
}


def build_linear_config(run_name: str, variant: str) -> dict:
    """Derive a matched config without mutating the completed cosine configs."""
    config = copy.deepcopy(_cosine_variant(run_name, variant))
    project = config["project"]
    project["run_name"] = run_name
    project["comparability_group"] = COMPARABILITY_GROUP
    project["paper_eligible"] = False
    project["protocol_status"] = "planned_diagnostic"
    project["diagnostic_role"] = "matched_linear_classifier_ablation"
    if variant == "local_expert":
        project["stage"] = "iclr2027_linear_expert_training"
    else:
        project["stage"] = "iclr2027_linear_head_diagnostic"

    for model in config["model_pool"]["models"].values():
        model["classifier"] = {"type": "linear"}
        expert = model.setdefault("expert_training", {})
        expert["masked_local_ce"] = True

    receiver = config["social_learning"]["receiver"]
    if variant == "ce_only":
        receiver["loss_switches"] = {"fr": False, "kd": False, "supcon": False}
        receiver["checkpoint_retention"] = "final_only"
        config["expert_reuse"]["source_run"] = EXPERT_RUN
    elif variant == "full":
        receiver["loss_switches"] = {"fr": True, "kd": True, "supcon": True}
        receiver["checkpoint_retention"] = "final_only"
        config["expert_reuse"]["source_run"] = EXPERT_RUN
    return config


def validate_linear_config(config: dict, variant: str) -> None:
    """Fail closed on any drift from the matched-linear contract."""
    class_split = config["agents"]["class_split"]
    groups = [[int(value) for value in class_split[f"agent_{agent_id}"]] for agent_id in range(5)]
    flat = [class_id for group in groups for class_id in group]
    if any(len(group) != 20 or len(set(group)) != 20 for group in groups):
        raise ValueError("matched-linear config requires five unique 20-class sets")
    if len(flat) != 100 or len(set(flat)) != 100 or set(flat) != set(range(100)):
        raise ValueError("matched-linear class sets must partition global labels 0-99")
    expected_models = [
        "convnet3w1",
        "convnet4w15",
        "alexnet",
        "resnet10_standard",
        "resnet18_standard",
    ]
    actual_models = [config["agents"]["model_split"][f"agent_{index}"] for index in range(5)]
    if actual_models != expected_models:
        raise ValueError(f"matched-linear model assignment drifted: {actual_models}")
    for model_name in expected_models:
        model = config["model_pool"]["models"][model_name]
        if model.get("classifier") != {"type": "linear"}:
            raise ValueError(f"{model_name}: classifier is not the existing linear head")
        if model.get("expert_training", {}).get("masked_local_ce") is not True:
            raise ValueError(f"{model_name}: masked local CE is disabled")
    if int(config["distillation"]["ipc"]) != 10:
        raise ValueError("matched-linear first round must use IPC=10")
    if config["communication"].get("pool_catalog") != CATALOG:
        raise ValueError("matched-linear config must reuse the immutable v2 pool catalog")
    if config.get("pool_reuse", {}).get("protocol") != POOL_PROTOCOL:
        raise ValueError("matched-linear pool protocol drifted")
    if config["project"].get("paper_eligible") is not False:
        raise ValueError("matched-linear run must remain paper-ineligible diagnostic evidence")
    if variant in {"ce_only", "full"}:
        receiver = config["social_learning"]["receiver"]
        expected_switches = {
            "ce_only": {"fr": False, "kd": False, "supcon": False},
            "full": {"fr": True, "kd": True, "supcon": True},
        }[variant]
        if receiver.get("loss_switches") != expected_switches:
            raise ValueError(f"{variant}: loss switches drifted")
        if receiver.get("checkpoint_retention") != "final_only":
            raise ValueError(f"{variant}: receiver retention is not final_only")
        if config.get("expert_reuse", {}).get("source_run") != EXPERT_RUN:
            raise ValueError(f"{variant}: expert source is not the matched linear run")
        expected_logits = variant == "full"
        if config["communication"].get("use_sender_logits") is not expected_logits:
            raise ValueError(f"{variant}: sender-logit switch drifted")


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for filename, (run_name, variant) in TARGETS.items():
        config = build_linear_config(run_name, variant)
        validate_linear_config(config, variant)
        target = TARGET_DIR / filename
        if target.exists():
            if _load(target) != config:
                raise FileExistsError(f"refusing to replace a different linear config: {target}")
            print(f"{target.relative_to(ROOT)} (reused identical)")
            continue
        target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        print(target.relative_to(ROOT))


if __name__ == "__main__":
    main()
