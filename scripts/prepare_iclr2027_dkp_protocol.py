#!/usr/bin/env python3
"""Prepare the isolated seed0 ICLR 2027 DKP-SL diagnostic configs."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/pat_class_split/main_cifar100_pat5agent_seed0_ipc10.yaml"
MODEL_SOURCES = {
    "convnet3w1": ROOT / "configs/teacher_quality/train_conv3_seed0.yaml",
    "convnet4w15": ROOT / "configs/teacher_quality/train_conv4_seed0.yaml",
    "alexnet": ROOT / "configs/teacher_quality/train_alexnet_seed0.yaml",
    "resnet10_standard": ROOT / "configs/teacher_quality/train_resnet10_standard_seed0.yaml",
    "resnet18_standard": ROOT / "configs/teacher_quality/train_resnet18_standard_seed0.yaml",
}
TARGET_DIR = ROOT / "configs/iclr2027"
CATALOG = "configs/packet_pools/cifar100_fullclass_ipc10_seed0_dkp_v2.yaml"
PROTOCOL = "dkp_sl_iclr2027_cifar100_5x20_ipc10_v1"
POOL_PROTOCOL = "backbone_specific_fullclass_pool_slice_v2_dkp"
EXPERT_RUN = "cifar100_5agent20cls_dkp_cosine_experts_seed0_v1"
CE_ONLY_RUN = "cifar100_5agent20cls_dkp_ce_only_ipc10_seed0_v2"
FULL_RUN = "cifar100_5agent20cls_dkp_sl_full_ipc10_seed0_v2"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _model_configs() -> dict:
    models = {}
    for model_name, source_path in MODEL_SOURCES.items():
        source = _load(source_path)
        model = copy.deepcopy(source["model_pool"]["models"][model_name])
        model["classifier"] = {"type": "cosine", "scale_init": 10.0}
        expert = model.setdefault("expert_training", {})
        expert.update(
            {
                "separate": True,
                "masked_local_ce": True,
                "validation_fraction": 0.1,
                "eval_interval": 5,
                "retrain_full": True,
            }
        )
        models[model_name] = model
    return models


def _base_config() -> dict:
    config = _load(BASE)
    config["project"].update(
        {
            "stage": "iclr2027_dkp_sl_diagnostic",
            "comparability_group": "cifar100_5agent20cls_iclr2027_dkp_seed0_ipc10",
            "paper_eligible": False,
            "protocol_status": "planned_diagnostic",
        }
    )
    config["dataset"].update(
        {
            "partition": "pat5_class_disjoint_fullclass_pool_slice",
            "source_split": "original_cifar100_train_test",
        }
    )
    config["model_pool"] = {"default": "convnet3w1", "models": _model_configs()}
    config["agents"]["model_split"] = {
        "agent_0": "convnet3w1",
        "agent_1": "convnet4w15",
        "agent_2": "alexnet",
        "agent_3": "resnet10_standard",
        "agent_4": "resnet18_standard",
    }
    config["communication"].update(
        {
            "enabled": True,
            "protocol": PROTOCOL,
            "receiver_protocol": "dkp_sl_v1",
            "mode": "logical_all_share_once",
            "packet_source": "backbone_specific_fullclass_pool_slice",
            "pool_catalog": CATALOG,
            "strict_packet_validation": True,
            "use_generalist_logits": False,
        }
    )
    config["social_learning"]["receiver"].update(
        {
            "protocol": "dkp_sl_v1",
            "epochs": 60,
            "lr": 0.01,
            "local_batch_size": 64,
            "external_batch_size": 64,
            "self_data_mode": "real",
            "self_real_per_class": 0,
            "augment": True,
            "scheduler": "multistep",
            "scheduler_milestones": [39, 51],
            "scheduler_gamma": 0.2,
            "prototype_decoded_per_class": 40,
            "lambda_sc": 0.1,
            "supcon_temperature": 0.07,
        }
    )
    config["logits"].update({"temperature": 2.0})
    config["pool_reuse"] = {
        "protocol": POOL_PROTOCOL,
        "catalog": CATALOG,
        "source_scope": "full_cifar100_seed0",
        "sender_slice": "configured_expert_classes",
        "global_labels_preserved": True,
        "simulated_local_distillation": True,
        "guide_weights_communicated": False,
        "paper_role": "communication_logic_validation",
        "immutable_sources": True,
    }
    config["runtime"].update({"seed": 0, "workers": 8, "device": "cuda", "gpu_id": 0})
    return config


def _variant(run_name: str, variant: str) -> dict:
    config = _base_config()
    config["project"]["run_name"] = run_name
    receiver = config["social_learning"]["receiver"]
    receiver["dkp_variant"] = variant
    if variant == "local_expert":
        config["project"]["stage"] = "iclr2027_cosine_expert_training"
        config["communication"]["enabled"] = False
        config["communication"]["use_sender_logits"] = False
        receiver.update({"lambda_fr": 0.0, "lambda_sc": 0.0})
        config["logits"].update({"enabled": False, "lambda_kd": 0.0})
    elif variant == "ce_only":
        config["communication"]["use_sender_logits"] = False
        receiver.update({"lambda_fr": 0.0, "lambda_sc": 0.0})
        config["logits"].update({"enabled": False, "lambda_kd": 0.0})
        config["expert_reuse"] = {
            "source_run": EXPERT_RUN,
            "require_checkpoint_sha_match": True,
        }
    elif variant == "full":
        config["communication"]["use_sender_logits"] = True
        receiver.update({"lambda_fr": 0.2, "lambda_sc": 0.1})
        config["logits"].update({"enabled": True, "lambda_kd": 0.6})
        config["expert_reuse"] = {
            "source_run": EXPERT_RUN,
            "require_checkpoint_sha_match": True,
        }
    else:
        raise ValueError(f"unsupported variant: {variant}")
    return config


def _validate(config: dict) -> None:
    split = config["agents"]["class_split"]
    groups = [[int(c) for c in split[f"agent_{agent_id}"]] for agent_id in range(5)]
    if any(len(group) != 20 for group in groups):
        raise ValueError("ICLR 2027 seed0 config requires exactly 20 classes per sender")
    flat = [class_id for group in groups for class_id in group]
    if len(set(flat)) != 100 or set(flat) != set(range(100)):
        raise ValueError("ICLR 2027 seed0 class split must be a disjoint partition of 0-99")
    expected_models = [
        "convnet3w1",
        "convnet4w15",
        "alexnet",
        "resnet10_standard",
        "resnet18_standard",
    ]
    actual_models = [config["agents"]["model_split"][f"agent_{i}"] for i in range(5)]
    if actual_models != expected_models:
        raise ValueError(f"unexpected model assignment: {actual_models}")
    if config["communication"].get("pool_catalog") != CATALOG:
        raise ValueError("ICLR 2027 DKP config must use the immutable v2 pool catalog")
    if config.get("pool_reuse", {}).get("protocol") != POOL_PROTOCOL:
        raise ValueError("ICLR 2027 DKP config pool protocol drifted")
    for model_name in expected_models:
        classifier = config["model_pool"]["models"][model_name]["classifier"]
        if classifier != {"type": "cosine", "scale_init": 10.0}:
            raise ValueError(f"{model_name}: cosine classifier contract drifted")


def main() -> None:
    targets = {
        "cifar100_5agent20cls_dkp_cosine_experts_seed0_v1.yaml": _variant(EXPERT_RUN, "local_expert"),
        "cifar100_5agent20cls_dkp_ce_only_ipc10_seed0_v2.yaml": _variant(CE_ONLY_RUN, "ce_only"),
        "cifar100_5agent20cls_dkp_sl_full_ipc10_seed0_v2.yaml": _variant(FULL_RUN, "full"),
    }
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for filename, config in targets.items():
        _validate(config)
        target = TARGET_DIR / filename
        if target.exists():
            if _load(target) != config:
                raise FileExistsError(f"refusing to replace a different protocol config: {target}")
            print(f"{target.relative_to(ROOT)} (reused identical)")
            continue
        target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        print(target.relative_to(ROOT))


if __name__ == "__main__":
    main()
