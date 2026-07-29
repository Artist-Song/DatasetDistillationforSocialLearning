#!/usr/bin/env python3
"""Prepare the isolated 4-agent communication gate that reuses full-class image pools."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/main_cifar100_one_resnet_seed0_ipc10.yaml"
R18_SOURCE = ROOT / "configs/fullclass_dsdm/fullclass_resnet18_standard_modelbest_e0200_ipc10_seed0_recovery.yaml"
TARGET = ROOT / "configs/pool_communication/main_cifar100_4agent_fullclass_pool_seed0_ipc10.yaml"
CATALOG = "configs/packet_pools/cifar100_fullclass_ipc10_seed0.yaml"


CONV_EXPERT = {
    "separate": True,
    "epochs": 500,
    "lr": 0.01,
    "batch_size": 128,
    "augment": False,
    "use_dsdm_train": True,
    "scheduler": "multistep",
    "scheduler_milestones": [333, 416],
    "scheduler_gamma": 0.2,
    "validation_fraction": 0.1,
    "eval_interval": 5,
    "retrain_full": True,
}
IMAGE_EXPERT = {
    "separate": True,
    "epochs": 500,
    "lr": 0.01,
    "batch_size": 128,
    "augment": True,
    "use_dsdm_train": False,
    "scheduler": "multistep",
    "scheduler_milestones": [333, 416],
    "scheduler_gamma": 0.2,
    "validation_fraction": 0.1,
    "eval_interval": 5,
    "retrain_full": True,
}


def main():
    config = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    r18_config = yaml.safe_load(R18_SOURCE.read_text(encoding="utf-8"))
    config["project"].update(
        {
            "stage": "fullclass_pool_communication_gate",
            "run_name": "cifar100_4agent_25cls_fullclass_pool_seed0_ipc10",
            "comparability_group": "cifar100_4agent_25cls_fullclass_pool_standard_r18_ipc10",
        }
    )
    config["dataset"].update(
        {
            "partition": "class_disjoint_fullclass_pool_slice",
            "source_split": "original_cifar100_train_test",
        }
    )
    models = config["model_pool"]["models"]
    models["convnet3w1"]["expert_training"] = copy.deepcopy(CONV_EXPERT)
    models["convnet4w15"]["expert_training"] = copy.deepcopy(CONV_EXPERT)
    models["alexnet"]["expert_training"] = copy.deepcopy(IMAGE_EXPERT)
    models["resnet18_standard"] = copy.deepcopy(r18_config["model_pool"]["models"]["resnet18_standard"])
    models["resnet18_standard"]["expert_training"]["separate"] = True
    config["agents"]["model_split"]["agent_2"] = "resnet18_standard"
    config["communication"].update(
        {
            "packet_source": "backbone_specific_fullclass_pool_slice",
            "pool_catalog": CATALOG,
            "use_sender_logits": True,
        }
    )
    config["pool_reuse"] = {
        "protocol": "backbone_specific_fullclass_pool_slice_v1",
        "catalog": CATALOG,
        "source_scope": "full_cifar100_seed0",
        "sender_slice": "configured_expert_classes",
        "global_labels_preserved": True,
        "simulated_local_distillation": True,
        "guide_weights_communicated": False,
        "paper_role": "communication_logic_validation",
    }
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(TARGET.relative_to(ROOT))


if __name__ == "__main__":
    main()
