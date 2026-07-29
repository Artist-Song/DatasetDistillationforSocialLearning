#!/usr/bin/env python3
"""Prepare full-CIFAR-100 DSDM configs using the validated model recipes.

This queue is intentionally separate from the earlier uniform official-recipe
run. It keeps the model-specific image and guide settings, changes only the
class pool to global 0-99 and the guide maturity to epoch 200.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "fullclass_dsdm"
MODELS = ("alexnet", "resnet10_standard", "resnet18_standard")

MODEL_RECIPE = {
    "alexnet": {
        "source": "fullclass_alexnet_ipc10_seed0.yaml",
        "model_id": "alexnet",
        "run_name": "cifar100_fullclass_dsdm_alexnet_modelbest_e0200_ipc10_seed0",
        "lr_img": 0.005,
    },
    "resnet10_standard": {
        "source": "fullclass_resnet10_standard_ipc10_seed0.yaml",
        "model_id": "resnet10_standard",
        "run_name": "cifar100_fullclass_dsdm_resnet10_standard_modelbest_e0200_ipc10_seed0",
        "lr_img": 0.01,
    },
    "resnet18_standard": {
        "source": "fullclass_resnet18_standard_ipc10_seed0.yaml",
        "model_id": "resnet18_standard",
        "run_name": "cifar100_fullclass_dsdm_resnet18_standard_modelbest_e0200_ipc10_seed0",
        "lr_img": 0.01,
    },
}


def build_config(model_name: str) -> dict:
    spec = MODEL_RECIPE[model_name]
    with (CONFIG_ROOT / spec["source"]).open("r", encoding="utf-8") as handle:
        cfg = copy.deepcopy(yaml.safe_load(handle))

    model_cfg = cfg["model_pool"]["models"][spec["model_id"]]
    guide_cfg = model_cfg["guide_training"]
    guide_cfg.pop("source_root", None)
    guide_cfg.update(
        {
            "num_models": 10,
            "max_epochs": 200,
            "snapshot_epochs": [200],
            "selected_epoch": 200,
        }
    )
    model_cfg.setdefault("distillation", {})["lr_img"] = spec["lr_img"]
    model_cfg["distillation"]["niter"] = 10000
    model_cfg.setdefault("expert_training", {})["separate"] = False
    model_cfg["expert_training"].pop("source_root", None)

    cfg["project"].update(
        {
            "stage": "fullclass_dsdm_model_specific",
            "run_name": spec["run_name"],
            "comparability_group": "cifar100_fullclass_dsdm_model_specific_e0200_seed0",
        }
    )
    cfg["dataset"].update(
        {
            "partition": "full_class_pool",
            "source_split": "original_cifar100_train_test",
            "class_assignment": "global_0_99",
            "class_assignment_seed": 0,
        }
    )
    cfg["agents"] = {
        "num_agents": 1,
        "num_classes": 100,
        "class_split": {"agent_0": list(range(100))},
        "model_split": {"agent_0": spec["model_id"]},
    }

    # Preserve the previously validated DSDM recipe; only maturity and the
    # early-check schedule are changed for this all-class run.
    cfg["distillation"].update(
        {
            "ipc": 10,
            "factor": 2,
            "init": "mix",
            "decode_type": "single",
            "aug_type": "color_crop_cutout",
            "match": "semantic",
            "metric": "mse",
            "f_idx": "7" if model_name == "alexnet" else "5",
            "niter": 10000,
            "evaluate_iter": 500,
            "evaluate_iterations": [100, 500, 1000, 2000, 3000, 5000, 7500, 10000],
            "lr_img": spec["lr_img"],
            "pretrained_model_number": 10,
            "pretrained_epochs": 200,
        }
    )
    cfg["communication"].update(
        {
            "enabled": False,
            "use_sender_logits": False,
            "use_generalist_logits": False,
        }
    )
    cfg["logits"] = {"enabled": False, "lambda_kd": 0.0, "temperature": 2.0}
    cfg["fullclass_pool"] = {
        "enabled": True,
        "class_count": 100,
        "class_ids": list(range(100)),
        "raw_images_per_class": 10,
        "save_class_indices": True,
        "pool_role": "backbone_specific_sender_image_pool",
        "guide_only": True,
        "logits_attached": False,
        "guide_protocol": "validated_model_specific_recipe_e0200",
    }
    return cfg


def main() -> None:
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    for model_name in MODELS:
        cfg = build_config(model_name)
        out = CONFIG_ROOT / f"fullclass_{model_name}_modelbest_e0200_ipc10_seed0.yaml"
        with out.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)
        print(out)


if __name__ == "__main__":
    main()
