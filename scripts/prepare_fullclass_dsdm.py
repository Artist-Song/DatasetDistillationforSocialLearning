#!/usr/bin/env python3
"""Prepare one-agent, all-CIFAR-100-class DSDM pool configurations."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "configs" / "teacher_quality"
OUTPUT_ROOT = ROOT / "configs" / "fullclass_dsdm"
MODELS = (
    "conv3",
    "conv4",
    "alexnet",
    "resnet10_standard",
    "resnet18_standard",
)
MODEL_IDS = {
    "conv3": "convnet3w1",
    "conv4": "convnet4w15",
    "alexnet": "alexnet",
    "resnet10_standard": "resnet10_standard",
    "resnet18_standard": "resnet18_standard",
}


def build_config(model_name: str) -> dict:
    source_path = SOURCE_ROOT / f"packet_{model_name}_guidee0200_seed0_ipc10.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg = copy.deepcopy(cfg)
    run_name = f"cifar100_fullclass_dsdm_{model_name}_ipc10_seed0"
    model_id = MODEL_IDS[model_name]
    model_cfg = cfg["model_pool"]["models"][model_id]
    guide_cfg = model_cfg["guide_training"]
    guide_cfg.pop("source_root", None)
    guide_cfg["snapshot_epochs"] = [200]
    guide_cfg["selected_epoch"] = 200
    # This pool is image-only. Avoid an unnecessary second expert training run;
    # the legacy selected expert artifact is not consumed by DSDM.
    expert_cfg = model_cfg.get("expert_training", {})
    expert_cfg.pop("source_root", None)
    expert_cfg["separate"] = False

    cfg["project"].update(
        {
            "stage": "fullclass_dsdm_pool",
            "run_name": run_name,
            "comparability_group": "cifar100_fullclass_dsdm_seed0",
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
        "model_split": {"agent_0": model_id},
    }
    cfg["distillation"].update(
        {
            "ipc": 10,
            "niter": 10000,
            "evaluate_iterations": [100, 500, 1000, 2000, 3000, 5000, 7500, 10000],
        }
    )
    cfg["communication"] = {
        "enabled": False,
        "protocol": "none",
        "mode": "direct",
        "use_sender_logits": False,
        "use_generalist_logits": False,
    }
    cfg["logits"] = {
        "enabled": False,
        "lambda_kd": 0.0,
        "temperature": 2.0,
    }
    cfg["fullclass_pool"] = {
        "enabled": True,
        "class_count": 100,
        "class_ids": list(range(100)),
        "raw_images_per_class": 10,
        "save_class_indices": True,
        "pool_role": "backbone_specific_sender_image_pool",
        "guide_only": True,
        "logits_attached": False,
    }
    cfg.setdefault("runtime", {})["workers"] = 4
    return cfg


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for model_name in MODELS:
        cfg = build_config(model_name)
        path = OUTPUT_ROOT / f"fullclass_{model_name}_ipc10_seed0.yaml"
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=False)
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
