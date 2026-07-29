#!/usr/bin/env python3
"""Build the recoverable AlexNet recipe from the historical formal sweep."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "fullclass_dsdm"
SOURCE = CONFIG_ROOT / "fullclass_alexnet_ipc10_seed0.yaml"
OUTPUT = CONFIG_ROOT / "fullclass_alexnet_historicale0020_ipc10_seed0.yaml"
RUN_NAME = "cifar100_fullclass_dsdm_alexnet_historicale0020_ipc10_seed0"


def build_config() -> dict:
    with SOURCE.open("r", encoding="utf-8") as handle:
        cfg = copy.deepcopy(yaml.safe_load(handle))

    cfg["project"].update(
        {
            "stage": "fullclass_dsdm_historical_recipe_recovery",
            "run_name": RUN_NAME,
            "comparability_group": "cifar100_fullclass_alexnet_historical_recipe_seed0",
        }
    )

    alexnet = cfg["model_pool"]["models"]["alexnet"]
    alexnet["distillation"].update({"f_idx": "7", "lr_img": 0.005, "niter": 10000})
    alexnet["guide_training"] = {
        "num_models": 10,
        "max_epochs": 20,
        "snapshot_epochs": [20],
        "selected_epoch": 20,
        "lr": 0.01,
        "batch_size": 256,
        "augment": False,
        "scheduler": "none",
        "scheduler_milestones": [],
        "scheduler_gamma": 0.1,
        "training_style": "dsdm",
    }
    alexnet.setdefault("expert_training", {})["separate"] = False
    alexnet["expert_training"].pop("source_root", None)

    cfg["distillation"].update(
        {
            "method": "dsdm",
            "ipc": 10,
            "factor": 2,
            "init": "mix",
            "decode_type": "single",
            "aug_type": "color_crop_cutout",
            "match": "semantic",
            "metric": "mse",
            "f_idx": "7",
            "niter": 10000,
            "evaluate_iter": 500,
            "evaluate_iterations": list(range(500, 10001, 500)),
            "lr_img": 0.005,
            "mom_img": 0.5,
            "batch_real": 256,
            "batch_syn_max": 256,
            "smooth_iter": 2000,
            "cov_weight": 50.0,
            "h_p_weight": 0.2,
            "smooth_factor": 0.99,
            "pretrained_model_number": 10,
            "pretrained_epochs": 20,
            "load_memory": True,
            "mixup": "cut",
            "mixup_net": "cut",
            "beta": 1.0,
            "mix_p": 0.5,
            "dsa": True,
            "dsa_strategy": "color_crop_cutout_flip_scale_rotate",
            "bias": False,
            "fc": False,
            "grad_clip_norm": 0.0,
            "guide_model_mode": "eval",
            "freeze_guide_parameters": True,
            "official_dsdm_protocol": False,
            "reproduce": False,
        }
    )
    cfg["fullclass_pool"].update(
        {
            "guide_protocol": "historical_alexnet_dsdm_e0020_recovery",
            "source_artifact_status": "historical_46pct_images_missing_recipe_rebuilt",
        }
    )
    return cfg


def main() -> None:
    cfg = build_config()
    with OUTPUT.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    print(OUTPUT)


if __name__ == "__main__":
    main()
