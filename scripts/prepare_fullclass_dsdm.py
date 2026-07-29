#!/usr/bin/env python3
"""Prepare one-agent, all-CIFAR-100-class DSDM pool configurations."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_DSDM_COMMIT = "cb12851831e39da6b0169da84598166ad7706e01"
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
    run_name = f"cifar100_fullclass_dsdm_{model_name}_dsdmguidee0200_ipc10_seed0"
    model_id = MODEL_IDS[model_name]
    model_cfg = cfg["model_pool"]["models"][model_id]
    guide_cfg = model_cfg["guide_training"]
    guide_cfg.pop("source_root", None)
    guide_cfg["num_models"] = 10
    guide_cfg["max_epochs"] = 200
    guide_cfg["snapshot_epochs"] = [200]
    guide_cfg["selected_epoch"] = 200
    guide_cfg["lr"] = 0.01
    guide_cfg["batch_size"] = 256
    guide_cfg["augment"] = False
    guide_cfg["scheduler"] = "none"
    guide_cfg["scheduler_milestones"] = []
    guide_cfg["scheduler_gamma"] = 0.1
    guide_cfg["training_style"] = "dsdm"
    model_cfg["distillation"]["lr_img"] = 0.1
    model_cfg["distillation"]["niter"] = 10000
    # This pool is image-only. Avoid an unnecessary second expert training run;
    # the legacy selected expert artifact is not consumed by DSDM.
    expert_cfg = model_cfg.get("expert_training", {})
    expert_cfg.pop("source_root", None)
    expert_cfg["separate"] = False

    cfg["project"].update(
        {
            "stage": "fullclass_dsdm_pool",
            "run_name": run_name,
            "comparability_group": "cifar100_fullclass_dsdm_dsdmguidee0200_seed0",
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
            "factor": 2,
            "init": "mix",
            "decode_type": "single",
            "aug_type": "color_crop_cutout",
            "match": "grad",
            "metric": "mse",
            "niter": 10000,
            "evaluate_iter": 500,
            "evaluate_iterations": list(range(500, 10001, 500)),
            "lr_img": 0.1,
            "mom_img": 0.5,
            "batch_real": 256,
            "batch_syn_max": 256,
            "smooth_iter": 2000,
            "cov_weight": 50.0,
            "h_p_weight": 0.2,
            "smooth_factor": 0.99,
            "pretrained_model_number": 10,
            "pretrained_epochs": 200,
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
            "guide_model_mode": "train",
            "freeze_guide_parameters": False,
            "official_dsdm_protocol": True,
            "official_dsdm_commit": OFFICIAL_DSDM_COMMIT,
            "reproduce": True,
        }
    )
    cfg["evaluation"] = {
        "enabled": True,
        "epochs": 1500,
        "batch_size": 64,
        "repeat": 1,
    }
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
        "guide_protocol": "dsdm_original_except_guide_epochs_200_and_eval_interval_500",
        "official_source": "https://github.com/Li-Hongcheng/DSDM",
        "official_commit": OFFICIAL_DSDM_COMMIT,
    }
    cfg.setdefault("runtime", {})["workers"] = 8
    return cfg


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for model_name in MODELS:
        cfg = build_config(model_name)
        path = OUTPUT_ROOT / f"fullclass_{model_name}_dsdmguidee0200_ipc10_seed0.yaml"
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=False)
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
