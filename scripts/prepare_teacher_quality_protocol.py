#!/usr/bin/env python3
"""Prepare seed0 guide-maturity and converged-logit quality experiments."""

import argparse
import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs" / "pat_class_split" / "main_cifar100_pat5agent_seed0_ipc10.yaml"

MODELS = {
    "conv3": {
        "agent": 0,
        "model_id": "convnet3w1",
        "guide_epochs": [20, 50, 100, 200],
        "guide_max": 200,
        "guide_lr": 0.01,
        "guide_scheduler": "multistep",
        "guide_milestones": [100, 150],
        "guide_gamma": 0.2,
        "guide_augment": False,
        "expert_epochs": 500,
        "expert_lr": 0.01,
        "expert_scheduler": "multistep",
        "expert_milestones": [333, 416],
        "expert_gamma": 0.2,
        "expert_augment": False,
        "expert_use_dsdm_train": True,
    },
    "conv4": {
        "agent": 1,
        "model_id": "convnet4w15",
        "guide_epochs": [20, 50, 100, 200],
        "guide_max": 200,
        "guide_lr": 0.01,
        "guide_scheduler": "multistep",
        "guide_milestones": [100, 150],
        "guide_gamma": 0.2,
        "guide_augment": False,
        "expert_epochs": 500,
        "expert_lr": 0.01,
        "expert_scheduler": "multistep",
        "expert_milestones": [333, 416],
        "expert_gamma": 0.2,
        "expert_augment": False,
        "expert_use_dsdm_train": True,
    },
    "alexnet": {
        "agent": 2,
        "model_id": "alexnet",
        "guide_epochs": [20, 50, 100, 200],
        "guide_max": 200,
        "guide_lr": 0.01,
        "guide_scheduler": "multistep",
        "guide_milestones": [100, 150],
        "guide_gamma": 0.2,
        "guide_augment": False,
        "expert_epochs": 500,
        "expert_lr": 0.01,
        "expert_scheduler": "multistep",
        "expert_milestones": [333, 416],
        "expert_gamma": 0.2,
        "expert_augment": True,
        "expert_use_dsdm_train": False,
    },
    "resnet10_standard": {
        "agent": 3,
        "model_id": "resnet10_standard",
        "guide_epochs": [200],
        "guide_max": 200,
        "guide_lr": 0.1,
        "guide_scheduler": "cosine",
        "guide_milestones": [],
        "guide_gamma": 0.1,
        "guide_augment": True,
        "expert_epochs": 200,
        "expert_lr": 0.1,
        "expert_scheduler": "cosine",
        "expert_milestones": [],
        "expert_gamma": 0.1,
        "expert_augment": True,
        "expert_use_dsdm_train": False,
    },
    "resnet18_standard": {
        "agent": 4,
        "model_id": "resnet18_standard",
        "guide_epochs": [200],
        "guide_max": 200,
        "guide_lr": 0.1,
        "guide_scheduler": "cosine",
        "guide_milestones": [],
        "guide_gamma": 0.1,
        "guide_augment": True,
        "expert_epochs": 200,
        "expert_lr": 0.1,
        "expert_scheduler": "cosine",
        "expert_milestones": [],
        "expert_gamma": 0.1,
        "expert_augment": True,
        "expert_use_dsdm_train": False,
    },
}


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _model_definition(source, name, spec):
    if name == "resnet10_standard":
        model = {
            "family": "resnet_cifar_standard",
            "display_name": "ResNet-10-CIFAR-w1.0",
            "depth": 10,
            "norm_type": "batch",
            "distillation": {"f_idx": "5", "lr_img": 0.01, "niter": 10000},
        }
    elif name == "resnet18_standard":
        model = {
            "family": "resnet_cifar_standard",
            "display_name": "ResNet-18-CIFAR-w1.0",
            "depth": 18,
            "norm_type": "batch",
            "distillation": {"f_idx": "5", "lr_img": 0.01, "niter": 10000},
        }
    else:
        model = copy.deepcopy(source["model_pool"]["models"][spec["model_id"]])
    return model


def _training_config(source, name, spec, smoke):
    cfg = copy.deepcopy(source)
    suffix = "_smoke_v2" if smoke else ""
    run_name = f"cifar100_teacher_quality_{name}_seed0{suffix}"
    cfg["project"].update(
        {
            "stage": "teacher_quality_training_smoke" if smoke else "teacher_quality_training",
            "run_name": run_name,
            "comparability_group": "cifar100_teacher_quality_seed0",
        }
    )
    model = _model_definition(source, name, spec)
    guide_max = 2 if smoke else spec["guide_max"]
    guide_epochs = [1, 2] if smoke else spec["guide_epochs"]
    expert_epochs = 2 if smoke else spec["expert_epochs"]
    model["guide_training"] = {
        "num_models": 1 if smoke else 10,
        "max_epochs": guide_max,
        "snapshot_epochs": guide_epochs,
        "selected_epoch": guide_epochs[-1],
        "lr": spec["guide_lr"],
        "batch_size": 128,
        "augment": spec["guide_augment"],
        "scheduler": spec["guide_scheduler"],
        "scheduler_milestones": [v for v in spec["guide_milestones"] if v < guide_max],
        "scheduler_gamma": spec["guide_gamma"],
    }
    model["expert_training"] = {
        "separate": True,
        "epochs": expert_epochs,
        "lr": spec["expert_lr"],
        "batch_size": 128,
        "augment": spec["expert_augment"],
        "use_dsdm_train": spec["expert_use_dsdm_train"],
        "scheduler": spec["expert_scheduler"],
        "scheduler_milestones": [v for v in spec["expert_milestones"] if v < expert_epochs],
        "scheduler_gamma": spec["expert_gamma"],
        "validation_fraction": 0.1,
        "eval_interval": 1 if smoke else 5,
        "retrain_full": True,
    }
    cfg["model_pool"]["models"][spec["model_id"]] = model
    cfg["agents"]["model_split"][f"agent_{spec['agent']}"] = spec["model_id"]
    cfg["runtime"]["workers"] = 2 if smoke else 4
    return cfg


def _candidate_config(training_cfg, name, spec, epoch, smoke):
    cfg = copy.deepcopy(training_cfg)
    training_run = training_cfg["project"]["run_name"]
    suffix = "_smoke_v2" if smoke else ""
    cfg["project"].update(
        {
            "stage": "teacher_quality_packet_smoke" if smoke else "teacher_quality_packet",
            "run_name": f"cifar100_tq_{name}_guidee{epoch:04d}_seed0_ipc10{suffix}",
        }
    )
    model = cfg["model_pool"]["models"][spec["model_id"]]
    model["guide_training"]["selected_epoch"] = int(epoch)
    model["guide_training"]["source_root"] = f"outputs/{training_run}"
    model["expert_training"]["source_root"] = f"outputs/{training_run}"
    if smoke:
        model["distillation"]["niter"] = 2
        cfg["distillation"]["niter"] = 2
        cfg["distillation"]["evaluate_iterations"] = [1, 2]
        cfg["evaluation"]["epochs"] = 2
    return cfg


def main():
    args = _parse_args()
    with open(SOURCE, "r", encoding="utf-8") as handle:
        source = yaml.safe_load(handle)
    out_dir = ROOT / "configs" / ("teacher_quality_smoke" if args.smoke else "teacher_quality")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in MODELS.items():
        training_cfg = _training_config(source, name, spec, args.smoke)
        training_path = out_dir / f"train_{name}_seed0.yaml"
        with open(training_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(training_cfg, handle, sort_keys=False, allow_unicode=False)
        epochs = [1, 2] if args.smoke else spec["guide_epochs"]
        for epoch in epochs:
            candidate = _candidate_config(training_cfg, name, spec, epoch, args.smoke)
            candidate_path = out_dir / f"packet_{name}_guidee{epoch:04d}_seed0_ipc10.yaml"
            with open(candidate_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(candidate, handle, sort_keys=False, allow_unicode=False)
    print(f"prepared teacher-quality configs in {out_dir}")


if __name__ == "__main__":
    main()
