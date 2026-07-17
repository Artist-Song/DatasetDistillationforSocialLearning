#!/usr/bin/env python3
"""准备 compact/standard ResNet-18 的 FAST IPC=50 receiver 诊断。"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = ROOT / "configs" / "main_cifar100_one_resnet_seed0_ipc50_fast.yaml"
SOURCE_RUN = ROOT / "outputs" / "cifar100_4agent_25cls_one_resnet_seed0_ipc50_fast"

VARIANTS = {
    "compact": {
        "model_id": "resnet18",
        "run_name": "cifar100_resnet18_compact_recipe_diag_seed0_ipc50_fast",
    },
    "standard": {
        "model_id": "resnet18_standard",
        "run_name": "cifar100_resnet18_standard_recipe_diag_seed0_ipc50_fast",
    },
}


def _link_exact(source: Path, target: Path) -> None:
    """为只读复用 artifact 建立硬链接，已存在时要求内容一致。"""
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if os.path.samefile(source, target):
            return
        if source.read_bytes() != target.read_bytes():
            raise RuntimeError(f"目标已存在且内容不同: {target}")
        target.unlink()
    os.link(source, target)


def _build_config(source: dict, variant: str) -> dict:
    cfg = copy.deepcopy(source)
    spec = VARIANTS[variant]
    cfg["project"]["stage"] = f"resnet18_{variant}_recipe_diagnostic"
    cfg["project"]["run_name"] = spec["run_name"]
    cfg["agents"]["model_split"]["agent_2"] = spec["model_id"]
    cfg["model_pool"]["models"]["resnet18"]["display_name"] = "ResNet-18-CIFAR-w0.5"
    cfg["model_pool"]["models"]["resnet18_standard"] = {
        "family": "resnet_cifar_standard",
        "display_name": "ResNet-18-CIFAR-w1.0",
        "depth": 18,
        "norm_type": "batch",
        "expert_training": {
            "num_models": 1,
            "epochs": 200,
            "lr": 0.1,
            "batch_size": 128,
            "augment": True,
            "scheduler": "multistep",
            "scheduler_milestones": [100, 150],
            "scheduler_gamma": 0.1,
        },
        "distillation": {"f_idx": "5", "lr_img": 0.01, "niter": 10000},
    }
    cfg["social_learning"]["receiver"] = {
        "epochs": 200,
        "lr": 0.01,
        "lambda_fr": 0.05,
        "self_real_per_class": 20,
        "augment": True,
        "freeze_bn_stats": True,
        "scheduler": "multistep",
        "scheduler_milestones": [100, 150],
        "scheduler_gamma": 0.1,
    }
    cfg["communication"]["use_sender_logits"] = False
    cfg["logits"]["enabled"] = False
    cfg["logits"]["lambda_kd"] = 0.0
    return cfg


def _prepare_variant(source_cfg: dict, variant: str) -> Path:
    spec = VARIANTS[variant]
    config_path = ROOT / "configs" / f"diagnostic_resnet18_{variant}_seed0_ipc50_fast.yaml"
    config = _build_config(source_cfg, variant)
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=False)

    target_run = ROOT / "outputs" / spec["run_name"]
    for agent_id in range(4):
        _link_exact(
            SOURCE_RUN / "agents" / f"agent_{agent_id}" / "packets" / "fast_packet.pt",
            target_run / "agents" / f"agent_{agent_id}" / "packets" / "fast_packet.pt",
        )
    if variant == "compact":
        _link_exact(
            SOURCE_RUN / "agents" / "agent_2" / "checkpoints" / "expert_model.pt",
            target_run / "agents" / "agent_2" / "checkpoints" / "expert_model.pt",
        )
    return config_path


def main() -> None:
    with open(SOURCE_CONFIG, "r", encoding="utf-8") as handle:
        source_cfg = yaml.safe_load(handle)
    for variant in VARIANTS:
        path = _prepare_variant(source_cfg, variant)
        print(f"prepared {variant}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
