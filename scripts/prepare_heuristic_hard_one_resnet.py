#!/usr/bin/env python3
"""生成 one-ResNet Heuristic hard-label 配置并复用 expert checkpoint。"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def source_config_path(seed: int, ipc: int) -> Path:
    return ROOT / "configs" / f"main_cifar100_one_resnet_seed{seed}_ipc{ipc}.yaml"


def target_config_path(seed: int, ipc: int) -> Path:
    return ROOT / "configs" / f"main_cifar100_one_resnet_seed{seed}_ipc{ipc}_heuristic_hard.yaml"


def target_run_name(seed: int, ipc: int) -> str:
    return f"cifar100_4agent_25cls_one_resnet_seed{seed}_ipc{ipc}_heuristic_hard"


def build_config(seed: int, ipc: int) -> dict:
    with source_config_path(seed, ipc).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["project"]["stage"] = f"one_resnet_seed{seed}_ipc{ipc}_heuristic_hard"
    cfg["project"]["run_name"] = target_run_name(seed, ipc)
    cfg["selection"] = {
        "enabled": True,
        "methods": ["heuristic"],
        "heuristic": {"strategy": "random_per_class"},
    }
    cfg["communication"]["use_sender_logits"] = False
    cfg["communication"]["use_generalist_logits"] = False
    cfg["logits"]["enabled"] = False
    cfg["logits"]["lambda_kd"] = 0.0
    return cfg


def write_config(seed: int, ipc: int) -> Path:
    path = target_config_path(seed, ipc)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(build_config(seed, ipc), handle, sort_keys=False, allow_unicode=True)
    print(f"[config] wrote {path.relative_to(ROOT)}")
    return path


def _link_or_copy(source: Path, target: Path) -> str:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def prepare_experts(seed: int, ipc: int) -> None:
    target_root = ROOT / "outputs" / target_run_name(seed, ipc)
    for agent_id in range(4):
        candidates = [
            ROOT / "outputs" / f"cifar100_4agent_25cls_one_resnet_seed{seed}_ipc{ipc}"
            / "agents" / f"agent_{agent_id}" / "checkpoints" / "expert_model.pt",
            ROOT / "outputs" / f"cifar100_4agent_25cls_one_resnet_seed{seed}_ipc10"
            / "agents" / f"agent_{agent_id}" / "checkpoints" / "expert_model.pt",
        ]
        source = next((path for path in candidates if path.is_file()), None)
        if source is None:
            raise FileNotFoundError(f"seed={seed} agent={agent_id} 没有可复用 expert_model.pt")
        target = target_root / "agents" / f"agent_{agent_id}" / "checkpoints" / "expert_model.pt"
        mode = _link_or_copy(source, target)
        print(f"[expert] seed={seed} ipc={ipc} agent={agent_id} mode={mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--ipcs", nargs="+", type=int, default=[10, 50])
    args = parser.parse_args()
    for ipc in args.ipcs:
        for seed in args.seeds:
            write_config(seed, ipc)
            prepare_experts(seed, ipc)


if __name__ == "__main__":
    main()
