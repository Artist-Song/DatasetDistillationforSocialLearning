#!/usr/bin/env python3
"""生成 FAST one-ResNet 配置，并复用对应 expert checkpoint。"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
FAST_COMMIT = "6a218fcfdc93838634921399b0de6a36cdd29756"


def source_config_path(seed: int, ipc: int) -> Path:
    return ROOT / "configs" / f"main_cifar100_one_resnet_seed{seed}_ipc{ipc}.yaml"


def fast_config_path(seed: int, ipc: int) -> Path:
    return ROOT / "configs" / f"main_cifar100_one_resnet_seed{seed}_ipc{ipc}_fast.yaml"


def fast_run_name(seed: int, ipc: int) -> str:
    return f"cifar100_4agent_25cls_one_resnet_seed{seed}_ipc{ipc}_fast"


def build_fast_config(seed: int, ipc: int) -> dict:
    with source_config_path(seed, ipc).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["project"]["stage"] = f"one_resnet_seed{seed}_ipc{ipc}_fast"
    cfg["project"]["run_name"] = fast_run_name(seed, ipc)
    cfg["selection"] = {
        "enabled": True,
        "methods": ["fast"],
        "fast": {
            "strategy": "official_pixels_per_class_minmax",
            "repo_path": "external_baselines/repos/FAST",
            "cache_root": "external_baselines/outputs/fast_cache",
            "commit": FAST_COMMIT,
            "seed": 0,
        },
    }
    cfg["communication"]["use_sender_logits"] = False
    cfg["communication"]["use_generalist_logits"] = False
    cfg["logits"]["enabled"] = False
    return cfg


def write_config(seed: int, ipc: int) -> Path:
    path = fast_config_path(seed, ipc)
    cfg = build_fast_config(seed, ipc)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)
    print(f"[config] wrote {path.relative_to(ROOT)}")
    return path


def _link_or_copy(src: Path, dst: Path) -> str:
    if not src.exists():
        raise FileNotFoundError(f"missing expert checkpoint: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "existing"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def prepare_experts(seed: int, ipc: int) -> None:
    target_root = ROOT / "outputs" / fast_run_name(seed, ipc)
    for agent_id in range(4):
        candidates = [
            ROOT
            / "outputs"
            / f"cifar100_4agent_25cls_one_resnet_seed{seed}_ipc{ipc}"
            / "agents"
            / f"agent_{agent_id}"
            / "checkpoints"
            / "expert_model.pt",
            ROOT
            / "outputs"
            / f"cifar100_4agent_25cls_one_resnet_seed{seed}_ipc10"
            / "agents"
            / f"agent_{agent_id}"
            / "checkpoints"
            / "expert_model.pt",
        ]
        source = next((path for path in candidates if path.exists()), None)
        if source is None:
            raise FileNotFoundError(f"seed={seed} agent={agent_id} 没有可复用 expert_model.pt")
        target = target_root / "agents" / f"agent_{agent_id}" / "checkpoints" / "expert_model.pt"
        mode = _link_or_copy(source, target)
        print(f"[expert] seed={seed} ipc={ipc} agent={agent_id} mode={mode} source={source.relative_to(ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--ipcs", nargs="+", type=int, default=[10, 50])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for ipc in args.ipcs:
        for seed in args.seeds:
            write_config(seed, ipc)
            prepare_experts(seed, ipc)


if __name__ == "__main__":
    main()
