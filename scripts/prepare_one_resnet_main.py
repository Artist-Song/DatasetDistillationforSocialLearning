#!/usr/bin/env python3
"""生成 one-ResNet 主实验配置，并复用已有 artifact。"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

CLASS_SPLIT = {
    "agent_0": list(range(0, 25)),
    "agent_1": list(range(25, 50)),
    "agent_2": list(range(50, 75)),
    "agent_3": list(range(75, 100)),
}

SEED_MODELS = {
    0: ["alexnet", "convnet3w1", "resnet18", "convnet4w15"],
    1: ["convnet4w15", "alexnet", "resnet10", "convnet3w1"],
    2: ["alexnet", "resnet18", "convnet3w1", "convnet4w15"],
}

MODEL_POOL = {
    "default": "convnet3w1",
    "models": {
        "convnet3w1": {
            "family": "convnet",
            "depth": 3,
            "width": 1.0,
            "norm_type": "instance",
            "distillation": {"f_idx": "2", "lr_img": 0.1, "niter": 10000},
        },
        "convnet4w15": {
            "family": "convnet",
            "depth": 4,
            "width": 1.5,
            "norm_type": "instance",
            "distillation": {"f_idx": "3", "lr_img": 0.1, "niter": 10000},
        },
        "alexnet": {
            "family": "alexnet",
            "norm_type": "batch",
            "distillation": {"f_idx": "7", "lr_img": 0.005, "niter": 10000},
        },
        "resnet10": {
            "family": "resnet",
            "display_name": "ResNet-10-CIFAR-w0.5",
            "depth": 10,
            "norm_type": "batch",
            "distillation": {"f_idx": "5", "lr_img": 0.01, "niter": 10000},
        },
        "resnet18": {
            "family": "resnet",
            "display_name": "ResNet-18-CIFAR-w0.5",
            "depth": 18,
            "norm_type": "batch",
            "distillation": {"f_idx": "5", "lr_img": 0.01, "niter": 10000},
        },
    },
}

# exact packet 只在 model/class/ipc/feature 语义一致时复用；其余 agent 重新蒸馏。
EXACT_PACKET_SOURCE = {
    (0, 10, 0): ("cifar100_4agent_25cls_novgg_r18_backbone_penult_ipc10", 0),
    (0, 10, 1): ("cifar100_4agent_25cls_novgg_r18_backbone_penult_ipc10", 1),
    (0, 10, 2): ("cifar100_4agent_25cls_novgg_r18_backbone_penult_ipc10", 2),
    (0, 10, 3): ("cifar100_4agent_25cls_novgg_r18_backbone_penult_ipc10", 3),
    (0, 50, 0): ("cifar100_4agent_25cls_hetero4arch_ipc50", 3),
    (0, 50, 1): ("cifar100_4agent_25cls_hetero4arch_ipc50", 0),
    (0, 50, 3): ("cifar100_4agent_25cls_hetero4arch_ipc50", 1),
    (2, 10, 0): ("cifar100_4agent_25cls_hetero4arch_ipc10", 3),
    (2, 10, 3): ("cifar100_4agent_25cls_hetero4arch_ipc10", 1),
    (2, 50, 0): ("cifar100_4agent_25cls_hetero4arch_ipc50", 3),
    (2, 50, 3): ("cifar100_4agent_25cls_hetero4arch_ipc50", 1),
}

# guide/expert 可以从旧实验复用；需要重新蒸馏的 agent 仍只复用 guide pool。
GUIDE_SOURCE = {
    (0, 0): ("cifar100_4agent_25cls_novgg_r18_backbone_penult_ipc10", 0),
    (0, 1): ("cifar100_4agent_25cls_novgg_r18_backbone_penult_ipc10", 1),
    (0, 2): ("cifar100_4agent_25cls_novgg_r18_backbone_penult_ipc10", 2),
    (0, 3): ("cifar100_4agent_25cls_novgg_r18_backbone_penult_ipc10", 3),
    (1, 0): ("cifar100_4agent_25cls_hetero6arch_ipc10_seed1", 0),
    (1, 1): ("cifar100_4agent_25cls_hetero6arch_ipc10_seed1", 1),
    (1, 2): ("cifar100_4agent_25cls_novgg_r10_backbone_penult_ipc10", 2),
    (2, 0): ("cifar100_4agent_25cls_hetero4arch_ipc10", 3),
    (2, 1): ("cifar100_4agent_25cls_hetero6arch_ipc10_seed2", 1),
    (2, 2): ("cifar100_4agent_25cls_hetero6arch_ipc10_seed2", 2),
    (2, 3): ("cifar100_4agent_25cls_hetero4arch_ipc10", 1),
}


def run_name(seed: int, ipc: int, full_real: bool = False) -> str:
    """返回 one-ResNet 主实验 run_name。"""
    base = f"cifar100_4agent_25cls_one_resnet_seed{seed}_ipc{ipc}"
    return f"{base}_fullreal" if full_real else base


def config_path(seed: int, ipc: int, full_real: bool = False) -> Path:
    """返回生成配置路径。"""
    suffix = "_fullreal" if full_real else ""
    return ROOT / "configs" / f"main_cifar100_one_resnet_seed{seed}_ipc{ipc}{suffix}.yaml"


def build_config(seed: int, ipc: int, full_real: bool = False) -> dict:
    """构造 one-ResNet 主实验 YAML 配置。"""
    model_split = {f"agent_{idx}": model for idx, model in enumerate(SEED_MODELS[seed])}
    if ipc == 50:
        receiver = {
            "epochs": 225,
            "lr": 0.001,
            "lambda_fr": 0.05,
            "self_real_per_class": 20,
            "scheduler": "none",
            "scheduler_gamma": 0.2,
            "scheduler_milestones": [150, 187],
        }
        lambda_kd = 0.5
    else:
        receiver = {
            "epochs": 60,
            "lr": 0.01,
            "lambda_fr": 0.2,
            "self_real_per_class": 20,
            "scheduler": "multistep",
            "scheduler_gamma": 0.2,
            "scheduler_milestones": [39, 51],
        }
        lambda_kd = 0.6

    return {
        "project": {
            "name": "social_packet_learning",
            "stage": f"one_resnet_seed{seed}_ipc{ipc}",
            "output_root": "./outputs",
            "run_name": run_name(seed, ipc, full_real=full_real),
        },
        "dataset": {"name": "cifar100", "data_dir": "./data", "num_classes": 100, "image_size": 32},
        "model_pool": MODEL_POOL,
        "distillation": {
            "method": "dsdm",
            "ipc": int(ipc),
            "factor": 2,
            "init": "mix",
            "decode_type": "single",
            "aug_type": "color_crop_cutout",
            "match": "semantic",
            "metric": "mse",
            "f_idx": "2",
            "niter": 10000,
            # Preserve the dense evaluation protocol used by the ongoing three-seed main run.
            "evaluate_iterations": [],
            "lr_img": 0.1,
            "mom_img": 0.5,
            "batch_real": 256,
            "batch_syn_max": 256,
            "smooth_iter": 2000,
            "cov_weight": 50.0,
            "h_p_weight": 0.2,
            "smooth_factor": 0.99,
            "pretrained_model_number": 10,
            "pretrained_epochs": 20,
        },
        "selection": {"enabled": True, "methods": ["heuristic", "full_real"], "heuristic": {"strategy": "random_per_class"}},
        "packet": {"enabled": True, "format": "hard_label_images"},
        "communication": {
            "enabled": True,
            "protocol": "none",
            "mode": "direct",
            "use_sender_logits": not full_real,
            "use_generalist_logits": False,
        },
        "agents": {
            "num_agents": 4,
            "num_classes": 100,
            "class_split": CLASS_SPLIT,
            "model_split": model_split,
        },
        "social_learning": {"receiver": receiver},
        "logits": {"enabled": not full_real, "lambda_kd": lambda_kd, "temperature": 2.0},
        "evaluation": {"enabled": True, "epochs": 1500, "batch_size": 64, "repeat": 1},
        "runtime": {"seed": int(seed), "workers": 8, "device": "cuda", "gpu_id": 0},
    }


def write_configs(seeds: list[int], ipcs: list[int]) -> None:
    """写出主方法和 full-real 配置文件。"""
    for seed in seeds:
        for ipc in ipcs:
            for full_real in (False, True):
                path = config_path(seed, ipc, full_real=full_real)
                path.write_text(
                    yaml.safe_dump(build_config(seed, ipc, full_real=full_real), sort_keys=False, allow_unicode=True),
                    encoding="utf-8",
                )
                print(f"[config] wrote {path.relative_to(ROOT)}")


def _copy_file(src: Path, dst: Path, overwrite: bool = False) -> bool:
    """复制文件，默认不覆盖已有目标。"""
    if not src.exists():
        raise FileNotFoundError(f"missing source artifact: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return False
    shutil.copy2(src, dst)
    return True


def _copy_guides(src_agent_dir: Path, dst_agent_dir: Path, overwrite: bool = False) -> list[str]:
    """复制 expert 和 guide pool。"""
    copied = []
    for name in ["expert_model.pt", "expert_selection.json"]:
        src = src_agent_dir / "checkpoints" / name
        if src.exists() and _copy_file(src, dst_agent_dir / "checkpoints" / name, overwrite=overwrite):
            copied.append(str(dst_agent_dir / "checkpoints" / name))
    for idx in range(10):
        src = src_agent_dir / "checkpoints" / f"guide_model_{idx}.pt"
        if src.exists() and _copy_file(src, dst_agent_dir / "checkpoints" / f"guide_model_{idx}.pt", overwrite=overwrite):
            copied.append(str(dst_agent_dir / "checkpoints" / f"guide_model_{idx}.pt"))
    return copied


def prepare_artifacts(seed: int, ipc: int, full_real: bool = False, overwrite_packets: bool = False) -> dict:
    """准备某个 seed/ipc 的可复用 artifact，并返回缺失训练计划。"""
    target_run = run_name(seed, ipc, full_real=full_real)
    target_root = ROOT / "outputs" / target_run
    target_root.mkdir(parents=True, exist_ok=True)
    provenance = {
        "seed": seed,
        "ipc": ipc,
        "run_name": target_run,
        "model_split": {f"agent_{idx}": model for idx, model in enumerate(SEED_MODELS[seed])},
        "copied": [],
        "need_train_experts": [],
        "need_distill_packets": [],
    }

    for agent_id in range(4):
        dst_agent = target_root / "agents" / f"agent_{agent_id}"
        (dst_agent / "checkpoints").mkdir(parents=True, exist_ok=True)
        (dst_agent / "packets").mkdir(parents=True, exist_ok=True)

        guide_key = (seed, agent_id)
        guide_source = GUIDE_SOURCE.get(guide_key)
        if guide_source is None and seed == 1 and agent_id == 3:
            guide_source = ("cifar100_4agent_25cls_one_resnet_seed1_ipc10", 3)
        if guide_source is not None and guide_source[0] != target_run:
            src_run, src_agent = guide_source
            src_agent_dir = ROOT / "outputs" / src_run / "agents" / f"agent_{src_agent}"
            provenance["copied"].extend(_copy_guides(src_agent_dir, dst_agent))

        expert_path = dst_agent / "checkpoints" / "expert_model.pt"
        guides_ok = all((dst_agent / "checkpoints" / f"guide_model_{idx}.pt").exists() for idx in range(10))
        if not expert_path.exists() or not guides_ok:
            provenance["need_train_experts"].append(agent_id)

        if not full_real:
            packet_key = (seed, ipc, agent_id)
            packet_path = dst_agent / "packets" / "dsdm_packet.pt"
            if packet_key in EXACT_PACKET_SOURCE:
                src_run, src_agent = EXACT_PACKET_SOURCE[packet_key]
                src_packet = ROOT / "outputs" / src_run / "agents" / f"agent_{src_agent}" / "packets" / "dsdm_packet.pt"
                if _copy_file(src_packet, packet_path, overwrite=overwrite_packets):
                    provenance["copied"].append(str(packet_path))
            if not packet_path.exists():
                provenance["need_distill_packets"].append(agent_id)

    prov_path = target_root / "metrics" / "one_resnet_artifact_provenance.json"
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    prov_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[prepare] {target_run} train={provenance['need_train_experts']} distill={provenance['need_distill_packets']}")
    return provenance


def parse_args() -> argparse.Namespace:
    """解析命令行。"""
    parser = argparse.ArgumentParser(description="Prepare one-ResNet main experiment configs/artifacts.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--ipcs", nargs="+", type=int, default=[10, 50])
    parser.add_argument("--generate-configs", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--full-real", action="store_true")
    parser.add_argument("--overwrite-packets", action="store_true")
    return parser.parse_args()


def main() -> None:
    """脚本入口。"""
    args = parse_args()
    if args.generate_configs:
        write_configs(args.seeds, args.ipcs)
    if args.prepare:
        for seed in args.seeds:
            for ipc in args.ipcs:
                prepare_artifacts(seed, ipc, full_real=False, overwrite_packets=args.overwrite_packets)
                if args.full_real:
                    prepare_artifacts(seed, ipc, full_real=True, overwrite_packets=args.overwrite_packets)


if __name__ == "__main__":
    main()
