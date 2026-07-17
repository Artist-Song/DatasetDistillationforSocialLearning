#!/usr/bin/env python3
"""组装 no-VGG backbone-guided IPC=10 复用实验所需 artifact。"""

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_data import get_agent_dir
from config_adapter import build_dsdm_args_from_config, load_config
from social_output_manager import prepare_social_output_dirs


def _copy_file(src, dst, overwrite=False):
    """复制单个文件，默认保留已存在目标文件。"""
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"missing source artifact: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return False
    shutil.copyfile(src, dst)
    return True


def _copy_guides(src_ckpt_dir, dst_ckpt_dir, overwrite=False):
    """复制 guide model pool 与选择元数据。"""
    copied = []
    for model_idx in range(10):
        src = Path(src_ckpt_dir) / f"guide_model_{model_idx}.pt"
        dst = Path(dst_ckpt_dir) / f"guide_model_{model_idx}.pt"
        if _copy_file(src, dst, overwrite=overwrite):
            copied.append(str(dst))
    meta_src = Path(src_ckpt_dir) / "expert_selection.json"
    if meta_src.exists():
        meta_dst = Path(dst_ckpt_dir) / "expert_selection.json"
        if _copy_file(meta_src, meta_dst, overwrite=overwrite):
            copied.append(str(meta_dst))
    return copied


def _source_root(output_root, run_name):
    """返回复用来源 run 根目录。"""
    return Path(output_root) / run_name


def _artifact_plan(variant):
    """返回不同 no-VGG 变体的 artifact 复用计划。"""
    common = {
        0: {
            "source_run": "cifar100_4agent_25cls_hetero4arch_ipc10",
            "source_agent": 3,
            "copy_expert": True,
            "copy_guides": False,
            "copy_dsdm_packet": True,
            "note": "AlexNet classes 0-24 from early hetero4arch",
        },
        1: {
            "source_run": "cifar100_4agent_25cls_hetero4arch_ipc10",
            "source_agent": 0,
            "copy_expert": True,
            "copy_guides": False,
            "copy_dsdm_packet": True,
            "note": "ConvNet3-w1.0 classes 25-49 from early hetero4arch",
        },
        3: {
            "source_run": "cifar100_4agent_25cls_hetero4arch_ipc10",
            "source_agent": 1,
            "copy_expert": True,
            "copy_guides": False,
            "copy_dsdm_packet": True,
            "note": "ConvNet4-w1.5 classes 75-99 from early hetero4arch",
        },
    }
    if variant == "r10":
        common[2] = {
            "source_run": "cifar100_4agent_25cls_hetero6arch_ipc10_seed1",
            "source_agent": 2,
            "copy_expert": True,
            "copy_guides": True,
            "copy_dsdm_packet": False,
            "note": "ResNet10 classes 50-74 guide pool; packet will be re-distilled with f_idx=5",
        }
    elif variant == "r18":
        common[2] = {
            "source_run": "",
            "source_agent": 2,
            "copy_expert": False,
            "copy_guides": False,
            "copy_dsdm_packet": False,
            "note": "ResNet18 classes 50-74 will train guides then distill with f_idx=5",
        }
    else:
        raise ValueError(f"unsupported variant: {variant}")
    return common


def assemble(config_path, variant, overwrite=False):
    """按固定槽位复制旧 packet/expert 到当前 no-VGG run。"""
    cfg = load_config(config_path)
    args = build_dsdm_args_from_config(cfg, config_path=config_path)
    prepare_social_output_dirs(args)

    copied = []
    provenance = {
        "config": str(config_path),
        "target_run": args.run_name,
        "variant": variant,
        "artifacts": [],
    }
    output_root = Path(args.output_root)
    for target_agent, item in _artifact_plan(variant).items():
        target_dir = get_agent_dir(args, target_agent)
        artifact = {
            "target_agent": int(target_agent),
            "note": item["note"],
            "source_run": item["source_run"],
            "source_agent": item["source_agent"],
            "copied": [],
        }
        if item["source_run"]:
            src_root = _source_root(output_root, item["source_run"])
            src_agent_dir = src_root / "agents" / f"agent_{item['source_agent']}"
            if item["copy_expert"]:
                src = src_agent_dir / "checkpoints" / "expert_model.pt"
                dst = target_dir / "checkpoints" / "expert_model.pt"
                if _copy_file(src, dst, overwrite=overwrite):
                    copied.append(str(dst))
                    artifact["copied"].append(str(dst))
                meta_src = src_agent_dir / "checkpoints" / "expert_selection.json"
                if meta_src.exists():
                    meta_dst = target_dir / "checkpoints" / "expert_selection.json"
                    if _copy_file(meta_src, meta_dst, overwrite=overwrite):
                        copied.append(str(meta_dst))
                        artifact["copied"].append(str(meta_dst))
            if item["copy_guides"]:
                guide_copies = _copy_guides(
                    src_agent_dir / "checkpoints",
                    target_dir / "checkpoints",
                    overwrite=overwrite,
                )
                copied.extend(guide_copies)
                artifact["copied"].extend(guide_copies)
            if item["copy_dsdm_packet"]:
                src = src_agent_dir / "packets" / "dsdm_packet.pt"
                dst = target_dir / "packets" / "dsdm_packet.pt"
                if _copy_file(src, dst, overwrite=overwrite):
                    copied.append(str(dst))
                    artifact["copied"].append(str(dst))
        provenance["artifacts"].append(artifact)

    prov_path = output_root / args.run_name / "metrics" / f"novgg_{variant}_artifact_provenance.json"
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    prov_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[assemble] target_run={args.run_name} variant={variant} copied={len(copied)}")
    print(f"[assemble] provenance={prov_path}")
    for path in copied:
        print(f"  copied {path}")


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Assemble no-VGG backbone-guided IPC10 artifacts")
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", required=True, choices=["r10", "r18"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    """脚本入口。"""
    args = parse_args()
    assemble(Path(args.config), args.variant, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
