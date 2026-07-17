#!/usr/bin/env python3
"""运行 hetero4arch IPC=50 主实验队列。"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "main_cifar100_hetero4arch_ipc50.yaml"
CONV_SOURCE_ROOT = ROOT / "outputs" / "cifar100_4agent_25cls_conv_family_ipc50_lastfeat" / "agents"
TARGET_ROOT = ROOT / "outputs" / "cifar100_4agent_25cls_hetero4arch_ipc50" / "agents"

COPY_MAP = {
    "agent_1": "agent_0",
    "agent_3": "agent_1",
}

COPY_ITEMS = [
    "checkpoints/expert_model.pt",
    "checkpoints/dsdm_pretrained",
    "checkpoints/args.txt",
    "packets/dsdm_packet.pt",
    "synthetic/data_best.pt",
]


def copy_item(src: Path, dst: Path, overwrite: bool) -> None:
    """复制单个文件或目录，默认保留已有产物。"""
    if not src.exists():
        print(f"[copy skip] missing source: {src}", flush=True)
        return
    if dst.exists() and not overwrite:
        print(f"[copy skip] existing target: {dst}", flush=True)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    print(f"[copy] {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}", flush=True)


def prepare_conv_reuse(overwrite: bool) -> None:
    """复用 ConvNet IPC50 last-feature sender 产物。"""
    for src_agent, dst_agent in COPY_MAP.items():
        for item in COPY_ITEMS:
            copy_item(CONV_SOURCE_ROOT / src_agent / item, TARGET_ROOT / dst_agent / item, overwrite)


def run(cmd: list[str], gpu: int) -> None:
    """在指定 GPU 上执行一个 pipeline 子命令。"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"[run gpu={gpu}] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def main() -> None:
    """解析参数并顺序运行 IPC50 实验。"""
    parser = argparse.ArgumentParser(description="Run hetero4arch IPC50 queue.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--overwrite-conv-copy", action="store_true")
    args = parser.parse_args()

    py = args.python
    cfg = str(CONFIG.relative_to(ROOT))
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[start] hetero4arch IPC50 queue at {started}", flush=True)
    prepare_conv_reuse(args.overwrite_conv_copy)

    for agent_id in [2, 3]:
        run([py, "run_social_pipeline.py", "--config", cfg, "--stage", "train_experts", "--only-agent", str(agent_id), "--resume"], args.gpu)
        run([py, "run_social_pipeline.py", "--config", cfg, "--stage", "distill_packets", "--packet-method", "dsdm", "--only-agent", str(agent_id)], args.gpu)

    for method in ["dsdm", "heuristic"]:
        if method == "heuristic":
            run([py, "run_social_pipeline.py", "--config", cfg, "--stage", "build_selection_packets", "--packet-method", method], args.gpu)
        run([py, "run_social_pipeline.py", "--config", cfg, "--stage", "attach_logits", "--packet-method", method], args.gpu)
        run([py, "run_social_pipeline.py", "--config", cfg, "--stage", "build_communication", "--packet-method", method], args.gpu)
        run([py, "validate_packets.py", "--config", cfg, "--packet-method", method], args.gpu)
        run([py, "run_social_pipeline.py", "--config", cfg, "--stage", "train_receivers", "--packet-method", method, "--init-mode", "expert"], args.gpu)

    run([py, "scripts/update_experiment_registry.py"], args.gpu)
    print("[done] hetero4arch IPC50 queue", flush=True)


if __name__ == "__main__":
    main()
