#!/usr/bin/env python3
"""复用已完成 packets/checkpoints，运行 receiver ablation。"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_data import get_agent_class_split, get_run_dir
from config_adapter import build_dsdm_args_from_config, load_config


def _copy_artifact(src: Path, dst: Path, overwrite: bool) -> bool:
    """复制单个 artifact，返回是否发生复制。"""
    if not src.exists():
        raise FileNotFoundError(f"missing source artifact: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return False
    shutil.copyfile(src, dst)
    return True


def prepare_reuse_artifacts(cfg, args, packet_method: str) -> int:
    """从 reuse.source_run_name 复制 packet 和 expert checkpoint。"""
    reuse = cfg.get("reuse", {})
    source_run = reuse.get("source_run_name")
    if not source_run:
        raise ValueError("config must set reuse.source_run_name")
    source_root = Path(args.output_root) / str(source_run)
    target_root = get_run_dir(args)
    copied = 0
    for agent_id in get_agent_class_split(args):
        if bool(reuse.get("reuse_packets", True)):
            copied += int(
                _copy_artifact(
                    source_root / "agents" / f"agent_{agent_id}" / "packets" / f"{packet_method}_packet.pt",
                    target_root / "agents" / f"agent_{agent_id}" / "packets" / f"{packet_method}_packet.pt",
                    bool(reuse.get("overwrite_packets", False)),
                )
            )
        if bool(reuse.get("reuse_experts", True)):
            copied += int(
                _copy_artifact(
                    source_root / "agents" / f"agent_{agent_id}" / "checkpoints" / "expert_model.pt",
                    target_root / "agents" / f"agent_{agent_id}" / "checkpoints" / "expert_model.pt",
                    bool(reuse.get("overwrite_experts", False)),
                )
            )
    return copied


def run(cmd: list[str]) -> None:
    """执行子命令并直接继承 stdout/stderr。"""
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one receiver ablation config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--packet-method", default="dsdm")
    parser.add_argument("--init-mode", default="expert")
    args_cli = parser.parse_args()

    cfg = load_config(args_cli.config)
    args = build_dsdm_args_from_config(cfg, args_cli.config)
    copied = prepare_reuse_artifacts(cfg, args, args_cli.packet_method)
    print(f"[reuse] copied={copied} run_dir={get_run_dir(args)}", flush=True)

    py = sys.executable
    run([py, "run_social_pipeline.py", "--config", args_cli.config, "--stage", "build_communication", "--packet-method", args_cli.packet_method])
    run([py, "validate_packets.py", "--config", args_cli.config, "--packet-method", args_cli.packet_method])
    run([
        py,
        "run_social_pipeline.py",
        "--config",
        args_cli.config,
        "--stage",
        "train_receivers",
        "--packet-method",
        args_cli.packet_method,
        "--init-mode",
        args_cli.init_mode,
    ])
    run([py, "scripts/update_experiment_registry.py"])


if __name__ == "__main__":
    main()
