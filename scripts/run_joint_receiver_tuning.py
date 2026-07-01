#!/usr/bin/env python3
"""生成并运行 IPC=10/50 receiver 联合调参任务。"""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


IPC10_BASE = ROOT / "configs" / "main_cifar100_conv_family_ipc10.yaml"
IPC50_BASE = ROOT / "configs" / "main_cifar100_allconv.yaml"
CONFIG_DIR = ROOT / "configs" / "receiver_joint_tuning"
LOG_DIR = ROOT / "logs" / "receiver_joint_tuning"


VARIANTS = [
    {
        "tag": "ipc10_ep075_fr020_kd050",
        "base": IPC10_BASE,
        "source_run": "cifar100_4agent_25cls_conv_family_ipc10",
        "epochs": 75,
        "lr": 0.01,
        "lambda_fr": 0.2,
        "lambda_kd": 0.5,
        "temperature": 2.0,
    },
    {
        "tag": "ipc10_ep100_fr020_kd040",
        "base": IPC10_BASE,
        "source_run": "cifar100_4agent_25cls_conv_family_ipc10",
        "epochs": 100,
        "lr": 0.01,
        "lambda_fr": 0.2,
        "lambda_kd": 0.4,
        "temperature": 2.0,
    },
    {
        "tag": "ipc10_ep100_fr020_kd060",
        "base": IPC10_BASE,
        "source_run": "cifar100_4agent_25cls_conv_family_ipc10",
        "epochs": 100,
        "lr": 0.01,
        "lambda_fr": 0.2,
        "lambda_kd": 0.6,
        "temperature": 2.0,
    },
    {
        "tag": "ipc10_ep125_fr020_kd050",
        "base": IPC10_BASE,
        "source_run": "cifar100_4agent_25cls_conv_family_ipc10",
        "epochs": 125,
        "lr": 0.01,
        "lambda_fr": 0.2,
        "lambda_kd": 0.5,
        "temperature": 2.0,
    },
    {
        "tag": "ipc50_ep300_fr005_kd050",
        "base": IPC50_BASE,
        "source_run": "cifar100_4agent_25cls_ipc50_allconvnet",
        "epochs": 300,
        "lr": 0.001,
        "lambda_fr": 0.05,
        "lambda_kd": 0.5,
        "temperature": 2.0,
    },
    {
        "tag": "ipc50_ep700_fr005_kd050",
        "base": IPC50_BASE,
        "source_run": "cifar100_4agent_25cls_ipc50_allconvnet",
        "epochs": 700,
        "lr": 0.001,
        "lambda_fr": 0.05,
        "lambda_kd": 0.5,
        "temperature": 2.0,
    },
    {
        "tag": "ipc50_ep500_fr003_kd050",
        "base": IPC50_BASE,
        "source_run": "cifar100_4agent_25cls_ipc50_allconvnet",
        "epochs": 500,
        "lr": 0.001,
        "lambda_fr": 0.03,
        "lambda_kd": 0.5,
        "temperature": 2.0,
    },
    {
        "tag": "ipc50_ep500_fr0075_kd050",
        "base": IPC50_BASE,
        "source_run": "cifar100_4agent_25cls_ipc50_allconvnet",
        "epochs": 500,
        "lr": 0.001,
        "lambda_fr": 0.075,
        "lambda_kd": 0.5,
        "temperature": 2.0,
    },
]


def _load_yaml(path: Path) -> dict:
    """读取 YAML 配置。"""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, data: dict) -> None:
    """写入生成后的 YAML 配置。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def build_config(variant: dict) -> Path:
    """基于已有主配置生成独立调参配置。"""
    cfg = copy.deepcopy(_load_yaml(variant["base"]))
    run_name = f"cifar100_4agent_25cls_joint_tuning_{variant['tag']}"
    cfg["project"]["stage"] = "joint_receiver_tuning"
    cfg["project"]["run_name"] = run_name
    cfg["reuse"] = {
        "source_run_name": variant["source_run"],
        "reuse_packets": True,
        "reuse_experts": True,
        "overwrite_packets": False,
        "overwrite_experts": False,
    }
    cfg.setdefault("social_learning", {}).setdefault("receiver", {})
    cfg["social_learning"]["receiver"]["epochs"] = variant["epochs"]
    cfg["social_learning"]["receiver"]["lr"] = variant["lr"]
    cfg["social_learning"]["receiver"]["lambda_fr"] = variant["lambda_fr"]
    cfg.setdefault("logits", {})
    cfg["logits"]["enabled"] = True
    cfg["logits"]["lambda_kd"] = variant["lambda_kd"]
    cfg["logits"]["temperature"] = variant["temperature"]
    path = CONFIG_DIR / f"main_cifar100_joint_tuning_{variant['tag']}.yaml"
    _write_yaml(path, cfg)
    return path


def run_queue(configs: list[Path], max_jobs: int, python_bin: str) -> None:
    """按最大并发数运行 receiver 调参队列。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    running: list[tuple[subprocess.Popen, Path, Path]] = []
    pending = list(configs)

    while pending or running:
        still_running: list[tuple[subprocess.Popen, Path, Path]] = []
        for proc, cfg, log_path in running:
            status = proc.poll()
            if status is None:
                still_running.append((proc, cfg, log_path))
                continue
            print(f"[done] {cfg.name} status={status} log={log_path}", flush=True)
        running = still_running

        while pending and len(running) < max_jobs:
            cfg = pending.pop(0)
            run_name = _load_yaml(cfg)["project"]["run_name"]
            result_file = ROOT / "outputs" / run_name / "metrics" / "social_results.csv"
            if result_file.exists():
                print(f"[skip] existing result {result_file}", flush=True)
                continue
            log_path = LOG_DIR / f"{cfg.stem}_{time.strftime('%Y%m%d_%H%M%S')}.log"
            cmd = [
                python_bin,
                "scripts/run_receiver_ablation.py",
                "--config",
                str(cfg.relative_to(ROOT)),
                "--packet-method",
                "dsdm",
                "--init-mode",
                "expert",
            ]
            log_f = log_path.open("w", encoding="utf-8")
            print(f"[start] {' '.join(cmd)} log={log_path}", flush=True)
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log_f, stderr=subprocess.STDOUT)
            running.append((proc, cfg, log_path))

        if pending or running:
            time.sleep(20)

    subprocess.run([python_bin, "scripts/update_experiment_registry.py"], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run joint IPC=10/50 receiver tuning.")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=2)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    configs = [build_config(variant) for variant in VARIANTS]
    print("[configs]", *[str(path.relative_to(ROOT)) for path in configs], sep="\n  ", flush=True)
    if not args.generate_only:
        run_queue(configs, max(1, args.max_jobs), args.python)


if __name__ == "__main__":
    main()
