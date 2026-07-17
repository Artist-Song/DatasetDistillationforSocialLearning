#!/usr/bin/env python3
"""运行 hetero4arch IPC=10 receiver-only 超参数队列。"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "main_cifar100_hetero4arch_ipc10.yaml"
SOURCE_RUN = "cifar100_4agent_25cls_hetero4arch_ipc10"
CONFIG_DIR = ROOT / "configs" / "hetero4arch_receiver_tuning"
LOG_DIR = ROOT / "logs" / "hetero4arch_ipc10_tuning"


VARIANTS = [
    {"tag": "ep050_fr020_kd050_t2", "epochs": 50, "lr": 0.01, "lambda_fr": 0.20, "lambda_kd": 0.50, "temperature": 2.0},
    {"tag": "ep055_fr020_kd060_t2", "epochs": 55, "lr": 0.01, "lambda_fr": 0.20, "lambda_kd": 0.60, "temperature": 2.0},
    {"tag": "ep060_fr015_kd050_t2", "epochs": 60, "lr": 0.01, "lambda_fr": 0.15, "lambda_kd": 0.50, "temperature": 2.0},
    {"tag": "ep060_fr020_kd050_t2", "epochs": 60, "lr": 0.01, "lambda_fr": 0.20, "lambda_kd": 0.50, "temperature": 2.0},
    {"tag": "ep060_fr025_kd050_t2", "epochs": 60, "lr": 0.01, "lambda_fr": 0.25, "lambda_kd": 0.50, "temperature": 2.0},
    {"tag": "ep060_fr020_kd060_t1", "epochs": 60, "lr": 0.01, "lambda_fr": 0.20, "lambda_kd": 0.60, "temperature": 1.0},
    {"tag": "ep060_fr020_kd060_t2", "epochs": 60, "lr": 0.01, "lambda_fr": 0.20, "lambda_kd": 0.60, "temperature": 2.0},
    {"tag": "ep060_fr020_kd060_t3", "epochs": 60, "lr": 0.01, "lambda_fr": 0.20, "lambda_kd": 0.60, "temperature": 3.0},
    {"tag": "ep060_fr020_kd060_t4", "epochs": 60, "lr": 0.01, "lambda_fr": 0.20, "lambda_kd": 0.60, "temperature": 4.0},
    {"tag": "ep060_fr025_kd060_t2", "epochs": 60, "lr": 0.01, "lambda_fr": 0.25, "lambda_kd": 0.60, "temperature": 2.0},
    {"tag": "ep060_fr030_kd060_t2", "epochs": 60, "lr": 0.01, "lambda_fr": 0.30, "lambda_kd": 0.60, "temperature": 2.0},
    {"tag": "ep060_fr020_kd070_t3", "epochs": 60, "lr": 0.01, "lambda_fr": 0.20, "lambda_kd": 0.70, "temperature": 3.0},
    {"tag": "ep065_fr020_kd060_t2", "epochs": 65, "lr": 0.01, "lambda_fr": 0.20, "lambda_kd": 0.60, "temperature": 2.0},
    {"tag": "ep060_fr020_kd060_t2_sched", "epochs": 60, "lr": 0.01, "lambda_fr": 0.20, "lambda_kd": 0.60, "temperature": 2.0, "scheduler": "multistep"},
    {"tag": "ep060_fr020_kd060_t3_sched", "epochs": 60, "lr": 0.01, "lambda_fr": 0.20, "lambda_kd": 0.60, "temperature": 3.0, "scheduler": "multistep"},
    {"tag": "ep060_fr020_kd060_t2_self15", "epochs": 60, "lr": 0.01, "lambda_fr": 0.20, "lambda_kd": 0.60, "temperature": 2.0, "self_class_weight": 1.5},
    {"tag": "ep060_fr020_kd060_t2_self20", "epochs": 60, "lr": 0.01, "lambda_fr": 0.20, "lambda_kd": 0.60, "temperature": 2.0, "self_class_weight": 2.0},
    {"tag": "ep060_fr020_kd060_t2_sched_self15", "epochs": 60, "lr": 0.01, "lambda_fr": 0.20, "lambda_kd": 0.60, "temperature": 2.0, "scheduler": "multistep", "self_class_weight": 1.5},
    {"tag": "ep060_fr020_kd060_t2_latesched", "epochs": 60, "lr": 0.01, "lambda_fr": 0.20, "lambda_kd": 0.60, "temperature": 2.0, "lambda_schedule": "late_fr"},
    {"tag": "ep060_fr020_kd060_t2_sched_latesched", "epochs": 60, "lr": 0.01, "lambda_fr": 0.20, "lambda_kd": 0.60, "temperature": 2.0, "scheduler": "multistep", "lambda_schedule": "late_fr"},
]


def _load_yaml(path: Path) -> dict:
    """读取 YAML 配置。"""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, data: dict) -> None:
    """写入 YAML 配置。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _completed_rows(path: Path) -> int:
    """统计已完成 receiver 结果行数。"""
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def _reset_optional_receiver_keys(receiver_cfg: dict) -> None:
    """清理 base config 中可能遗留的可选调参字段。"""
    for key in [
        "scheduler",
        "scheduler_gamma",
        "scheduler_milestones",
        "self_class_weight",
        "lambda_schedule",
        "lambda_schedule_switch",
        "lambda_fr_late_multiplier",
        "lambda_kd_late_multiplier",
    ]:
        receiver_cfg.pop(key, None)


def build_config(variant: dict, gpu: int) -> Path:
    """基于 hetero4arch IPC10 主配置生成单个调参配置。"""
    cfg = copy.deepcopy(_load_yaml(BASE_CONFIG))
    run_name = f"cifar100_4agent_25cls_hetero4arch_ipc10_tune_{variant['tag']}"
    cfg["project"]["stage"] = "hetero4arch_ipc10_receiver_tuning"
    cfg["project"]["run_name"] = run_name
    cfg["reuse"] = {
        "source_run_name": SOURCE_RUN,
        "reuse_packets": True,
        "reuse_experts": True,
        "overwrite_packets": False,
        "overwrite_experts": False,
    }
    cfg.setdefault("runtime", {})["gpu_id"] = int(gpu)
    receiver_cfg = cfg.setdefault("social_learning", {}).setdefault("receiver", {})
    _reset_optional_receiver_keys(receiver_cfg)
    receiver_cfg["epochs"] = int(variant["epochs"])
    receiver_cfg["lr"] = float(variant["lr"])
    receiver_cfg["lambda_fr"] = float(variant["lambda_fr"])
    if "scheduler" in variant:
        receiver_cfg["scheduler"] = variant["scheduler"]
        receiver_cfg["scheduler_gamma"] = float(variant.get("scheduler_gamma", 0.2))
        receiver_cfg["scheduler_milestones"] = variant.get(
            "scheduler_milestones",
            [int(0.65 * variant["epochs"]), int(0.85 * variant["epochs"])],
        )
    if "self_class_weight" in variant:
        receiver_cfg["self_class_weight"] = float(variant["self_class_weight"])
    if "lambda_schedule" in variant:
        receiver_cfg["lambda_schedule"] = variant["lambda_schedule"]
        receiver_cfg["lambda_schedule_switch"] = float(variant.get("lambda_schedule_switch", 0.7))
        receiver_cfg["lambda_fr_late_multiplier"] = float(variant.get("lambda_fr_late_multiplier", 1.5))
        receiver_cfg["lambda_kd_late_multiplier"] = float(variant.get("lambda_kd_late_multiplier", 0.7))
    cfg.setdefault("logits", {})
    cfg["logits"]["enabled"] = True
    cfg["logits"]["lambda_kd"] = float(variant["lambda_kd"])
    cfg["logits"]["temperature"] = float(variant["temperature"])
    cfg.setdefault("communication", {})["use_sender_logits"] = True
    path = CONFIG_DIR / f"main_cifar100_hetero4arch_ipc10_tune_{variant['tag']}.yaml"
    _write_yaml(path, cfg)
    return path


def generate_configs(gpu: int) -> list[Path]:
    """生成全部 hetero4arch IPC10 调参配置。"""
    return [build_config(variant, gpu) for variant in VARIANTS]


def run_queue(configs: list[Path], max_jobs: int, python_bin: str, gpu: int) -> None:
    """按最大并发数运行 receiver-only 调参队列。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    pending = list(configs)
    running: list[tuple[subprocess.Popen, Path, Path]] = []
    while pending or running:
        active: list[tuple[subprocess.Popen, Path, Path]] = []
        for proc, cfg, log_path in running:
            status = proc.poll()
            if status is None:
                active.append((proc, cfg, log_path))
            else:
                print(f"[done] {cfg.name} status={status} log={log_path}", flush=True)
        running = active

        while pending and len(running) < max_jobs:
            cfg = pending.pop(0)
            run_name = _load_yaml(cfg)["project"]["run_name"]
            result_file = ROOT / "outputs" / run_name / "metrics" / "social_results.csv"
            completed = _completed_rows(result_file)
            if completed >= 4:
                print(f"[skip] completed {result_file.relative_to(ROOT)}", flush=True)
                continue
            if 0 < completed < 4:
                print(f"[skip] partial rows={completed} {result_file.relative_to(ROOT)}", flush=True)
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
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            print(f"[start] {' '.join(cmd)} log={log_path.relative_to(ROOT)}", flush=True)
            running.append((subprocess.Popen(cmd, cwd=ROOT, stdout=log_f, stderr=subprocess.STDOUT, env=env), cfg, log_path))

        if pending or running:
            time.sleep(20)


def main() -> None:
    """解析参数并启动 hetero4arch IPC10 调参队列。"""
    parser = argparse.ArgumentParser(description="Run hetero4arch IPC10 receiver tuning.")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--max-jobs", type=int, default=1)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()

    configs = generate_configs(args.gpu)
    print("[configs]", flush=True)
    for cfg in configs:
        print(f"  {cfg.relative_to(ROOT)}", flush=True)
    if args.generate_only:
        return
    run_queue(configs, max(1, args.max_jobs), args.python, args.gpu)
    subprocess.run([args.python, "scripts/update_experiment_registry.py"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
