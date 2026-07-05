#!/usr/bin/env python3
"""运行 ConvNet/ResNet/ResNetAP 跨架构 receiver-only 超参数排查队列。"""

from __future__ import annotations

import argparse
import copy
import csv
import subprocess
import sys
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "outputs" / "cifar100_4agent_25cls_ipc50_logit" / "config" / "main.yaml"
SOURCE_RUN = "cifar100_4agent_25cls_ipc50_logit"
CONFIG_DIR = ROOT / "configs" / "receiver_followup_tuning"
LOG_DIR = ROOT / "logs" / "receiver_followup_tuning"


MODEL_POOL = {
    "default": "convnet",
    "models": {
        "convnet": {
            "family": "convnet",
            "depth": 3,
            "width": 1.0,
            "norm_type": "instance",
        },
        "resnet": {
            "family": "resnet",
            "depth": 10,
            "width": 1.0,
            "norm_type": "instance",
        },
        "resnet_ap": {
            "family": "resnet_ap",
            "depth": 10,
            "width": 1.0,
            "norm_type": "instance",
        },
    },
}


VARIANTS = [
    {"tag": "ep150_lr0010_fr005_kd050_t2", "epochs": 150, "lr": 0.001, "lambda_fr": 0.05, "lambda_kd": 0.5, "temperature": 2.0},
    {"tag": "ep200_lr0010_fr005_kd050_t2", "epochs": 200, "lr": 0.001, "lambda_fr": 0.05, "lambda_kd": 0.5, "temperature": 2.0},
    {"tag": "ep300_lr0010_fr005_kd050_t2", "epochs": 300, "lr": 0.001, "lambda_fr": 0.05, "lambda_kd": 0.5, "temperature": 2.0},
    {"tag": "ep500_lr0010_fr005_kd050_t2", "epochs": 500, "lr": 0.001, "lambda_fr": 0.05, "lambda_kd": 0.5, "temperature": 2.0},
    {"tag": "ep300_lr0003_fr005_kd050_t2", "epochs": 300, "lr": 0.0003, "lambda_fr": 0.05, "lambda_kd": 0.5, "temperature": 2.0},
    {"tag": "ep300_lr0005_fr005_kd050_t2", "epochs": 300, "lr": 0.0005, "lambda_fr": 0.05, "lambda_kd": 0.5, "temperature": 2.0},
    {"tag": "ep300_lr0020_fr005_kd050_t2", "epochs": 300, "lr": 0.002, "lambda_fr": 0.05, "lambda_kd": 0.5, "temperature": 2.0},
    {"tag": "ep300_lr0010_fr010_kd050_t2", "epochs": 300, "lr": 0.001, "lambda_fr": 0.10, "lambda_kd": 0.5, "temperature": 2.0},
    {"tag": "ep300_lr0010_fr020_kd050_t2", "epochs": 300, "lr": 0.001, "lambda_fr": 0.20, "lambda_kd": 0.5, "temperature": 2.0},
    {"tag": "ep300_lr0010_fr050_kd050_t2", "epochs": 300, "lr": 0.001, "lambda_fr": 0.50, "lambda_kd": 0.5, "temperature": 2.0},
    {"tag": "ep300_lr0010_fr100_kd050_t2", "epochs": 300, "lr": 0.001, "lambda_fr": 1.00, "lambda_kd": 0.5, "temperature": 2.0},
    {"tag": "ep300_lr0010_fr020_kd025_t2", "epochs": 300, "lr": 0.001, "lambda_fr": 0.20, "lambda_kd": 0.25, "temperature": 2.0},
    {"tag": "ep300_lr0010_fr020_kd075_t2", "epochs": 300, "lr": 0.001, "lambda_fr": 0.20, "lambda_kd": 0.75, "temperature": 2.0},
    {"tag": "ep300_lr0010_fr020_kd050_t1", "epochs": 300, "lr": 0.001, "lambda_fr": 0.20, "lambda_kd": 0.5, "temperature": 1.0},
    {"tag": "ep300_lr0010_fr020_kd050_t3", "epochs": 300, "lr": 0.001, "lambda_fr": 0.20, "lambda_kd": 0.5, "temperature": 3.0},
    {"tag": "ep300_lr0010_fr020_kd050_t4", "epochs": 300, "lr": 0.001, "lambda_fr": 0.20, "lambda_kd": 0.5, "temperature": 4.0},
    {"tag": "ep300_lr0010_fr020_kd050_t2_ms", "epochs": 300, "lr": 0.001, "lambda_fr": 0.20, "lambda_kd": 0.5, "temperature": 2.0, "scheduler": "multistep"},
    {"tag": "ep300_lr0005_fr020_kd050_t2_ms", "epochs": 300, "lr": 0.0005, "lambda_fr": 0.20, "lambda_kd": 0.5, "temperature": 2.0, "scheduler": "multistep"},
    {"tag": "ep300_lr0010_fr020_kd050_t3_ms", "epochs": 300, "lr": 0.001, "lambda_fr": 0.20, "lambda_kd": 0.5, "temperature": 3.0, "scheduler": "multistep"},
    {"tag": "ep300_lr0010_fr020_kd050_t2_self15", "epochs": 300, "lr": 0.001, "lambda_fr": 0.20, "lambda_kd": 0.5, "temperature": 2.0, "self_class_weight": 1.5},
    {"tag": "ep300_lr0010_fr020_kd050_t2_self20", "epochs": 300, "lr": 0.001, "lambda_fr": 0.20, "lambda_kd": 0.5, "temperature": 2.0, "self_class_weight": 2.0},
    {"tag": "ep300_lr0010_fr020_kd050_t2_self30", "epochs": 300, "lr": 0.001, "lambda_fr": 0.20, "lambda_kd": 0.5, "temperature": 2.0, "self_class_weight": 3.0},
    {"tag": "ep300_lr0010_fr020_kd050_t3_ms_self15", "epochs": 300, "lr": 0.001, "lambda_fr": 0.20, "lambda_kd": 0.5, "temperature": 3.0, "scheduler": "multistep", "self_class_weight": 1.5},
    {"tag": "ep300_lr0005_fr020_kd050_t3_ms_self20", "epochs": 300, "lr": 0.0005, "lambda_fr": 0.20, "lambda_kd": 0.5, "temperature": 3.0, "scheduler": "multistep", "self_class_weight": 2.0},
    {"tag": "ep300_lr0010_fr020_kd000_nologit", "epochs": 300, "lr": 0.001, "lambda_fr": 0.20, "lambda_kd": 0.0, "temperature": 2.0, "use_logits": False},
    {"tag": "ep300_lr0005_fr020_kd000_nologit_self20", "epochs": 300, "lr": 0.0005, "lambda_fr": 0.20, "lambda_kd": 0.0, "temperature": 2.0, "use_logits": False, "self_class_weight": 2.0},
    {"tag": "ep300_lr0010_fr020_kd050_t2_latefr", "epochs": 300, "lr": 0.001, "lambda_fr": 0.20, "lambda_kd": 0.5, "temperature": 2.0, "lambda_schedule": "late_fr"},
    {"tag": "ep300_lr0010_fr020_kd050_t3_ms_latefr", "epochs": 300, "lr": 0.001, "lambda_fr": 0.20, "lambda_kd": 0.5, "temperature": 3.0, "scheduler": "multistep", "lambda_schedule": "late_fr"},
]


def _load_yaml(path: Path) -> dict:
    """读取 YAML 配置文件。"""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, data: dict) -> None:
    """写入 YAML 配置文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _completed_result_rows(path: Path) -> int:
    """返回 social_results.csv 中已完成的 receiver 行数。"""
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def _apply_receiver_config(receiver_cfg: dict, variant: dict) -> None:
    """把单个 variant 的 receiver 超参数写入配置。"""
    receiver_cfg["epochs"] = variant["epochs"]
    receiver_cfg["lr"] = variant["lr"]
    receiver_cfg["lambda_fr"] = variant["lambda_fr"]
    if "scheduler" in variant:
        receiver_cfg["scheduler"] = variant["scheduler"]
        receiver_cfg["scheduler_gamma"] = variant.get("scheduler_gamma", 0.2)
        receiver_cfg["scheduler_milestones"] = variant.get(
            "scheduler_milestones",
            [int(0.65 * variant["epochs"]), int(0.85 * variant["epochs"])],
        )
    if "self_class_weight" in variant:
        receiver_cfg["self_class_weight"] = variant["self_class_weight"]
    if "lambda_schedule" in variant:
        receiver_cfg["lambda_schedule"] = variant["lambda_schedule"]
        receiver_cfg["lambda_schedule_switch"] = variant.get("lambda_schedule_switch", 0.7)
        receiver_cfg["lambda_fr_late_multiplier"] = variant.get("lambda_fr_late_multiplier", 1.5)
        receiver_cfg["lambda_kd_late_multiplier"] = variant.get("lambda_kd_late_multiplier", 0.7)


def build_config(variant: dict) -> Path:
    """基于旧跨架构 IPC50 输出配置生成 receiver-only 调参配置。"""
    cfg = copy.deepcopy(_load_yaml(BASE_CONFIG))
    run_name = f"cifar100_4agent_25cls_hetero_receiver_tuning_ipc50_{variant['tag']}"
    cfg["project"]["stage"] = "receiver_hetero_tuning"
    cfg["project"]["run_name"] = run_name
    cfg["model_pool"] = copy.deepcopy(MODEL_POOL)
    cfg["reuse"] = {
        "source_run_name": SOURCE_RUN,
        "reuse_packets": True,
        "reuse_experts": True,
        "overwrite_packets": False,
        "overwrite_experts": False,
    }
    receiver_cfg = cfg.setdefault("social_learning", {}).setdefault("receiver", {})
    _apply_receiver_config(receiver_cfg, variant)
    use_logits = bool(variant.get("use_logits", True))
    cfg.setdefault("logits", {})
    cfg["logits"]["enabled"] = use_logits
    cfg["logits"]["lambda_kd"] = variant["lambda_kd"]
    cfg["logits"]["temperature"] = variant["temperature"]
    cfg.setdefault("communication", {})
    cfg["communication"]["use_sender_logits"] = use_logits
    path = CONFIG_DIR / f"main_cifar100_hetero_receiver_tuning_ipc50_{variant['tag']}.yaml"
    _write_yaml(path, cfg)
    return path


def generate_configs() -> list[Path]:
    """生成全部跨架构 receiver 调参配置。"""
    if not BASE_CONFIG.exists():
        raise FileNotFoundError(f"missing base config: {BASE_CONFIG}")
    return [build_config(variant) for variant in VARIANTS]


def run_queue(configs: list[Path], max_jobs: int, python_bin: str) -> None:
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
            completed = _completed_result_rows(result_file)
            if completed >= 4:
                print(f"[skip] completed result {result_file}", flush=True)
                continue
            if 0 < completed < 4:
                print(f"[skip] partial result rows={completed} {result_file}", flush=True)
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


def main() -> None:
    """解析命令行并启动跨架构 receiver 调参队列。"""
    parser = argparse.ArgumentParser(description="Run hetero receiver-only hyperparameter tuning queue.")
    parser.add_argument("--max-jobs", type=int, default=1)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()

    configs = generate_configs()
    print("[configs]", flush=True)
    for cfg in configs:
        print(f"  {cfg.relative_to(ROOT)}", flush=True)
    if args.generate_only:
        return
    run_queue(configs, max(1, args.max_jobs), args.python)
    subprocess.run([args.python, "scripts/update_experiment_registry.py"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
