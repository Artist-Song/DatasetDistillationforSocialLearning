#!/usr/bin/env python3
"""运行第二批 receiver-only follow-up 调参并保存最佳超参数。"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import subprocess
import sys
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
IPC10_BASE = ROOT / "configs" / "main_cifar100_conv_family_ipc10.yaml"
IPC50_BASE = ROOT / "configs" / "main_cifar100_allconv.yaml"
CONFIG_DIR = ROOT / "configs" / "receiver_followup_tuning"
LOG_DIR = ROOT / "logs" / "receiver_followup_tuning"
SUMMARY_DIR = ROOT / "outputs" / "experiment_registry"


VARIANTS = [
    {
        "tag": "ipc10_ep050_fr020_kd050",
        "base": IPC10_BASE,
        "source_run": "cifar100_4agent_25cls_conv_family_ipc10",
        "epochs": 50,
        "lr": 0.01,
        "lambda_fr": 0.2,
        "lambda_kd": 0.5,
        "temperature": 2.0,
    },
    {
        "tag": "ipc10_ep060_fr015_kd050",
        "base": IPC10_BASE,
        "source_run": "cifar100_4agent_25cls_conv_family_ipc10",
        "epochs": 60,
        "lr": 0.01,
        "lambda_fr": 0.15,
        "lambda_kd": 0.5,
        "temperature": 2.0,
    },
    {
        "tag": "ipc10_ep060_fr025_kd050",
        "base": IPC10_BASE,
        "source_run": "cifar100_4agent_25cls_conv_family_ipc10",
        "epochs": 60,
        "lr": 0.01,
        "lambda_fr": 0.25,
        "lambda_kd": 0.5,
        "temperature": 2.0,
    },
    {
        "tag": "ipc10_ep060_fr020_kd060",
        "base": IPC10_BASE,
        "source_run": "cifar100_4agent_25cls_conv_family_ipc10",
        "epochs": 60,
        "lr": 0.01,
        "lambda_fr": 0.2,
        "lambda_kd": 0.6,
        "temperature": 2.0,
    },
    {
        "tag": "ipc50_ep225_fr005_kd050",
        "base": IPC50_BASE,
        "source_run": "cifar100_4agent_25cls_ipc50_allconvnet",
        "epochs": 225,
        "lr": 0.001,
        "lambda_fr": 0.05,
        "lambda_kd": 0.5,
        "temperature": 2.0,
    },
    {
        "tag": "ipc50_ep275_fr005_kd050",
        "base": IPC50_BASE,
        "source_run": "cifar100_4agent_25cls_ipc50_allconvnet",
        "epochs": 275,
        "lr": 0.001,
        "lambda_fr": 0.05,
        "lambda_kd": 0.5,
        "temperature": 2.0,
    },
    {
        "tag": "ipc50_ep250_fr004_kd050",
        "base": IPC50_BASE,
        "source_run": "cifar100_4agent_25cls_ipc50_allconvnet",
        "epochs": 250,
        "lr": 0.001,
        "lambda_fr": 0.04,
        "lambda_kd": 0.5,
        "temperature": 2.0,
    },
    {
        "tag": "ipc50_ep250_fr006_kd050",
        "base": IPC50_BASE,
        "source_run": "cifar100_4agent_25cls_ipc50_allconvnet",
        "epochs": 250,
        "lr": 0.001,
        "lambda_fr": 0.06,
        "lambda_kd": 0.5,
        "temperature": 2.0,
    },
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


def _to_float(value: str) -> float:
    """把 CSV 字段安全转换为 float。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def build_config(variant: dict) -> Path:
    """基于已有配置生成 follow-up run 配置。"""
    cfg = copy.deepcopy(_load_yaml(variant["base"]))
    run_name = f"cifar100_4agent_25cls_refine_tuning_followup_{variant['tag']}"
    cfg["project"]["stage"] = "receiver_followup_tuning"
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
    path = CONFIG_DIR / f"main_cifar100_refine_tuning_followup_{variant['tag']}.yaml"
    _write_yaml(path, cfg)
    return path


def _mean(rows: list[dict], key: str) -> float:
    """计算有效数值均值。"""
    values = [_to_float(row.get(key, "")) for row in rows]
    values = [value for value in values if not math.isnan(value)]
    return sum(values) / len(values) if values else math.nan


def summarize_best_hyperparams() -> None:
    """汇总所有已完成 receiver tuning 结果并保存最佳超参数。"""
    import glob

    paths = [
        ROOT / path
        for pattern in [
            "outputs/*joint_tuning*/metrics/social_results.csv",
            "outputs/*refine_tuning*/metrics/social_results.csv",
        ]
        for path in glob.glob(pattern)
    ]
    baselines = [
        ROOT / "outputs/cifar100_4agent_25cls_conv_family_tuning_A_ep100_fr020_kd050/metrics/social_results.csv",
        ROOT / "outputs/cifar100_4agent_25cls_ipc50_allconvnet/metrics/social_results.csv",
    ]
    paths.extend(path for path in baselines if path.exists())

    rows_out = []
    for path in sorted(set(paths)):
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != 4:
            continue
        first = rows[0]
        rows_out.append(
            {
                "run_name": first.get("run_name", ""),
                "ipc": first.get("ipc", ""),
                "method": first.get("method", ""),
                "lambda_fr": first.get("lambda_fr", ""),
                "lambda_kd": first.get("lambda_kd", ""),
                "kd_temperature": first.get("kd_temperature", ""),
                "mean_acc_global_after": f"{_mean(rows, 'acc_global_after'):.6f}",
                "mean_acc_new_after": f"{_mean(rows, 'acc_new_after'):.6f}",
                "mean_acc_expert_after": f"{_mean(rows, 'acc_expert_after'):.6f}",
                "mean_forgetting": f"{_mean(rows, 'forgetting'):.6f}",
                "source_file": str(path.relative_to(ROOT)),
            }
        )

    rows_out.sort(key=lambda row: (row["ipc"], -float(row["mean_acc_global_after"])))
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "ipc",
        "method",
        "lambda_fr",
        "lambda_kd",
        "kd_temperature",
        "mean_acc_global_after",
        "mean_acc_new_after",
        "mean_acc_expert_after",
        "mean_forgetting",
        "source_file",
    ]
    summary_csv = SUMMARY_DIR / "best_hyperparams_candidates.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    best_rows = {}
    for row in rows_out:
        if row["method"] != "DSDM_LOGIT":
            continue
        best_rows.setdefault(row["ipc"], row)
    best_csv = SUMMARY_DIR / "best_hyperparams.csv"
    with best_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(best_rows[key] for key in sorted(best_rows, key=lambda x: int(x)))
    print(f"[summary] saved {summary_csv}", flush=True)
    print(f"[summary] saved {best_csv}", flush=True)


def run_queue(configs: list[Path], max_jobs: int, python_bin: str) -> None:
    """按最大并发数运行 follow-up 队列。"""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run receiver follow-up tuning queue.")
    parser.add_argument("--max-jobs", type=int, default=1)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()

    configs = [build_config(variant) for variant in VARIANTS]
    print("[configs]", flush=True)
    for cfg in configs:
        print(f"  {cfg.relative_to(ROOT)}", flush=True)
    if args.generate_only:
        summarize_best_hyperparams()
        return
    run_queue(configs, max(1, args.max_jobs), args.python)
    summarize_best_hyperparams()


if __name__ == "__main__":
    main()
