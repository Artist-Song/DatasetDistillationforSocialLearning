#!/usr/bin/env python3
"""持续监控并补充 hetero4arch IPC=10 的 VGG/AlexNet receiver 调参到指定时间。"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "main_cifar100_hetero4arch_ipc10.yaml"
SOURCE_RUN = "cifar100_4agent_25cls_hetero4arch_ipc10"
CONFIG_DIR = ROOT / "configs" / "hetero4arch_vgg_alex_guardian"
LOG_DIR = ROOT / "logs" / "hetero4arch_ipc10_vgg_alex_guardian"
SUMMARY_DIR = ROOT / "outputs" / "experiment_registry"


EXTRA_VARIANTS = [
    # VGG 第三轮：围绕第一轮/第二轮推断的 forgetting 问题继续增强保守项。
    {"tag": "vgg_g3_fr035_kd060_t3_sched_self20", "receiver": 2, "epochs": 60, "lr": 0.01, "lambda_fr": 0.35, "lambda_kd": 0.60, "temperature": 3.0, "scheduler": "multistep", "self_class_weight": 2.0},
    {"tag": "vgg_g3_fr040_kd060_t3_sched_self20", "receiver": 2, "epochs": 60, "lr": 0.01, "lambda_fr": 0.40, "lambda_kd": 0.60, "temperature": 3.0, "scheduler": "multistep", "self_class_weight": 2.0},
    {"tag": "vgg_g3_fr025_kd050_t3_sched_self30", "receiver": 2, "epochs": 60, "lr": 0.01, "lambda_fr": 0.25, "lambda_kd": 0.50, "temperature": 3.0, "scheduler": "multistep", "self_class_weight": 3.0},
    {"tag": "vgg_g3_fr030_kd050_t3_sched_self30", "receiver": 2, "epochs": 60, "lr": 0.01, "lambda_fr": 0.30, "lambda_kd": 0.50, "temperature": 3.0, "scheduler": "multistep", "self_class_weight": 3.0},
    {"tag": "vgg_g3_fr030_kd040_t3_sched_self30", "receiver": 2, "epochs": 60, "lr": 0.01, "lambda_fr": 0.30, "lambda_kd": 0.40, "temperature": 3.0, "scheduler": "multistep", "self_class_weight": 3.0},
    {"tag": "vgg_g3_fr020_kd040_t4_sched_self30", "receiver": 2, "epochs": 60, "lr": 0.01, "lambda_fr": 0.20, "lambda_kd": 0.40, "temperature": 4.0, "scheduler": "multistep", "self_class_weight": 3.0},
    {"tag": "vgg_g3_ep090_fr025_kd050_t3_sched_self20", "receiver": 2, "epochs": 90, "lr": 0.01, "lambda_fr": 0.25, "lambda_kd": 0.50, "temperature": 3.0, "scheduler": "multistep", "self_class_weight": 2.0},
    {"tag": "vgg_g3_ep090_fr030_kd050_t3_sched_self30", "receiver": 2, "epochs": 90, "lr": 0.01, "lambda_fr": 0.30, "lambda_kd": 0.50, "temperature": 3.0, "scheduler": "multistep", "self_class_weight": 3.0},
    # AlexNet 第三轮：围绕 T=3/scheduler，轻微提高保留项与自类权重。
    {"tag": "alex_g3_fr030_kd060_t3_sched", "receiver": 3, "epochs": 60, "lr": 0.01, "lambda_fr": 0.30, "lambda_kd": 0.60, "temperature": 3.0, "scheduler": "multistep"},
    {"tag": "alex_g3_fr020_kd080_t3_sched", "receiver": 3, "epochs": 60, "lr": 0.01, "lambda_fr": 0.20, "lambda_kd": 0.80, "temperature": 3.0, "scheduler": "multistep"},
    {"tag": "alex_g3_fr020_kd070_t4_sched", "receiver": 3, "epochs": 60, "lr": 0.01, "lambda_fr": 0.20, "lambda_kd": 0.70, "temperature": 4.0, "scheduler": "multistep"},
    {"tag": "alex_g3_fr025_kd070_t3_sched_self15", "receiver": 3, "epochs": 60, "lr": 0.01, "lambda_fr": 0.25, "lambda_kd": 0.70, "temperature": 3.0, "scheduler": "multistep", "self_class_weight": 1.5},
    {"tag": "alex_g3_fr020_kd060_t3_sched_self20", "receiver": 3, "epochs": 60, "lr": 0.01, "lambda_fr": 0.20, "lambda_kd": 0.60, "temperature": 3.0, "scheduler": "multistep", "self_class_weight": 2.0},
    {"tag": "alex_g3_ep090_fr020_kd060_t3_sched", "receiver": 3, "epochs": 90, "lr": 0.01, "lambda_fr": 0.20, "lambda_kd": 0.60, "temperature": 3.0, "scheduler": "multistep"},
]


def _tag_num(value: float | int) -> str:
    """把超参数数值转换成安全 tag 片段。"""
    return str(value).replace(".", "p")


def _append_variant(variant: dict, seen: set[str]) -> None:
    """追加不重复的 guardian 变体。"""
    tag = str(variant["tag"])
    if tag in seen:
        return
    EXTRA_VARIANTS.append(variant)
    seen.add(tag)


def _extend_dense_variants() -> None:
    """补充更密集的 VGG/AlexNet 搜索空间，避免 19 点前 GPU1 空转。"""
    seen = {str(item["tag"]) for item in EXTRA_VARIANTS}

    # VGG: 已观察到 self_weight=3.0 有效降低 forgetting，因此优先围绕 self/fr/kd/lr 加密。
    for lr in [0.01, 0.005]:
        for epochs in [60, 75, 90]:
            for self_weight in [3.0, 4.0, 2.5]:
                for lambda_fr in [0.25, 0.30, 0.35, 0.40]:
                    for lambda_kd in [0.50, 0.40, 0.60, 0.30]:
                        for temperature in [3.0, 4.0]:
                            tag = (
                                f"vgg_dense_ep{epochs}_lr{_tag_num(lr)}_fr{_tag_num(lambda_fr)}"
                                f"_kd{_tag_num(lambda_kd)}_t{_tag_num(temperature)}"
                                f"_self{_tag_num(self_weight)}"
                            )
                            _append_variant(
                                {
                                    "tag": tag,
                                    "receiver": 2,
                                    "epochs": epochs,
                                    "lr": lr,
                                    "lambda_fr": lambda_fr,
                                    "lambda_kd": lambda_kd,
                                    "temperature": temperature,
                                    "scheduler": "multistep",
                                    "self_class_weight": self_weight,
                                },
                                seen,
                            )

    # VGG: 单独检查无 logits / 弱 KD 是否能进一步保护 expert classes。
    for lambda_fr in [0.25, 0.30, 0.35]:
        for self_weight in [3.0, 4.0]:
            tag = f"vgg_dense_nologit_ep75_fr{_tag_num(lambda_fr)}_self{_tag_num(self_weight)}"
            _append_variant(
                {
                    "tag": tag,
                    "receiver": 2,
                    "epochs": 75,
                    "lr": 0.005,
                    "lambda_fr": lambda_fr,
                    "lambda_kd": 0.0,
                    "temperature": 3.0,
                    "scheduler": "multistep",
                    "self_class_weight": self_weight,
                    "use_logits": False,
                },
                seen,
            )

    # AlexNet: 第一轮整体稳定，围绕 T=3/4、KD 较高和轻度 self 权重加密。
    for lr in [0.01, 0.005]:
        for epochs in [60, 75, 90]:
            for lambda_fr in [0.20, 0.25, 0.30]:
                for lambda_kd in [0.60, 0.70, 0.80, 0.50]:
                    for temperature in [3.0, 4.0]:
                        for self_weight in [None, 1.5, 2.0]:
                            tag = (
                                f"alex_dense_ep{epochs}_lr{_tag_num(lr)}_fr{_tag_num(lambda_fr)}"
                                f"_kd{_tag_num(lambda_kd)}_t{_tag_num(temperature)}"
                            )
                            if self_weight is not None:
                                tag += f"_self{_tag_num(self_weight)}"
                            variant = {
                                "tag": tag,
                                "receiver": 3,
                                "epochs": epochs,
                                "lr": lr,
                                "lambda_fr": lambda_fr,
                                "lambda_kd": lambda_kd,
                                "temperature": temperature,
                                "scheduler": "multistep",
                            }
                            if self_weight is not None:
                                variant["self_class_weight"] = self_weight
                            _append_variant(variant, seen)


_extend_dense_variants()


def _load_yaml(path: Path) -> dict:
    """读取 YAML 配置。"""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, data: dict) -> None:
    """写入 YAML 配置。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _completed_rows(result_file: Path) -> int:
    """返回 social_results.csv 中已有结果行数。"""
    if not result_file.exists():
        return 0
    with result_file.open("r", encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def _reset_optional_receiver_keys(receiver_cfg: dict) -> None:
    """清理 receiver 可选调参字段，避免 base config 泄漏旧设置。"""
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
    """生成单 receiver guardian 调参配置。"""
    cfg = copy.deepcopy(_load_yaml(BASE_CONFIG))
    run_name = f"cifar100_4agent_25cls_hetero4arch_ipc10_guardian_{variant['tag']}"
    cfg["project"]["stage"] = "hetero4arch_ipc10_vgg_alex_guardian"
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
    cfg.setdefault("logits", {})
    use_logits = bool(variant.get("use_logits", True))
    cfg["logits"]["enabled"] = use_logits
    cfg["logits"]["lambda_kd"] = float(variant["lambda_kd"])
    cfg["logits"]["temperature"] = float(variant["temperature"])
    cfg.setdefault("communication", {})["use_sender_logits"] = use_logits
    path = CONFIG_DIR / f"main_cifar100_hetero4arch_ipc10_guardian_{variant['tag']}.yaml"
    _write_yaml(path, cfg)
    return path


def _running_patterns() -> list[str]:
    """返回当前 Python 进程命令行列表。"""
    out = subprocess.check_output(["ps", "-eo", "pid,ppid,pgid,stat,etime,cmd"], text=True)
    return [line for line in out.splitlines() if "python" in line]


def is_tune2_active() -> bool:
    """判断第二轮 targeted tuning 是否仍在运行。"""
    return any("run_hetero4arch_ipc10_vgg_alex_tuning.py" in line for line in _running_patterns())


def is_guardian_active_child() -> bool:
    """判断 guardian 是否已有子任务在运行。"""
    lines = _running_patterns()
    return any("run_receiver_ablation.py" in line and "hetero4arch_vgg_alex_guardian" in line for line in lines)


def run_one(cfg: Path, receiver: int, python_bin: str, gpu: int) -> int:
    """运行一个单 receiver 调参任务。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{cfg.stem}_r{receiver}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    cmd = [
        python_bin,
        "scripts/run_receiver_ablation.py",
        "--config",
        str(cfg.relative_to(ROOT)),
        "--packet-method",
        "dsdm",
        "--init-mode",
        "expert",
        "--only-receiver",
        str(receiver),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with log_path.open("w", encoding="utf-8") as f:
        print(f"[guardian start] {' '.join(cmd)} log={log_path.relative_to(ROOT)}", flush=True)
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
        return proc.wait()


def summarize_results() -> None:
    """汇总 VGG/AlexNet 第二轮和 guardian 结果，便于晚间汇报。"""
    rows = []
    patterns = [
        "outputs/cifar100_4agent_25cls_hetero4arch_ipc10_tune2_*/metrics/social_results.csv",
        "outputs/cifar100_4agent_25cls_hetero4arch_ipc10_guardian_*/metrics/social_results.csv",
    ]
    for pattern in patterns:
        for path in sorted(ROOT.glob(pattern)):
            with path.open("r", encoding="utf-8", newline="") as f:
                data = list(csv.DictReader(f))
            if not data:
                continue
            row = data[0]
            rows.append(
                {
                    "run_name": row.get("run_name", ""),
                    "receiver_agent": row.get("receiver_agent", ""),
                    "receiver_model": row.get("receiver_model", ""),
                    "lambda_fr": row.get("lambda_fr", ""),
                    "lambda_kd": row.get("lambda_kd", ""),
                    "kd_temperature": row.get("kd_temperature", ""),
                    "acc_global_after": row.get("acc_global_after", ""),
                    "acc_new_after": row.get("acc_new_after", ""),
                    "acc_expert_after": row.get("acc_expert_after", ""),
                    "forgetting": row.get("forgetting", ""),
                    "source_file": str(path.relative_to(ROOT)),
                }
            )
    rows.sort(key=lambda r: (int(r["receiver_agent"] or -1), -float(r["acc_global_after"] or 0)))
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUMMARY_DIR / "hetero4arch_ipc10_vgg_alex_guardian_summary.csv"
    fieldnames = [
        "run_name",
        "receiver_agent",
        "receiver_model",
        "lambda_fr",
        "lambda_kd",
        "kd_temperature",
        "acc_global_after",
        "acc_new_after",
        "acc_expert_after",
        "forgetting",
        "source_file",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[guardian summary] saved {out_path.relative_to(ROOT)} rows={len(rows)}", flush=True)


def main() -> None:
    """持续运行调参任务到 deadline。"""
    parser = argparse.ArgumentParser(description="Run VGG/AlexNet tuning until deadline.")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--until", default="19:00", help="本机时间 HH:MM 截止启动新任务")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    now = datetime.now()
    hour, minute = [int(x) for x in args.until.split(":", 1)]
    deadline = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if deadline <= now:
        deadline += timedelta(days=1)
    configs = [(build_config(variant, args.gpu), int(variant["receiver"])) for variant in EXTRA_VARIANTS]
    print(f"[guardian] deadline={deadline.isoformat(timespec='seconds')} configs={len(configs)}", flush=True)

    pending = list(configs)
    while datetime.now() < deadline:
        summarize_results()
        if is_tune2_active():
            print("[guardian] tune2 active; waiting", flush=True)
            time.sleep(args.poll_seconds)
            continue
        if is_guardian_active_child():
            print("[guardian] child active; waiting", flush=True)
            time.sleep(args.poll_seconds)
            continue
        while pending:
            cfg, receiver = pending.pop(0)
            run_name = _load_yaml(cfg)["project"]["run_name"]
            result_file = ROOT / "outputs" / run_name / "metrics" / "social_results.csv"
            if _completed_rows(result_file) >= 1:
                print(f"[guardian skip] completed {result_file.relative_to(ROOT)}", flush=True)
                continue
            if datetime.now() >= deadline:
                break
            status = run_one(cfg, receiver, args.python, args.gpu)
            print(f"[guardian done] {cfg.name} receiver={receiver} status={status}", flush=True)
            break
        if not pending:
            print("[guardian] no pending configs; monitoring until deadline", flush=True)
        time.sleep(args.poll_seconds)

    summarize_results()
    subprocess.run([args.python, "scripts/update_experiment_registry.py"], cwd=ROOT, check=True)
    print("[guardian] finished deadline reached", flush=True)


if __name__ == "__main__":
    main()
