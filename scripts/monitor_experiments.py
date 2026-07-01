#!/usr/bin/env python3
"""持续监测调参与主实验进度，并定期保存状态快照。"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_cmd(cmd: list[str], timeout: int = 30) -> str:
    """执行只读状态命令，失败时返回错误文本。"""
    try:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return result.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"[monitor-error] {' '.join(cmd)}: {exc}"


def to_float(value: str) -> float:
    """安全转换浮点数。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def summarize_csv(path: Path) -> str:
    """汇总一个 social_results.csv 的 4-agent 均值。"""
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return ""
    def mean(key: str) -> float:
        values = [to_float(row.get(key, "")) for row in rows]
        values = [value for value in values if not math.isnan(value)]
        return sum(values) / len(values) if values else math.nan

    first = rows[0]
    return (
        f"{first.get('run_name','')} rows={len(rows)} ipc={first.get('ipc','')} "
        f"global={mean('acc_global_after'):.3f} new={mean('acc_new_after'):.3f} "
        f"expert={mean('acc_expert_after'):.3f} forget={mean('forgetting'):.3f}"
    )


def latest_results() -> str:
    """生成最新 tuning/refinement 结果摘要。"""
    patterns = [
        "outputs/*joint_tuning*/metrics/social_results.csv",
        "outputs/*refine_tuning*/metrics/social_results.csv",
        "outputs/cifar100_4agent_25cls_conv_family_ipc50/metrics/social_results.csv",
    ]
    paths = [Path(path) for pattern in patterns for path in glob.glob(str(ROOT / pattern))]
    paths = sorted(set(paths), key=lambda path: path.stat().st_mtime if path.exists() else 0, reverse=True)
    lines = []
    for path in paths[:12]:
        summary = summarize_csv(path)
        if summary:
            lines.append(summary)
    return "\n".join(lines)


def snapshot(log_path: Path, python_bin: str) -> None:
    """写入一次监测快照。"""
    now = datetime.now().isoformat(timespec="seconds")
    subprocess.run(
        [python_bin, "scripts/run_receiver_refinement_tuning.py", "--generate-only"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    sections = [
        ("time", now),
        (
            "processes",
            run_cmd(
                [
                    "pgrep",
                    "-af",
                    "run_receiver_refinement_tuning|run_receiver_ablation|receiver_refinement_tuning|main_cifar100_conv_family_ipc50|run_social_pipeline",
                ]
            ),
        ),
        ("best_hyperparams", run_cmd(["cat", "outputs/experiment_registry/best_hyperparams.csv"])),
        ("latest_results", latest_results()),
        (
            "refinement_queue_tail",
            run_cmd(["bash", "-lc", "ls -t logs/receiver_refinement_tuning/queue_*.log 2>/dev/null | head -1 | xargs -r tail -60"]),
        ),
        (
            "conv_family_ipc50_tail",
            run_cmd(["bash", "-lc", "ls -t logs/conv_family_ipc50/run_*.log 2>/dev/null | head -1 | xargs -r tail -60"]),
        ),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        for title, body in sections:
            f.write(f"[{title}]\n{body}\n\n")


def main() -> None:
    """运行持续监测循环。"""
    parser = argparse.ArgumentParser(description="Monitor running experiments.")
    parser.add_argument("--duration-hours", type=float, default=8.0)
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--log", default="")
    args = parser.parse_args()

    start = datetime.now()
    deadline = start + timedelta(hours=args.duration_hours)
    log_path = Path(args.log) if args.log else ROOT / "logs" / "experiment_monitor" / f"monitor_{start:%Y%m%d_%H%M%S}.log"
    print(f"[monitor] start={start.isoformat(timespec='seconds')} deadline={deadline.isoformat(timespec='seconds')} log={log_path}", flush=True)

    while datetime.now() < deadline:
        snapshot(log_path, args.python)
        time.sleep(max(30, int(args.interval_seconds)))
    snapshot(log_path, args.python)
    print(f"[monitor] done log={log_path}", flush=True)


if __name__ == "__main__":
    main()
