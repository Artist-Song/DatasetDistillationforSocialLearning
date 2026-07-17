#!/usr/bin/env python3
"""监控 hetero4arch IPC=50 的 VGG/AlexNet 蒸馏中间准确率。"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IPC50_LOG_DIR = ROOT / "logs" / "hetero4arch_ipc50"
OUT_PATH = ROOT / "outputs" / "experiment_registry" / "hetero4arch_ipc50_mid_accuracy_monitor.csv"


CONV_REFERENCE_BEST = {
    0: 71.8,
    1: 70.7,
    2: 70.2,
    3: 75.5,
}


def _latest_log() -> Path | None:
    """返回最新 IPC50 队列日志。"""
    logs = sorted(IPC50_LOG_DIR.glob("queue_*.log"), key=lambda path: path.stat().st_mtime)
    return logs[-1] if logs else None


def _parse_current_sender(lines: list[str]) -> tuple[int | None, str]:
    """从日志中解析当前 distill sender agent 和模型名。"""
    current_agent = None
    current_model = ""
    for line in lines:
        if "[distill_packets] agent=" not in line:
            continue
        m = re.search(r"agent=(\d+)\s+model=([^\s]+)", line)
        if m:
            current_agent = int(m.group(1))
            current_model = m.group(2)
    return current_agent, current_model


def _parse_latest_progress(lines: list[str]) -> tuple[int | None, int | None, float | None, float | None]:
    """解析最新 DSDM condense 进度和 best accuracy。"""
    latest = (None, None, None, None)
    for line in lines:
        if "[DSDM condense]" not in line:
            continue
        m = re.search(r"(\d+)/(\d+)\s+([0-9.]+)%.*best=([0-9.]+)", line)
        if m:
            latest = (int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4)))
    return latest


def _process_alive() -> bool:
    """检查 IPC50 队列是否仍在运行。"""
    out = subprocess.check_output(["ps", "-eo", "cmd"], text=True)
    return "scripts/run_hetero4arch_ipc50_queue.py" in out


def _write_header_if_needed(path: Path) -> None:
    """创建监控 CSV 表头。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "time",
                "agent",
                "model",
                "iteration",
                "total_iterations",
                "progress_pct",
                "best_acc",
                "conv_reference_best",
                "gap_to_reference",
                "status",
                "log_path",
            ],
        )
        writer.writeheader()


def _append_row(row: dict) -> None:
    """追加一行监控结果。"""
    _write_header_if_needed(OUT_PATH)
    with OUT_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "time",
                "agent",
                "model",
                "iteration",
                "total_iterations",
                "progress_pct",
                "best_acc",
                "conv_reference_best",
                "gap_to_reference",
                "status",
                "log_path",
            ],
        )
        writer.writerow(row)


def monitor_once() -> dict:
    """执行一次 IPC50 中间准确率监控。"""
    log_path = _latest_log()
    if log_path is None:
        row = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "agent": "",
            "model": "",
            "iteration": "",
            "total_iterations": "",
            "progress_pct": "",
            "best_acc": "",
            "conv_reference_best": "",
            "gap_to_reference": "",
            "status": "no_log",
            "log_path": "",
        }
        _append_row(row)
        return row
    lines = log_path.read_text(errors="ignore").replace("\r", "\n").splitlines()
    agent, model = _parse_current_sender(lines)
    iteration, total, progress, best = _parse_latest_progress(lines)
    reference = CONV_REFERENCE_BEST.get(agent) if agent is not None else None
    gap = reference - best if reference is not None and best is not None else None
    status = "ok"
    if model in {"vgg", "alexnet"} and progress is not None and best is not None and reference is not None:
        if progress >= 20.0 and gap >= 8.0:
            status = "watch_adjust"
        elif progress >= 10.0 and gap >= 10.0:
            status = "watch"
    row = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "agent": "" if agent is None else agent,
        "model": model,
        "iteration": "" if iteration is None else iteration,
        "total_iterations": "" if total is None else total,
        "progress_pct": "" if progress is None else f"{progress:.2f}",
        "best_acc": "" if best is None else f"{best:.4f}",
        "conv_reference_best": "" if reference is None else f"{reference:.4f}",
        "gap_to_reference": "" if gap is None else f"{gap:.4f}",
        "status": status,
        "log_path": str(log_path.relative_to(ROOT)),
    }
    _append_row(row)
    return row


def main() -> None:
    """按固定间隔监控到截止时间。"""
    parser = argparse.ArgumentParser(description="Monitor IPC50 VGG/AlexNet distill mid accuracy.")
    parser.add_argument("--until", default="19:00")
    parser.add_argument("--interval-seconds", type=int, default=300)
    args = parser.parse_args()

    now = datetime.now()
    hour, minute = [int(x) for x in args.until.split(":", 1)]
    deadline = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if deadline <= now:
        deadline += timedelta(days=1)

    print(f"[monitor] deadline={deadline.isoformat(timespec='seconds')} out={OUT_PATH.relative_to(ROOT)}", flush=True)
    while datetime.now() < deadline and _process_alive():
        row = monitor_once()
        print("[monitor]", row, flush=True)
        time.sleep(max(30, args.interval_seconds))
    row = monitor_once()
    print("[monitor final]", row, flush=True)


if __name__ == "__main__":
    main()
