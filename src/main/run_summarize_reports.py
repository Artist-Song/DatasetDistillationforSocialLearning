"""
Summarize compare reports into CSV and Markdown tables.
"""

import argparse
import csv
import json
from pathlib import Path

from src.utils.config import load_yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="实验配置文件路径")
    parser.add_argument(
        "--sort-by",
        type=str,
        default="social_average_general_accuracy",
        choices=[
            "social_average_general_accuracy",
            "delta_average_general_accuracy",
            "social_average_expert_accuracy",
            "delta_average_expert_accuracy",
        ],
        help="表格排序指标",
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        default=True,
        help="按降序排序",
    )
    return parser.parse_args()


def load_compare_report(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)

    social_cfg = report.get("social", {})
    summary = report["summary"]
    return {
        "report_name": path.stem,
        "packet_kd_mode": social_cfg.get("packet_kd_mode", ""),
        "lambda_retain": social_cfg.get("lambda_retain", ""),
        "lambda_packet": social_cfg.get("lambda_packet", ""),
        "lambda_kd": social_cfg.get("lambda_kd", ""),
        "local_average_expert_accuracy": summary["local_average_expert_accuracy"],
        "social_average_expert_accuracy": summary["social_average_expert_accuracy"],
        "delta_average_expert_accuracy": summary["delta_average_expert_accuracy"],
        "local_average_general_accuracy": summary["local_average_general_accuracy"],
        "social_average_general_accuracy": summary["social_average_general_accuracy"],
        "delta_average_general_accuracy": summary["delta_average_general_accuracy"],
        "path": str(path),
    }


def fmt(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_csv(rows, path: Path, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, path: Path, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(fmt(row[name]) for name in fieldnames) + " |\n")


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    compare_dir = Path(cfg["output"]["root"]) / "reports" / "compare"
    report_paths = sorted(compare_dir.glob("compare_*.json"))
    if not report_paths:
        raise FileNotFoundError(f"未找到 compare 报告: {compare_dir}")

    rows = [load_compare_report(path) for path in report_paths]
    rows.sort(key=lambda row: row[args.sort_by], reverse=args.descending)

    fieldnames = [
        "packet_kd_mode",
        "lambda_retain",
        "lambda_packet",
        "lambda_kd",
        "social_average_general_accuracy",
        "delta_average_general_accuracy",
        "social_average_expert_accuracy",
        "delta_average_expert_accuracy",
        "report_name",
    ]

    summary_dir = Path(cfg["output"]["root"]) / "reports" / "summary"
    csv_path = summary_dir / "compare_summary.csv"
    md_path = summary_dir / "compare_summary.md"
    write_csv(rows, csv_path, fieldnames)
    write_markdown(rows, md_path, fieldnames)

    print("=== run_summarize_reports ===")
    print(f"compare_reports: {len(rows)}")
    print(f"sort_by: {args.sort_by}")
    print(f"saved_csv: {csv_path}")
    print(f"saved_markdown: {md_path}")
    print("=== top results ===")
    for row in rows[:10]:
        print(
            f"kd={row['packet_kd_mode']} retain={row['lambda_retain']} packet={row['lambda_packet']} "
            f"general={row['social_average_general_accuracy']:.4f} "
            f"delta_general={row['delta_average_general_accuracy']:+.4f} "
            f"expert={row['social_average_expert_accuracy']:.4f} "
            f"delta_expert={row['delta_average_expert_accuracy']:+.4f}"
        )


if __name__ == "__main__":
    main()