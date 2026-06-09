"""
Summarize generalist-packet compare reports across isolated experiments.
"""

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=str, default="./outputs")
    parser.add_argument("--out-json", type=str, default=None)
    parser.add_argument("--out-csv", type=str, default=None)
    return parser.parse_args()


def flatten_report(report, report_path: Path):
    experiment = report.get("experiment", {})
    packet = experiment.get("packet") or {}
    social_head = experiment.get("social_head") or {}
    summary = report.get("summary", {})
    communication = report.get("communication", {})
    return {
        "experiment_id": report.get("experiment_id"),
        "report_path": str(report_path),
        "config_path": experiment.get("config_path"),
        "seed": experiment.get("seed"),
        "dataset": report.get("dataset"),
        "split_name": report.get("split_name"),
        "packet_source": packet.get("source"),
        "ipc": packet.get("ipc"),
        "social_epochs": social_head.get("epochs"),
        "samples_per_class": social_head.get("samples_per_class"),
        "steps_per_epoch": social_head.get("steps_per_epoch"),
        "lambda_packet_ce": social_head.get("lambda_packet_ce"),
        "lambda_packet_kd": social_head.get("lambda_packet_kd"),
        "lambda_known_ce": social_head.get("lambda_known_ce"),
        "lambda_retain": social_head.get("lambda_retain"),
        "known_accuracy": summary.get("known_accuracy"),
        "missing_accuracy": summary.get("missing_accuracy"),
        "general_accuracy": summary.get("general_accuracy"),
        "delta_known_accuracy": summary.get("delta_known_accuracy"),
        "delta_missing_accuracy": summary.get("delta_missing_accuracy"),
        "delta_general_accuracy": summary.get("delta_general_accuracy"),
        "stage1_generalist_training_mib": communication.get("stage1_generalist_training_mib"),
        "stage2_missing_packet_mib": communication.get("stage2_missing_packet_mib"),
    }


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    report_paths = sorted(output_root.glob("*/reports/generalist_packet/compare_generalist_*.json"))
    rows = []
    seen = set()
    for report_path in report_paths:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        row = flatten_report(report, report_path)
        key = (row["experiment_id"], report_path.name)
        if key in seen:
            raise RuntimeError(f"duplicate summary key found: {key}")
        seen.add(key)
        rows.append(row)

    out_json = Path(args.out_json) if args.out_json else output_root / "generalist_experiment_summary.json"
    out_csv = Path(args.out_csv) if args.out_csv else output_root / "generalist_experiment_summary.csv"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"num_experiments": len(rows), "experiments": rows}, f, ensure_ascii=False, indent=2)

    fieldnames = list(rows[0].keys()) if rows else ["experiment_id"]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("=== run_summarize_generalist_experiments ===")
    print(f"output_root: {output_root}")
    print(f"num_experiments: {len(rows)}")
    print(f"saved_json: {out_json}")
    print(f"saved_csv: {out_csv}")


if __name__ == "__main__":
    main()
