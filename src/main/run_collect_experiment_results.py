"""
Collect generalist packet experiment metrics into a global index.
"""

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=str, default="./outputs")
    return parser.parse_args()


def read_metrics(path: Path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"empty metrics.csv: {path}")
    row = rows[0]
    row["metrics_path"] = str(path)
    return row


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    metric_paths = sorted(output_root.glob("*/reports/generalist_packet/metrics.csv"))
    rows = [read_metrics(path) for path in metric_paths]

    out_csv = output_root / "experiment_index.csv"
    out_json = output_root / "experiment_index.json"
    output_root.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else ["experiment_id"]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"num_experiments": len(rows), "experiments": rows}, f, ensure_ascii=False, indent=2)

    print("=== run_collect_experiment_results ===")
    print(f"output_root: {output_root}")
    print(f"num_experiments: {len(rows)}")
    print(f"saved_csv: {out_csv}")
    print(f"saved_json: {out_json}")


if __name__ == "__main__":
    main()
