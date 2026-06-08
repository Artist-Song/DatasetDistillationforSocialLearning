"""
Plot experiment comparison results from compare_summary.csv.

The main figure shows:
- final general accuracy
- final expert accuracy
- general/expert deltas from local_pretrain

It also exports a Pareto-style scatter plot for expert/general trade-offs.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from src.utils.config import load_yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="experiment config path")
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="",
        help="optional compare_summary.csv path; defaults to output.root/reports/summary/compare_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="optional output directory; defaults to output.root/reports/figures",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=12,
        help="number of rows to show in bar plots, sorted by general accuracy",
    )
    return parser.parse_args()


def to_float(value, default=0.0):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def infer_packet_source(row):
    source = row.get("packet_source", "")
    if source:
        return source
    report_name = row.get("report_name", "")
    if "src-dsdm" in report_name:
        return "dsdm"
    if "src-raw" in report_name:
        return "raw"
    return "unknown"


def infer_strategy(row):
    strategy = row.get("training_strategy", "")
    if strategy:
        return strategy

    report_name = row.get("report_name", "")
    if "packet_only_train" not in report_name:
        return "mixed_social_train"

    head = "head" if "_po-head" in report_name else "full"
    retain = "retain" if "_po-retain-1p0" in report_name else "no_retain"
    return f"packet_only_{head}_{retain}"


def load_rows(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    parsed = []
    seen = set()
    for row in rows:
        report_name = row.get("report_name", "")
        if report_name in seen:
            continue
        seen.add(report_name)

        packet_source = infer_packet_source(row)
        strategy = infer_strategy(row)
        label = f"{packet_source}\n{strategy.replace('packet_only_', 'po_').replace('mixed_social_train', 'mixed')}"
        parsed.append(
            {
                **row,
                "label": label,
                "packet_source": packet_source,
                "training_strategy": strategy,
                "general": to_float(row.get("social_average_general_accuracy")),
                "expert": to_float(row.get("social_average_expert_accuracy")),
                "delta_general": to_float(row.get("delta_average_general_accuracy")),
                "delta_expert": to_float(row.get("delta_average_expert_accuracy")),
            }
        )
    return parsed


def color_for(row):
    source = row["packet_source"]
    strategy = row["training_strategy"]
    if source == "dsdm" and "head" in strategy:
        return "#5B8FF9"
    if source == "dsdm":
        return "#61DDAA"
    if source == "raw" and "head" in strategy:
        return "#F6BD16"
    if source == "raw":
        return "#E8684A"
    return "#999999"


def annotate_bars(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_overview(rows, output_path: Path, top_n: int):
    rows = sorted(rows, key=lambda row: row["general"], reverse=True)[:top_n]
    labels = [row["label"] for row in rows]
    colors = [color_for(row) for row in rows]
    x = list(range(len(rows)))
    width = 0.38

    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(rows) * 1.1), 9), constrained_layout=True)

    ax = axes[0]
    general_bars = ax.bar([i - width / 2 for i in x], [row["general"] for row in rows], width, label="General", color=colors)
    expert_bars = ax.bar(
        [i + width / 2 for i in x],
        [row["expert"] for row in rows],
        width,
        label="Expert",
        color="#3D3D3D",
        alpha=0.72,
    )
    ax.set_title("Final Accuracy by Training Strategy")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    annotate_bars(ax, general_bars)
    annotate_bars(ax, expert_bars)

    ax = axes[1]
    delta_general_bars = ax.bar(
        [i - width / 2 for i in x],
        [row["delta_general"] for row in rows],
        width,
        label="Delta General",
        color=colors,
    )
    delta_expert_bars = ax.bar(
        [i + width / 2 for i in x],
        [row["delta_expert"] for row in rows],
        width,
        label="Delta Expert",
        color="#3D3D3D",
        alpha=0.72,
    )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Accuracy Change from Local Pretrain")
    ax.set_ylabel("Delta Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    annotate_bars(ax, delta_general_bars)
    annotate_bars(ax, delta_expert_bars)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_pareto(rows, output_path: Path):
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    for row in rows:
        ax.scatter(row["expert"], row["general"], s=90, color=color_for(row), alpha=0.85, edgecolor="black", linewidth=0.5)
        text = f"{row['packet_source']}:{row['training_strategy'].replace('packet_only_', 'po_').replace('mixed_social_train', 'mixed')}"
        ax.annotate(text, (row["expert"], row["general"]), xytext=(5, 4), textcoords="offset points", fontsize=8)

    ax.set_title("Expert vs General Accuracy Trade-off")
    ax.set_xlabel("Expert Accuracy")
    ax.set_ylabel("General Accuracy")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, max(0.4, max(row["general"] for row in rows) + 0.05))
    ax.grid(linestyle="--", alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    output_root = Path(cfg["output"]["root"])
    summary_csv = Path(args.summary_csv) if args.summary_csv else output_root / "reports" / "summary" / "compare_summary.csv"
    output_dir = Path(args.output_dir) if args.output_dir else output_root / "reports" / "figures"

    rows = load_rows(summary_csv)
    if not rows:
        raise RuntimeError(f"no rows found in summary csv: {summary_csv}")

    overview_path = output_dir / "experiment_overview.png"
    pareto_path = output_dir / "expert_general_tradeoff.png"
    plot_overview(rows, overview_path, args.top_n)
    plot_pareto(rows, pareto_path)

    print("=== run_plot_results ===")
    print(f"summary_csv: {summary_csv}")
    print(f"rows: {len(rows)}")
    print(f"saved_overview: {overview_path}")
    print(f"saved_tradeoff: {pareto_path}")


if __name__ == "__main__":
    main()
