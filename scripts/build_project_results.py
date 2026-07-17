#!/usr/bin/env python3
"""Build canonical project results from the curated experiment registry."""

from __future__ import annotations

import csv
import json
import statistics
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "experiments" / "registry.yaml"
METRICS = ("global", "new", "expert", "forgetting")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_registry() -> dict:
    with REGISTRY_PATH.open("r", encoding="utf-8") as handle:
        registry = yaml.safe_load(handle)
    if not isinstance(registry, dict) or not isinstance(registry.get("experiments"), list):
        raise ValueError(f"Invalid registry: {REGISTRY_PATH}")
    return registry


def aggregate(values: list[dict[str, float]]) -> dict[str, float | int]:
    result: dict[str, float | int] = {"seeds": len(values)}
    for metric in METRICS:
        metric_values = [row[metric] for row in values]
        result[f"{metric}_mean"] = statistics.fmean(metric_values)
        result[f"{metric}_std"] = statistics.pstdev(metric_values) if len(metric_values) > 1 else 0.0
    return result


def collect_pipeline(entry: dict) -> dict:
    source = entry["source"]
    expected_receivers = {str(value) for value in source["expected_receivers"]}
    seed_means = []
    source_files = []
    missing_seeds = []
    for seed in source["expected_seeds"]:
        path = ROOT / source["path_template"].format(seed=seed)
        if not path.is_file():
            missing_seeds.append(int(seed))
            continue
        selected = [row for row in read_csv(path) if row.get("method") == source["method"]]
        # Appended CSVs may contain reruns. The latest row for each receiver is canonical.
        latest = {}
        for row in selected:
            latest[str(row[source["receiver_key"]])] = row
        if set(latest) != expected_receivers:
            missing_seeds.append(int(seed))
            continue
        seed_means.append(
            {
                metric: statistics.fmean(float(row[column]) for row in latest.values())
                for metric, column in source["metrics"].items()
            }
        )
        source_files.append(str(path.relative_to(ROOT)))
    if not seed_means:
        raise FileNotFoundError(f"No complete seed found for {entry['id']}")
    if missing_seeds and not entry.get("allow_partial", False):
        raise ValueError(f"{entry['id']} missing complete seeds: {missing_seeds}")
    result = aggregate(seed_means)
    result.update(
        {
            "expected_seeds": len(source["expected_seeds"]),
            "missing_seeds": missing_seeds,
            "actual_status": "complete" if not missing_seeds else "interim",
            "source_files": source_files,
        }
    )
    return result


def collect_average_row(entry: dict) -> dict:
    source = entry["source"]
    path = ROOT / source["path"]
    average = next(
        (
            row
            for row in read_csv(path)
            if str(row.get(source["average_key"], "")) == str(source["average_value"])
        ),
        None,
    )
    if average is None:
        raise ValueError(f"{entry['id']} has no average row in {path}")
    result: dict[str, float | int | list] = {
        "seeds": 1,
        "expected_seeds": 1,
        "missing_seeds": [],
        "actual_status": "complete",
        "source_files": [str(path.relative_to(ROOT))],
    }
    for metric, column in source["metrics"].items():
        result[f"{metric}_mean"] = float(average[column])
        result[f"{metric}_std"] = 0.0
    return result


def collect_entry(entry: dict) -> dict:
    source_type = entry["source"]["type"]
    if source_type == "pipeline":
        result = collect_pipeline(entry)
    elif source_type == "average_row":
        result = collect_average_row(entry)
    else:
        raise ValueError(f"Unsupported numeric source type for {entry['id']}: {source_type}")
    return {**entry, **result}


def formatted(row: dict, metric: str) -> str:
    mean = float(row[f"{metric}_mean"])
    # Normalize binary floating-point noise before applying paper-style half-up rounding.
    mean_text = str(
        Decimal(f"{mean:.10f}").quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    )
    if int(row["seeds"]) > 1:
        std = float(row[f"{metric}_std"])
        std_text = str(
            Decimal(f"{std:.10f}").quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        )
        return f"{mean_text} +/- {std_text}"
    return mean_text


def display_title(title: str) -> str:
    return f"{title}*" if title in {"MASC-complete", "DeSA-CIL"} else title


def build_numeric_rows(registry: dict) -> list[dict]:
    return [
        collect_entry(entry)
        for entry in registry["experiments"]
        if entry.get("paper_eligible") and entry["source"]["type"] != "evidence"
    ]


def best_values(rows: list[dict], ipc: int) -> dict[str, float]:
    direct_group = f"one_resnet_ipc{ipc}_image_budget"
    subset = [
        row
        for row in rows
        if row.get("table_section") == "ipc"
        and row.get("ipc") == ipc
        and row.get("comparability_group") == direct_group
    ]
    return {
        "global": max(float(row["global_mean"]) for row in subset),
        "new": max(float(row["new_mean"]) for row in subset),
        "expert": max(float(row["expert_mean"]) for row in subset),
        "forgetting": min(float(row["forgetting_mean"]) for row in subset),
    }


def main_markdown_table(rows: list[dict]) -> list[str]:
    ipc_rows = [row for row in rows if row.get("table_section") == "ipc"]
    by_key = {(row["title"], int(row["ipc"])): row for row in ipc_rows}
    titles = sorted(
        {row["title"] for row in ipc_rows},
        key=lambda title: min(int(row["table_order"]) for row in ipc_rows if row["title"] == title),
    )
    best = {ipc: best_values(rows, ipc) for ipc in (10, 50)}
    lines = [
        "| Method | Packet | Soft labels | n10 | IPC10 Global | New | Expert | Forgetting | n50 | IPC50 Global | New | Expert | Forgetting |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for title in titles:
        groups = []
        metadata = None
        for ipc in (10, 50):
            row = by_key[(title, ipc)]
            metadata = row
            values = []
            for metric in METRICS:
                value = formatted(row, metric)
                direct_group = f"one_resnet_ipc{ipc}_image_budget"
                if (
                    row.get("comparability_group") == direct_group
                    and abs(float(row[f"{metric}_mean"]) - best[ipc][metric]) < 1e-10
                ):
                    value = f"**{value}**"
                values.append(value)
            groups.append([str(row["seeds"]), *values])
        lines.append(
            f"| {display_title(title)} | {metadata['packet']} | {metadata['soft_labels']} | "
            + " | ".join(groups[0] + groups[1])
            + " |"
        )
    return lines


def validation_table(rows: list[dict]) -> list[str]:
    validation = sorted(
        (row for row in rows if row.get("table_section") == "validation"),
        key=lambda row: int(row["table_order"]),
    )
    lines = [
        "| Method | Protocol | n | Global | New | Expert | Forgetting |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in validation:
        lines.append(
            f"| {row['title']} | {row['protocol']} | {row['seeds']} | "
            + " | ".join(formatted(row, metric) for metric in METRICS)
            + " |"
        )
    return lines


def evidence_table(registry: dict) -> list[str]:
    entries = [
        entry for entry in registry["experiments"] if entry["source"]["type"] == "evidence"
    ]
    lines = [
        "| ID | Role | Status | Evidence retained |",
        "|---|---|---|---|",
    ]
    for entry in entries:
        paths = "<br>".join(f"`{path}`" for path in entry["source"]["paths"])
        lines.append(
            f"| `{entry['id']}` | {entry['role']} | {entry['status']} | "
            f"{entry.get('description', '')}<br>{paths} |"
        )
    return lines


def build_results_markdown(registry: dict, rows: list[dict]) -> str:
    lines = [
        "# RESULTS.md",
        "",
        f"Generated from `experiments/registry.yaml` on `{date.today().isoformat()}`.",
        "Do not edit numeric tables manually.",
        "",
        "## Canonical IPC Results",
        "",
        *main_markdown_table(rows),
        "",
        "Notes:",
        "",
        "- Ours is sender task-backbone-guided DSDM with sender expert-class logits.",
        "- Heuristic and FAST use hard labels only. FAST follows official pixels/per-class/minmax.",
        "- MASC-complete* and DeSA-CIL* are single-seed adapted baselines with protocol caveats.",
        "- Bold values are computed only within the directly comparable one-ResNet image-budget group.",
        "- Ours IPC=50 remains interim while any expected seed lacks four receiver rows.",
        "- High expert accuracy with near-zero new accuracy is not successful socialization.",
        "",
        "## Full-data Validation",
        "",
        *validation_table(rows),
        "",
        "Full-data rows validate implementations and are not IPC-matched method rankings.",
        "",
        "## Important Intermediate Evidence",
        "",
        *evidence_table(registry),
        "",
        "## Provenance",
        "",
        "| Experiment | Status | Seeds | Missing | Source files | Caveat |",
        "|---|---|---:|---|---|---|",
    ]
    for row in sorted(rows, key=lambda item: item["id"]):
        sources = "<br>".join(f"`{path}`" for path in row["source_files"])
        missing = ",".join(str(seed) for seed in row["missing_seeds"]) or "-"
        lines.append(
            f"| `{row['id']}` | {row['actual_status']} | {row['seeds']} | {missing} | "
            f"{sources} | {row.get('caveat', '')} |"
        )
    return "\n".join(lines) + "\n"


def write_csv(rows: list[dict]) -> None:
    fields = [
        "id", "title", "role", "status", "actual_status", "paper_eligible", "table_section",
        "ipc", "packet", "soft_labels", "protocol", "comparability_group", "seeds",
        "expected_seeds", "missing_seeds", "global_mean", "global_std", "new_mean", "new_std",
        "expert_mean", "expert_std", "forgetting_mean", "forgetting_std", "caveat", "source_files",
    ]
    destinations = [
        ROOT / "experiments" / "generated" / "canonical_results.csv",
        ROOT / "paper_tables" / "main_results_ipc10_ipc50.csv",
    ]
    for path in destinations:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                clean = {field: row.get(field, "") for field in fields}
                clean["missing_seeds"] = json.dumps(row["missing_seeds"])
                clean["source_files"] = json.dumps(row["source_files"], ensure_ascii=False)
                writer.writerow(clean)


def write_latex(rows: list[dict]) -> None:
    ipc_rows = [row for row in rows if row.get("table_section") == "ipc"]
    by_key = {(row["title"], int(row["ipc"])): row for row in ipc_rows}
    titles = sorted(
        {row["title"] for row in ipc_rows},
        key=lambda title: min(int(row["table_order"]) for row in ipc_rows if row["title"] == title),
    )
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Canonical one-ResNet socialized-learning results. Values are percentages.}",
        r"\begin{tabular}{lcrrrrcrrrr}",
        r"\toprule",
        r"& \multicolumn{5}{c}{IPC=10} & \multicolumn{5}{c}{IPC=50} \\",
        r"Method & $n$ & Global & New & Expert & Forgetting & $n$ & Global & New & Expert & Forgetting \\",
        r"\midrule",
    ]
    for title in titles:
        groups = []
        for ipc in (10, 50):
            row = by_key[(title, ipc)]
            values = [formatted(row, metric).replace("+/-", r"$\pm$") for metric in METRICS]
            groups.extend([str(row["seeds"]), *values])
        latex_title = title.replace("_", r"\_")
        lines.append(f"{latex_title} & " + " & ".join(groups) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (ROOT / "paper_tables" / "main_results_ipc10_ipc50.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    registry = load_registry()
    rows = build_numeric_rows(registry)
    results_markdown = build_results_markdown(registry, rows)
    (ROOT / "RESULTS.md").write_text(results_markdown, encoding="utf-8")
    paper_markdown = results_markdown.replace("# RESULTS.md", f"# Main Results ({date.today().isoformat()})", 1)
    (ROOT / "paper_tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "paper_tables" / "main_results_ipc10_ipc50.md").write_text(
        paper_markdown, encoding="utf-8"
    )
    write_csv(rows)
    write_latex(rows)
    provenance = ROOT / "paper_tables" / "main_results_ipc10_ipc50_provenance.json"
    provenance.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[results] canonical rows: {len(rows)}")
    print("[results] wrote RESULTS.md, experiments/generated/, and paper_tables/")


if __name__ == "__main__":
    main()
