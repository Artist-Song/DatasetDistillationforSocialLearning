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
SCALING_METRICS = ("global", "new", "expert")


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


def aggregate_selected(
    values: list[dict[str, float]], metrics: tuple[str, ...]
) -> dict[str, float | int]:
    result: dict[str, float | int] = {"seeds": len(values)}
    for metric in metrics:
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


def _complete_seed_result(
    entry: dict,
    seed_means: list[dict[str, float]],
    source_files: list[str],
    missing_seeds: list[int],
) -> dict:
    if not seed_means:
        raise FileNotFoundError(f"No complete seed found for {entry['id']}")
    if missing_seeds and not entry.get("allow_partial", False):
        raise ValueError(f"{entry['id']} missing complete seeds: {missing_seeds}")
    result = aggregate_selected(seed_means, SCALING_METRICS)
    result.update(
        {
            "expected_seeds": len(entry["source"]["expected_seeds"]),
            "missing_seeds": missing_seeds,
            "actual_status": "complete" if not missing_seeds else "interim",
            "source_files": source_files,
        }
    )
    return {**entry, **result}


def collect_scaling_pipeline(entry: dict) -> dict:
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
        latest = {}
        for row in selected:
            latest[str(row[source["receiver_key"]])] = row
        if set(latest) != expected_receivers:
            missing_seeds.append(int(seed))
            continue
        seed_means.append(
            {
                metric: statistics.fmean(float(row[source["metrics"][metric]]) for row in latest.values())
                for metric in SCALING_METRICS
            }
        )
        source_files.append(str(path.relative_to(ROOT)))
    return _complete_seed_result(entry, seed_means, source_files, missing_seeds)


def collect_scaling_average_rows(entry: dict) -> dict:
    source = entry["source"]
    seed_means = []
    source_files = []
    missing_seeds = []
    for seed in source["expected_seeds"]:
        path = ROOT / source["path_template"].format(seed=seed)
        if not path.is_file():
            missing_seeds.append(int(seed))
            continue
        average = next(
            (
                row
                for row in read_csv(path)
                if str(row.get(source["average_key"], "")) == str(source["average_value"])
            ),
            None,
        )
        if average is None:
            missing_seeds.append(int(seed))
            continue
        seed_means.append(
            {metric: float(average[source["metrics"][metric]]) for metric in SCALING_METRICS}
        )
        source_files.append(str(path.relative_to(ROOT)))
    return _complete_seed_result(entry, seed_means, source_files, missing_seeds)


def collect_scaling_json(entry: dict) -> dict:
    source = entry["source"]
    seed_means = []
    source_files = []
    missing_seeds = []
    for seed in source["expected_seeds"]:
        path = ROOT / source["path_template"].format(seed=seed)
        if not path.is_file():
            missing_seeds.append(int(seed))
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            missing_seeds.append(int(seed))
            continue
        seed_means.append(
            {metric: float(payload[source["metrics"][metric]]) for metric in SCALING_METRICS}
        )
        source_files.append(str(path.relative_to(ROOT)))
    return _complete_seed_result(entry, seed_means, source_files, missing_seeds)


def collect_scaling_entry(entry: dict) -> dict:
    source_type = entry["source"]["type"]
    if source_type == "scaling_pipeline":
        return collect_scaling_pipeline(entry)
    if source_type == "scaling_average_rows":
        return collect_scaling_average_rows(entry)
    if source_type == "scaling_json":
        return collect_scaling_json(entry)
    raise ValueError(f"Unsupported scaling source type for {entry['id']}: {source_type}")


def _read_communication_payload(path: Path, source_type: str, agent_count: int) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_status = {
        "dkp_summary": "complete_seed",
        "hard_label_audit": "passed",
        "desa_audit": "passed",
        "masc_audit": "passed",
        "fedre_summary": "complete",
    }[source_type]
    if payload.get("status") != expected_status:
        raise ValueError(
            f"Incomplete communication source {path}: "
            f"expected status {expected_status!r}, got {payload.get('status')!r}"
        )
    if source_type == "dkp_summary":
        communication = payload["communication"]
        return {
            "unique_images": int(communication["unique_sender_raw_images"]),
            "image_deliveries": int(communication["receiver_incidence_raw_images"]),
            "aux_bytes": int(communication["receiver_incidence_logit_bytes"]),
            "reported_rounds": 1,
            "executed_rounds": 1,
            "executed_aux_bytes": int(communication["receiver_incidence_logit_bytes"]),
        }
    if source_type == "hard_label_audit":
        unique_images = int(payload["unique_sender_images"])
        return {
            "unique_images": unique_images,
            "image_deliveries": unique_images * (agent_count - 1),
            "aux_bytes": int(payload.get("sender_logit_bytes", 0)),
            "reported_rounds": 1,
            "executed_rounds": 1,
            "executed_aux_bytes": int(payload.get("sender_logit_bytes", 0)),
        }
    if source_type == "desa_audit":
        communication = payload["communication"]
        return {
            "unique_images": int(communication["unique_sender_images"]),
            "image_deliveries": int(communication["receiver_incidence_images"]),
            "aux_bytes": int(communication["iterative_owner_logit_bytes_all_agents"]),
            "reported_rounds": int(communication["rounds"]),
            "executed_rounds": int(communication["rounds"]),
            "executed_aux_bytes": int(communication["iterative_owner_logit_bytes_all_agents"]),
        }
    if source_type == "masc_audit":
        return {
            "unique_images": int(payload["unique_sender_images"]),
            "image_deliveries": None,
            "aux_bytes": int(payload["model_parameter_bytes"]),
            "reported_rounds": 1,
            "executed_rounds": 1,
            "executed_aux_bytes": int(payload["model_parameter_bytes"]),
        }
    if source_type == "fedre_summary":
        communication = payload["communication"]
        executed_rounds = int(communication["official_loop_updates"])
        reported_rounds = int(payload["paper_round"])
        executed_aux_bytes = int(communication["total_logical_bytes"])
        if executed_aux_bytes % executed_rounds:
            raise ValueError(f"FedRE communication is not round-linear in {path}")
        return {
            "unique_images": 0,
            "image_deliveries": 0,
            "aux_bytes": executed_aux_bytes // executed_rounds * reported_rounds,
            "reported_rounds": reported_rounds,
            "executed_rounds": executed_rounds,
            "executed_aux_bytes": executed_aux_bytes,
        }
    raise ValueError(f"Unsupported communication source type: {source_type}")


def collect_scaling_communication(entry: dict) -> dict:
    communication = entry["communication"]
    source = communication["source"]
    records = []
    source_files = []
    for seed in entry["source"]["expected_seeds"]:
        path = ROOT / source["path_template"].format(seed=seed)
        if not path.is_file():
            raise FileNotFoundError(f"Missing communication source for {entry['id']}: {path}")
        records.append(_read_communication_payload(path, source["type"], int(entry["agent_count"])))
        source_files.append(str(path.relative_to(ROOT)))
    canonical = records[0]
    if any(record != canonical for record in records[1:]):
        raise ValueError(f"Communication accounting differs across seeds for {entry['id']}")
    return {
        "id": entry["id"],
        "title": entry["title"],
        "table_order": entry["table_order"],
        "agent_count": entry["agent_count"],
        "comparability_group": entry["comparability_group"],
        "object": communication["object"],
        "aux_kind": communication["aux_kind"],
        "note": communication.get("note", ""),
        "source_files": source_files,
        **canonical,
    }


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
        if entry.get("paper_eligible")
        and entry.get("table_section") != "scaling"
        and entry["source"]["type"] != "evidence"
    ]


def build_scaling_rows(registry: dict) -> list[dict]:
    return [
        collect_scaling_entry(entry)
        for entry in registry["experiments"]
        if entry.get("paper_eligible") and entry.get("table_section") == "scaling"
    ]


def build_scaling_communication_rows(registry: dict) -> list[dict]:
    return [
        collect_scaling_communication(entry)
        for entry in registry["experiments"]
        if entry.get("paper_eligible") and entry.get("table_section") == "scaling"
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


def scaling_best_values(rows: list[dict], agent_count: int) -> dict[str, float]:
    direct_group = f"iclr2027_{agent_count}agent_ipc10_image_budget"
    subset = [
        row
        for row in rows
        if int(row["agent_count"]) == agent_count
        and row.get("comparability_group") == direct_group
    ]
    return {
        metric: max(float(row[f"{metric}_mean"]) for row in subset)
        for metric in SCALING_METRICS
    }


def scaling_performance_table(rows: list[dict]) -> list[str]:
    by_key = {(row["title"], int(row["agent_count"])): row for row in rows}
    titles = sorted(
        {row["title"] for row in rows},
        key=lambda title: min(int(row["table_order"]) for row in rows if row["title"] == title),
    )
    best = {agent_count: scaling_best_values(rows, agent_count) for agent_count in (5, 10, 20)}
    lines = [
        "| Method | 5 agents Global | New | Expert | 10 agents Global | New | Expert | 20 agents Global | New | Expert |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for title in titles:
        values = []
        for agent_count in (5, 10, 20):
            row = by_key.get((title, agent_count))
            if row is None:
                values.extend(["-", "-", "-"])
                continue
            for metric in SCALING_METRICS:
                value = formatted(row, metric)
                direct_group = f"iclr2027_{agent_count}agent_ipc10_image_budget"
                if (
                    row.get("comparability_group") == direct_group
                    and abs(float(row[f"{metric}_mean"]) - best[agent_count][metric]) < 1e-10
                ):
                    value = f"**{value}**"
                values.append(value)
        lines.append(f"| {display_title(title)} | " + " | ".join(values) + " |")
    return lines


def format_bytes(value: int) -> str:
    if value == 0:
        return "0"
    return f"{value:,} ({value / 1024**2:.2f} MiB)"


def scaling_communication_table(rows: list[dict]) -> list[str]:
    lines = [
        "| Agents | Method | Native communication object | Unique/shared images | Image deliveries | Logical delivered auxiliary payload | Rounds to reported result |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (int(item["agent_count"]), int(item["table_order"]))):
        deliveries = "-" if row["image_deliveries"] is None else f"{int(row['image_deliveries']):,}"
        aux = f"{row['aux_kind']}: {format_bytes(int(row['aux_bytes']))}"
        lines.append(
            f"| {row['agent_count']} | {display_title(row['title'])} | {row['object']} | "
            f"{int(row['unique_images']):,} | {deliveries} | {aux} | {row['reported_rounds']} |"
        )
    return lines


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


def build_results_markdown(
    registry: dict,
    rows: list[dict],
    scaling_rows: list[dict],
    communication_rows: list[dict],
) -> str:
    lines = [
        "# RESULTS.md",
        "",
        f"Generated from `experiments/registry.yaml` on `{date.today().isoformat()}`.",
        "Do not edit numeric tables manually.",
        "",
        "## ICLR 2027 CIFAR-100 Scaling Results",
        "",
        *scaling_performance_table(scaling_rows),
        "",
        "Notes:",
        "",
        "- Every populated cell is mean +/- population standard deviation over three seed-level means; each seed mean first averages all receivers at that agent count.",
        "- Global is the primary metric; New and Expert are co-primary supporting metrics. Forgetting remains diagnostic-only.",
        "- Bold values are computed only among Ours, Heuristic, and FAST inside the matched IPC10 image-packet group.",
        "- Full Real is a 5-agent high-communication oracle. It was not scheduled for 10 or 20 agents.",
        "- DeSA-CIL* and MASC-complete* are explicitly adapted baselines. FedRE uses its native heterogeneous federated protocol.",
        "",
        "## ICLR 2027 Communication Accounting",
        "",
        *scaling_communication_table(communication_rows),
        "",
        "Communication notes:",
        "",
        "- Image counts and auxiliary bytes are reported separately; images are not converted to an assumed transport byte format.",
        "- Image deliveries count logical sender-to-receiver incidence. MASC uses one central CC pool, so peer delivery incidence is not applicable.",
        "- Auxiliary bytes are logical delivered bytes through the reported checkpoint, not unique tensor storage on disk.",
        "- DKP factor decoding does not increase transmitted image count.",
        "- FedRE bytes stop at the reported round-100 checkpoint. The official `range(global_rounds + 1)` loop executes one post-checkpoint update; that extra update is retained only in provenance and excluded from the paper communication value.",
        "",
        "## ICLR 2027 Scaling Provenance",
        "",
        *scaling_provenance_table(scaling_rows, communication_rows),
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
        "- Ours IPC=50 is the final two-seed result over seeds 0 and 2; Heuristic and FAST retain their three-seed aggregates.",
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


def scaling_provenance_table(rows: list[dict], communication_rows: list[dict]) -> list[str]:
    communication_by_id = {row["id"]: row for row in communication_rows}
    lines = [
        "| Experiment | Status | Seeds | Metric sources | Communication sources | Caveat |",
        "|---|---|---:|---|---|---|",
    ]
    for row in sorted(rows, key=lambda item: (int(item["table_order"]), int(item["agent_count"]))):
        communication = communication_by_id[row["id"]]
        metric_sources = "<br>".join(f"`{path}`" for path in row["source_files"])
        communication_sources = "<br>".join(
            f"`{path}`" for path in communication["source_files"]
        )
        lines.append(
            f"| `{row['id']}` | {row['actual_status']} | {row['seeds']} | {metric_sources} | "
            f"{communication_sources} | {row.get('caveat', '')} |"
        )
    return lines


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


def write_scaling_outputs(rows: list[dict], communication_rows: list[dict]) -> None:
    result_fields = [
        "id", "title", "role", "status", "actual_status", "paper_eligible", "agent_count",
        "ipc", "packet", "soft_labels", "protocol", "comparability_group", "seeds",
        "expected_seeds", "missing_seeds", "global_mean", "global_std", "new_mean", "new_std",
        "expert_mean", "expert_std", "caveat", "source_files",
    ]
    result_destinations = [
        ROOT / "experiments" / "generated" / "iclr2027_scaling_results.csv",
        ROOT / "paper_tables" / "iclr2027_scaling_results.csv",
    ]
    for path in result_destinations:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=result_fields, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                clean = {field: row.get(field, "") for field in result_fields}
                clean["missing_seeds"] = json.dumps(row["missing_seeds"])
                clean["source_files"] = json.dumps(row["source_files"], ensure_ascii=False)
                writer.writerow(clean)

    communication_fields = [
        "id", "title", "agent_count", "comparability_group", "object", "unique_images",
        "image_deliveries", "aux_kind", "aux_bytes", "reported_rounds", "executed_rounds",
        "executed_aux_bytes", "note", "source_files",
    ]
    communication_destinations = [
        ROOT / "experiments" / "generated" / "iclr2027_scaling_communication.csv",
        ROOT / "paper_tables" / "iclr2027_scaling_communication.csv",
    ]
    for path in communication_destinations:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=communication_fields, lineterminator="\n")
            writer.writeheader()
            for row in communication_rows:
                clean = {field: row.get(field, "") for field in communication_fields}
                clean["source_files"] = json.dumps(row["source_files"], ensure_ascii=False)
                writer.writerow(clean)

    performance_markdown = [
        "# ICLR 2027 CIFAR-100 Scaling Results",
        "",
        *scaling_performance_table(rows),
        "",
        "Values are three-seed mean +/- population standard deviation of seed-level receiver means.",
        "",
    ]
    communication_markdown = [
        "# ICLR 2027 Communication Accounting",
        "",
        *scaling_communication_table(communication_rows),
        "",
        "Image counts and logical delivered auxiliary tensor bytes are intentionally reported separately.",
        "",
    ]
    (ROOT / "paper_tables" / "iclr2027_scaling_results.md").write_text(
        "\n".join(performance_markdown), encoding="utf-8"
    )
    (ROOT / "paper_tables" / "iclr2027_scaling_communication.md").write_text(
        "\n".join(communication_markdown), encoding="utf-8"
    )

    result_by_key = {(row["title"], int(row["agent_count"])): row for row in rows}
    titles = sorted(
        {row["title"] for row in rows},
        key=lambda title: min(int(row["table_order"]) for row in rows if row["title"] == title),
    )
    latex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{CIFAR-100 social-learning scaling results. Values are three-seed mean $\pm$ population standard deviation.}",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{5 agents} & \multicolumn{3}{c}{10 agents} & \multicolumn{3}{c}{20 agents} \\",
        r"Method & Global & New & Expert & Global & New & Expert & Global & New & Expert \\",
        r"\midrule",
    ]
    for title in titles:
        latex_title = display_title(title).replace("_", r"\_")
        values = []
        for agent_count in (5, 10, 20):
            row = result_by_key.get((title, agent_count))
            if row is None:
                values.extend(["--", "--", "--"])
            else:
                values.extend(
                    formatted(row, metric).replace("+/-", r"$\pm$")
                    for metric in SCALING_METRICS
                )
        latex.append(f"{latex_title} & " + " & ".join(values) + r" \\")
    latex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (ROOT / "paper_tables" / "iclr2027_scaling_results.tex").write_text(
        "\n".join(latex) + "\n", encoding="utf-8"
    )

    provenance = ROOT / "paper_tables" / "iclr2027_scaling_provenance.json"
    provenance.write_text(
        json.dumps(
            {"performance": rows, "communication": communication_rows},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    registry = load_registry()
    rows = build_numeric_rows(registry)
    scaling_rows = build_scaling_rows(registry)
    communication_rows = build_scaling_communication_rows(registry)
    results_markdown = build_results_markdown(registry, rows, scaling_rows, communication_rows)
    (ROOT / "RESULTS.md").write_text(results_markdown, encoding="utf-8")
    paper_markdown = results_markdown.replace("# RESULTS.md", f"# Main Results ({date.today().isoformat()})", 1)
    (ROOT / "paper_tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "paper_tables" / "main_results_ipc10_ipc50.md").write_text(
        paper_markdown, encoding="utf-8"
    )
    write_csv(rows)
    write_latex(rows)
    write_scaling_outputs(scaling_rows, communication_rows)
    provenance = ROOT / "paper_tables" / "main_results_ipc10_ipc50_provenance.json"
    provenance.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[results] canonical rows: {len(rows)}; scaling rows: {len(scaling_rows)}")
    print("[results] wrote RESULTS.md, experiments/generated/, and paper_tables/")


if __name__ == "__main__":
    main()
