#!/usr/bin/env python3
"""整理 CIFAR-100 实验结果到统一 registry。"""

from __future__ import annotations

import argparse
import csv
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


SOCIAL_COLUMNS = [
    "registry_source",
    "source_file",
    "run_name",
    "receiver_agent",
    "receiver_model",
    "expert_classes",
    "packet_method",
    "method",
    "init_mode",
    "use_fr",
    "lambda_fr",
    "use_logits",
    "communication_mode",
    "use_generalist_logits",
    "kd_mix_beta",
    "lambda_kd",
    "kd_temperature",
    "ipc",
    "external_comm_images",
    "external_comm_logit_bytes",
    "external_comm_generalist_logit_bytes",
    "acc_global_before",
    "acc_expert_before",
    "acc_global_after",
    "acc_expert_after",
    "acc_new_after",
    "forgetting",
    "loss_cls",
    "loss_fr",
    "loss_kd",
    "loss_sender_kd",
    "loss_generalist_kd",
    "time",
    "notes",
]

NUMERIC_COLUMNS = [
    "ipc",
    "external_comm_images",
    "external_comm_logit_bytes",
    "external_comm_generalist_logit_bytes",
    "acc_global_before",
    "acc_expert_before",
    "acc_global_after",
    "acc_expert_after",
    "acc_new_after",
    "forgetting",
    "loss_cls",
    "loss_fr",
    "loss_kd",
    "loss_sender_kd",
    "loss_generalist_kd",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_social_row(row: dict[str, str], source: Path, registry_source: str) -> dict[str, str]:
    """把不同来源的 social result 行补齐成统一字段。"""
    out = {col: "" for col in SOCIAL_COLUMNS}
    out.update({k: v for k, v in row.items() if k in out})
    out["registry_source"] = registry_source
    out["source_file"] = str(source)
    return out


def col_to_idx(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    idx = 0
    for ch in letters:
        idx = idx * 26 + ord(ch.upper()) - 64
    return idx - 1


def read_first_xlsx_sheet(path: Path) -> list[list[str]]:
    """用标准库读取 xlsx 第一张表，避免额外依赖。"""
    ns = {
        "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }
    rel_ns = {"pr": "http://schemas.openxmlformats.org/package/2006/relationships"}
    with zipfile.ZipFile(path) as zf:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", ns):
                shared.append("".join(t.text or "" for t in si.findall(".//a:t", ns)))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rid_to_target = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels.findall("pr:Relationship", rel_ns)
        }
        first_sheet = workbook.find("a:sheets/a:sheet", ns)
        if first_sheet is None:
            return []
        rid = first_sheet.attrib[
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
        ]
        target = rid_to_target[rid]
        sheet_path = "xl/" + target.lstrip("/") if not target.startswith("xl/") else target
        sheet_root = ET.fromstring(zf.read(sheet_path))

        rows: list[list[str]] = []
        for row in sheet_root.findall("a:sheetData/a:row", ns):
            cells: dict[int, str] = {}
            max_idx = -1
            for cell in row.findall("a:c", ns):
                idx = col_to_idx(cell.attrib.get("r", "A1"))
                max_idx = max(max_idx, idx)
                value = cell.find("a:v", ns)
                text = ""
                if value is not None and value.text is not None:
                    text = shared[int(value.text)] if cell.attrib.get("t") == "s" else value.text
                cells[idx] = text
            rows.append([cells.get(i, "") for i in range(max_idx + 1)])
        return rows


def load_legacy_excel(path: Path) -> list[dict[str, str]]:
    """把旧 Excel 汇总表转换成 registry 行。"""
    rows = read_first_xlsx_sheet(path)
    if not rows:
        return []

    header = rows[0]
    index = {name: i for i, name in enumerate(header)}
    out: list[dict[str, str]] = []
    current_method = ""
    for values in rows[1:]:
        method = values[index.get("method", 0)] if index.get("method", 0) < len(values) else ""
        if method:
            current_method = method
        agent = values[index.get("agent", 1)] if index.get("agent", 1) < len(values) else ""
        model = values[index.get("Model", 2)] if index.get("Model", 2) < len(values) else ""
        if not agent and not model:
            continue
        match = re.search(r"(\d+)", agent)
        if not match:
            continue
        agent_id = match.group(1)
        row = {col: "" for col in SOCIAL_COLUMNS}
        row.update(
            {
                "registry_source": "legacy_excel_summary",
                "source_file": str(path),
                "run_name": "legacy_cifar100_excel_summary",
                "receiver_agent": agent_id,
                "receiver_model": model,
                "packet_method": current_method.lower(),
                "method": current_method,
                "acc_global_after": values[index.get("Global acc", 3)]
                if index.get("Global acc", 3) < len(values)
                else "",
                "acc_expert_after": values[index.get("Expert acc", 4)]
                if index.get("Expert acc", 4) < len(values)
                else "",
                "acc_new_after": values[index.get("New acc", 5)]
                if index.get("New acc", 5) < len(values)
                else "",
                "notes": "旧 Excel 仅含 after 指标，before/loss/通信量等字段缺失。",
            }
        )
        out.append(row)
    return out


def dedupe_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """按实验语义去重，避免同一 CSV 被旧目录和 run 目录重复记录。"""
    seen: set[tuple[str, ...]] = set()
    unique: list[dict[str, str]] = []
    keys = [
        "run_name",
        "receiver_agent",
        "receiver_model",
        "method",
        "init_mode",
        "use_logits",
        "ipc",
        "acc_global_after",
        "acc_expert_after",
        "acc_new_after",
        "time",
    ]
    for row in rows:
        ident = tuple(row.get(k, "") for k in keys)
        if ident in seen:
            continue
        seen.add(ident)
        unique.append(row)
    return unique


def to_float(value: str) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except ValueError:
        return None


def build_method_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """按 run/method/init/logits/ipc 聚合四个 agent 的均值。"""
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("run_name", ""),
            row.get("method", ""),
            row.get("packet_method", ""),
            row.get("init_mode", ""),
            row.get("use_logits", ""),
            row.get("use_generalist_logits", ""),
            row.get("communication_mode", ""),
            row.get("ipc", ""),
            row.get("registry_source", ""),
        )
        groups[key].append(row)

    summary: list[dict[str, str]] = []
    for key, items in sorted(groups.items()):
        (
            run_name,
            method,
            packet_method,
            init_mode,
            use_logits,
            use_generalist_logits,
            communication_mode,
            ipc,
            registry_source,
        ) = key
        out = {
            "run_name": run_name,
            "method": method,
            "packet_method": packet_method,
            "init_mode": init_mode,
            "use_logits": use_logits,
            "use_generalist_logits": use_generalist_logits,
            "communication_mode": communication_mode,
            "ipc": ipc,
            "registry_source": registry_source,
            "num_agents": str(len(items)),
        }
        for col in NUMERIC_COLUMNS:
            vals = [to_float(row.get(col, "")) for row in items]
            vals = [v for v in vals if v is not None]
            out[f"mean_{col}"] = f"{sum(vals) / len(vals):.6g}" if vals else ""
        summary.append(out)
    return summary


def load_social_results(outputs_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(outputs_dir.glob("*/metrics/social_results.csv")):
        for row in read_csv_rows(path):
            rows.append(normalize_social_row(row, path, "pipeline_social_results"))

    legacy_csv = outputs_dir / "cifar100output" / "cifar100ipc50logits三种对比方法.csv"
    if legacy_csv.exists():
        for row in read_csv_rows(legacy_csv):
            rows.append(normalize_social_row(row, legacy_csv, "legacy_csv_full"))

    legacy_xlsx = outputs_dir / "cifar100output" / "结果整理.xlsx"
    if legacy_xlsx.exists():
        rows.extend(load_legacy_excel(legacy_xlsx))
    return dedupe_rows(rows)


def load_centralized_results(outputs_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    columns: set[str] = {"registry_source", "source_file"}
    for path in sorted(outputs_dir.glob("*/centralized_full/*.csv")):
        for row in read_csv_rows(path):
            normalized = {"registry_source": "centralized_full", "source_file": str(path)}
            normalized.update(row)
            rows.append(normalized)
            columns.update(normalized.keys())
    if not rows:
        return []
    ordered = ["registry_source", "source_file"] + sorted(
        col for col in columns if col not in {"registry_source", "source_file"}
    )
    for row in rows:
        for col in ordered:
            row.setdefault(col, "")
    rows.insert(0, {"__columns__": ordered})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Update outputs/experiment_registry tables.")
    parser.add_argument("--outputs-dir", default="outputs", type=Path)
    parser.add_argument("--registry-dir", default=Path("outputs/experiment_registry"), type=Path)
    args = parser.parse_args()

    social_rows = load_social_results(args.outputs_dir)
    summary_rows = build_method_summary(social_rows)
    centralized_rows = load_centralized_results(args.outputs_dir)

    write_csv(args.registry_dir / "social_results_all.csv", social_rows, SOCIAL_COLUMNS)
    write_csv(
        args.registry_dir / "method_summary.csv",
        summary_rows,
        [
            "run_name",
            "method",
            "packet_method",
            "init_mode",
            "use_logits",
            "use_generalist_logits",
            "communication_mode",
            "ipc",
            "registry_source",
            "num_agents",
        ]
        + [f"mean_{col}" for col in NUMERIC_COLUMNS],
    )
    if centralized_rows:
        centralized_columns = centralized_rows[0]["__columns__"]  # type: ignore[index]
        write_csv(args.registry_dir / "centralized_results_all.csv", centralized_rows[1:], centralized_columns)

    readme = args.registry_dir / "README.md"
    readme.write_text(
        "# Experiment Registry\n\n"
        "This folder stores consolidated experiment result tables generated by "
        "`python scripts/update_experiment_registry.py`.\n\n"
        "- `social_results_all.csv`: per-agent social-learning rows from pipeline CSVs, "
        "legacy CSV, and the old Excel summary.\n"
        "- `method_summary.csv`: mean metrics grouped by run/method/init/logits/ipc.\n"
        "- `centralized_results_all.csv`: centralized full-data upper-bound CSV rows.\n",
        encoding="utf-8",
    )

    print(f"[registry] social rows: {len(social_rows)}")
    print(f"[registry] method summary rows: {len(summary_rows)}")
    print(f"[registry] output: {args.registry_dir}")


if __name__ == "__main__":
    main()
