#!/usr/bin/env python3
"""Build a review-only cleanup list without deleting experiment artifacts."""

from __future__ import annotations

import csv
import re
import stat
import statistics
import subprocess
from collections import defaultdict
from datetime import date
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
REGISTRY = ROOT / "experiments" / "registry.yaml"
CANDIDATE_PATTERN = re.compile(
    r"guardian|tuning|ablation|recipe|smoke|trash|midpacket|refine|joint", re.IGNORECASE
)
PROTECTED_PATTERNS = [
    re.compile(r"one_resnet"),
    re.compile(r"conv_family_ipc(?:10|50)(?:$|_)"),
    re.compile(r"ipc50_allconvnet(?:$|-)"),
    re.compile(r"hetero4arch_ipc(?:10|50)(?:$|_)"),
    re.compile(r"hetero6arch_ipc(?:10|50)_seed"),
    re.compile(r"upper_bound"),
    re.compile(r"novgg"),
    re.compile(r"cross_arch"),
    re.compile(r"resnet18_(?:compact|standard)_recipe_diag"),
    re.compile(r"experiment_registry"),
]
METRICS = {
    "global": "acc_global_after",
    "new": "acc_new_after",
    "expert": "acc_expert_after",
    "forgetting": "forgetting",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def protected_registry_runs() -> set[str]:
    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    names = set()
    for entry in registry["experiments"]:
        source = entry["source"]
        paths = []
        if source["type"] == "pipeline":
            paths = [source["path_template"].format(seed=seed) for seed in source["expected_seeds"]]
        elif source["type"] == "average_row":
            paths = [source["path"]]
        else:
            paths = source.get("paths", [])
        for value in paths:
            parts = Path(value).parts
            if len(parts) > 1 and parts[0] == "outputs":
                names.add(parts[1])
    return names


def is_protected(name: str, registry_names: set[str]) -> bool:
    return name in registry_names or any(pattern.search(name) for pattern in PROTECTED_PATTERNS)


def family_name(name: str) -> str:
    value = re.sub(r"_[A-I](?=_|$)", "", name)
    value = re.sub(r"_(?:ep|fr|kd|lr|t|self|p|w)\d.*$", "", value)
    return value


def summarize_run(path: Path) -> list[dict]:
    metrics_path = path / "metrics" / "social_results.csv"
    if not metrics_path.is_file():
        return []
    rows = read_csv(metrics_path)
    latest = {}
    for row in rows:
        receiver = row.get("receiver_agent", row.get("agent_id", ""))
        method = row.get("method", "")
        if receiver in {"", "avg"} or not method:
            continue
        latest[(method, receiver)] = row
    by_method = defaultdict(list)
    for (method, _), row in latest.items():
        by_method[method].append(row)
    summaries = []
    for method, items in by_method.items():
        if not all(all(row.get(column, "") != "" for column in METRICS.values()) for row in items):
            continue
        ipc = items[0].get("ipc", "")
        receivers = tuple(sorted(row.get("receiver_agent", row.get("agent_id", "")) for row in items))
        summary = {
            "run_name": path.name,
            "path": path,
            "family": family_name(path.name),
            "method": method,
            "ipc": ipc,
            "receiver_scope": ",".join(receivers),
            "num_receivers": len(items),
        }
        for metric, column in METRICS.items():
            summary[metric] = statistics.fmean(float(row[column]) for row in items)
        summaries.append(summary)
    return summaries


def select_best(items: list[dict]) -> dict:
    max_global = max(item["global"] for item in items)
    near = [item for item in items if item["global"] >= max_global - 0.5]
    return max(near, key=lambda item: (item["new"], -item["forgetting"], item["global"]))


def directory_bytes(path: Path) -> int:
    result = subprocess.run(
        ["du", "-sb", str(path)], capture_output=True, text=True, check=True
    )
    return int(result.stdout.split()[0])


def human_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return str(value)


def estimate_reclaim(candidates: list[dict]) -> tuple[int, int, int]:
    """Estimate physical blocks reclaimed when every listed directory is removed."""
    candidate_links = defaultdict(int)
    inode_info = {}
    for row in candidates:
        root = ROOT / row["path"]
        for path in root.rglob("*"):
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(info.st_mode):
                continue
            key = (info.st_dev, info.st_ino)
            candidate_links[key] += 1
            inode_info[key] = (info.st_nlink, info.st_blocks * 512)

    physical = sum(blocks for _, blocks in inode_info.values())
    reclaimable = sum(
        blocks
        for key, (link_count, blocks) in inode_info.items()
        if candidate_links[key] == link_count
    )
    externally_linked = physical - reclaimable
    return physical, reclaimable, externally_linked


def config_run_names() -> dict[str, list[str]]:
    mapping = defaultdict(list)
    for path in sorted((ROOT / "configs").rglob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        run_name = data.get("project", {}).get("run_name")
        if run_name:
            mapping[str(run_name)].append(str(path.relative_to(ROOT)))
    return mapping


def main() -> None:
    registry_names = protected_registry_runs()
    summaries = []
    tuning_dirs = []
    for path in sorted(item for item in OUTPUTS.iterdir() if item.is_dir()):
        if is_protected(path.name, registry_names) or not CANDIDATE_PATTERN.search(path.name):
            continue
        tuning_dirs.append(path)
        summaries.extend(summarize_run(path))

    groups = defaultdict(list)
    for item in summaries:
        key = (item["family"], item["method"], item["ipc"], item["receiver_scope"])
        groups[key].append(item)
    best_names = {select_best(items)["run_name"] for items in groups.values()}

    candidates = []
    for path in tuning_dirs:
        if path.name in best_names:
            continue
        candidates.append(
            {
                "run_name": path.name,
                "path": str(path.relative_to(ROOT)),
                "bytes": directory_bytes(path),
                "reason": "non-best tuning/smoke/intermediate run; review before deleting artifacts",
            }
        )
    candidates.sort(key=lambda row: row["bytes"], reverse=True)

    configs = config_run_names()
    candidate_names = {row["run_name"] for row in candidates}
    config_candidates = [
        {"run_name": run_name, "config": config}
        for run_name in sorted(candidate_names)
        for config in configs.get(run_name, [])
    ]

    operations = ROOT / "docs" / "operations"
    operations.mkdir(parents=True, exist_ok=True)
    csv_path = operations / f"deletion_candidates_{date.today().isoformat()}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["run_name", "path", "bytes", "reason"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(candidates)
    config_path = operations / f"config_candidates_{date.today().isoformat()}.csv"
    with config_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["run_name", "config"], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(config_candidates)

    best_path = ROOT / "experiments" / "generated" / "tuning_best_by_family.csv"
    fields = [
        "family", "run_name", "config", "method", "ipc", "receiver_scope", "num_receivers",
        "global", "new", "expert", "forgetting",
    ]
    with best_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        for key in sorted(groups):
            selected = select_best(groups[key])
            config_paths = configs.get(selected["run_name"], [])
            if not config_paths:
                snapshot = OUTPUTS / selected["run_name"] / "config" / "main.yaml"
                if snapshot.is_file():
                    config_paths = [str(snapshot.relative_to(ROOT))]
            writer.writerow({**selected, "config": "|".join(config_paths)})

    total_bytes = sum(row["bytes"] for row in candidates)
    physical_bytes, reclaimable_bytes, externally_linked_bytes = estimate_reclaim(candidates)
    deleted_config_path = operations / f"deleted_configs_{date.today().isoformat()}.csv"
    deleted_config_count = 0
    if deleted_config_path.is_file():
        deleted_config_count = max(
            0, len(deleted_config_path.read_text(encoding="utf-8").splitlines()) - 1
        )
    report = [
        f"# Cleanup Candidates ({date.today().isoformat()})",
        "",
        "This is a review-only inventory. No output, checkpoint, or packet was deleted.",
        "",
        f"- Candidate output directories: `{len(candidates)}`",
        f"- Candidate logical size: `{human_bytes(total_bytes)}`",
        f"- Candidate unique physical blocks: `{human_bytes(physical_bytes)}`",
        f"- Estimated reclaim if every candidate is approved: `{human_bytes(reclaimable_bytes)}`",
        f"- Candidate blocks retained by hard links outside the list: `{human_bytes(externally_linked_bytes)}`",
        f"- Config files attached to candidates: `{len(config_candidates)}`",
        f"- Obsolete configs already removed under policy 1A: `{deleted_config_count}`",
        f"- Best tuning rows retained: `{len(groups)}`",
        "",
        "Protected families include current one-ResNet, canonical Conv-family/all-ConvNet runs, "
        "hetero4/6, upper bounds, no-VGG and ResNet recipe diagnostics, cross-architecture "
        "diagnostics, registry sources, and each tuning family's selected best run. Non-best "
        "Conv-family tuning/ablation branches may still appear below; the canonical Conv-family "
        "IPC=10/50 evidence never does.",
        "",
        "## Largest Candidates",
        "",
        "| Run | Size | Reason |",
        "|---|---:|---|",
    ]
    for row in candidates[:30]:
        report.append(f"| `{row['run_name']}` | {human_bytes(row['bytes'])} | {row['reason']} |")
    report.extend(
        [
            "",
            "Detailed lists:",
            "",
            f"- `{csv_path.relative_to(ROOT)}`",
            f"- `{config_path.relative_to(ROOT)}`",
            f"- `{deleted_config_path.relative_to(ROOT)}` (completed config cleanup)",
            f"- `{best_path.relative_to(ROOT)}`",
            "",
            "Deletion requires explicit user approval after checking hard-link/source dependencies.",
        ]
    )
    report_path = operations / f"cleanup_candidates_{date.today().isoformat()}.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[cleanup] candidates: {len(candidates)} ({human_bytes(total_bytes)})")
    print(f"[cleanup] estimated reclaim: {human_bytes(reclaimable_bytes)}")
    print(f"[cleanup] report: {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
