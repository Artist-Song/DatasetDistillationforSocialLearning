#!/usr/bin/env python3
"""Validate completed current-protocol DeSA and homogeneous MASC scaling runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_data import get_agent_class_split, get_agent_model_split  # noqa: E402
from baseline_adapters.communication_accounting import desa_communication_accounting  # noqa: E402
from config_adapter import load_config  # noqa: E402
from output_manager import atomic_write_json  # noqa: E402
from packet_integrity import file_sha256  # noqa: E402


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _read_json(path: Path) -> dict:
    _require(path.is_file(), f"missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict]:
    _require(path.is_file(), f"missing CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _protocol(config_path: Path) -> tuple[dict, dict[int, list[int]], dict[int, str]]:
    config = load_config(config_path)
    classes = get_agent_class_split(config)
    models = get_agent_model_split(config)
    _require(len(classes) in {5, 10, 20}, "external scaling supports 5/10/20 agents")
    _require(sorted(classes) == list(range(len(classes))), "agent IDs are not contiguous")
    flattened = [class_id for values in classes.values() for class_id in values]
    _require(len(flattened) == 100 and sorted(flattened) == list(range(100)), "classes do not cover 0-99 exactly once")
    _require(len({len(values) for values in classes.values()}) == 1, "class split is unbalanced")
    return config, classes, models


def _validate_metrics(rows: list[dict], id_key: str, classes: dict[int, list[int]]) -> list[dict]:
    agents = {int(row[id_key]): row for row in rows if row[id_key] != "avg"}
    _require(sorted(agents) == sorted(classes), "result rows do not cover every agent")
    _require(sum(row[id_key] == "avg" for row in rows) == 1, "average row is missing or duplicated")
    reports = []
    for agent, row in agents.items():
        metrics = {
            name: float(row[name])
            for name in ("acc_global", "acc_new", "acc_expert", "forgetting", "expert_before")
        }
        _require(all(math.isfinite(value) for value in metrics.values()), f"agent {agent} metrics are non-finite")
        own_fraction = len(classes[agent]) / 100.0
        expected_global = own_fraction * metrics["acc_expert"] + (1.0 - own_fraction) * metrics["acc_new"]
        # Separate evaluation batch compositions can move at most one borderline CIFAR-100 prediction.
        _require(abs(metrics["acc_global"] - expected_global) <= 0.011, f"agent {agent} global metric weighting differs")
        _require(abs(metrics["forgetting"] - (metrics["expert_before"] - metrics["acc_expert"])) < 1e-9, f"agent {agent} forgetting differs")
        reports.append({"agent": agent, **metrics})
    return reports


def validate_desa(config_path: Path, output_root: Path, rounds: int) -> dict:
    config, classes, models = _protocol(config_path)
    run_name = str(config["project"]["run_name"])
    run_dir = output_root / run_name
    resolved = _read_json(run_dir / "resolved_protocol.json")
    _require(resolved["method"] == "DeSA-CIL", "DeSA protocol method differs")
    _require(int(resolved["agent_count"]) == len(classes), "DeSA agent count differs")
    _require(int(resolved["rounds"]) == rounds, "DeSA round count differs")
    _require(int(resolved["ipc"]) == 10, "DeSA IPC differs")
    _require(resolved["project_config_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest(), "DeSA config SHA differs")
    _require(resolved["class_split"] == {str(agent): values for agent, values in classes.items()}, "DeSA class split differs")
    _require(resolved["model_split"] == {str(agent): value for agent, value in models.items()}, "DeSA model split differs")

    anchor_payload = torch.load(run_dir / "anchors.pt", map_location="cpu", weights_only=False)
    anchors = anchor_payload["anchors"] if "anchors" in anchor_payload else anchor_payload
    _require(sorted(int(owner) for owner in anchors) == sorted(classes), "DeSA anchor owners differ")
    anchor_counts = {}
    for owner in classes:
        images, labels = anchors[owner]
        expected = len(classes[owner]) * 10
        _require(len(images) == expected and len(labels) == expected, f"DeSA owner {owner} anchor count differs")
        counts = Counter(int(value) for value in labels.tolist())
        _require(set(counts) == set(classes[owner]), f"DeSA owner {owner} labels differ")
        _require(all(counts[class_id] == 10 for class_id in classes[owner]), f"DeSA owner {owner} IPC differs")
        anchor_counts[owner] = expected

    communication = desa_communication_accounting(
        anchor_counts,
        {agent: len(values) for agent, values in classes.items()},
        rounds,
    )
    rows = _read_csv(run_dir / "social_results.csv")
    metrics = _validate_metrics(rows, "receiver", classes)
    by_agent = {int(row["receiver"]): row for row in rows if row["receiver"] != "avg"}
    for receiver, row in by_agent.items():
        _require(row["backbone"] == models[receiver], f"DeSA receiver {receiver} backbone differs")
        _require(int(row["external_comm_images"]) == communication["external_images_per_receiver"][receiver], f"DeSA receiver {receiver} image communication differs")
        _require(int(row["iterative_owner_logit_bytes_per_receiver"]) == communication["iterative_owner_logit_bytes_per_receiver"][receiver], f"DeSA receiver {receiver} logit bytes differ")
        _require(int(row["iterative_owner_logit_bytes_all_agents"]) == communication["iterative_owner_logit_bytes_all_agents"], "DeSA all-agent logit bytes differ")
        _require(row["expert_checkpoint_sha256"] == resolved["expert_checkpoint_sha256"][str(receiver)], f"DeSA receiver {receiver} expert SHA differs")

    checkpoint_path = run_dir / "checkpoint.pt"
    _require(checkpoint_path.is_file(), "DeSA checkpoint is missing")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _require(int(checkpoint["round"]) == rounds, "DeSA checkpoint round differs")
    _require(sorted(int(agent) for agent in checkpoint["models"]) == sorted(classes), "DeSA checkpoint models differ")
    return {
        "status": "passed",
        "method": "DeSA-CIL",
        "run_dir": str(run_dir),
        "agent_count": len(classes),
        "seed": int(config["runtime"]["seed"]),
        "communication": communication,
        "metrics": metrics,
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }


def validate_masc(config_path: Path, output_root: Path) -> dict:
    config, classes, _models = _protocol(config_path)
    source_run = str(config["project"]["run_name"])
    run_dir = output_root / f"{source_run}_masc_homogeneous_10"
    resolved = _read_json(run_dir / "resolved_config.json")
    _require(resolved["source_run_name"] == source_run, "MASC source run differs")
    _require(int(resolved["seed"]) == int(config["runtime"]["seed"]), "MASC seed differs")
    _require(int(resolved["ipc"]) == 10, "MASC IPC differs")
    _require(resolved["budget"] == "10", "MASC budget differs")
    _require(resolved["config_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest(), "MASC config SHA differs")
    _require(resolved["class_split"] == {str(agent): values for agent, values in classes.items()}, "MASC class split differs")

    rows = _read_csv(run_dir / "social_results.csv")
    metrics = _validate_metrics(rows, "agent_id", classes)
    by_agent = {int(row["agent_id"]): row for row in rows if row["agent_id"] != "avg"}
    expected_total_images = 1000
    for agent, row in by_agent.items():
        _require(row["backbone"] == "official-Netwider13-homogeneous", f"MASC agent {agent} is not homogeneous Netwider13")
        _require(int(row["comm_images_per_sender"]) == len(classes[agent]) * 10, f"MASC agent {agent} sender images differ")
        _require(int(row["total_cc_images"]) == expected_total_images, "MASC total CC images differ")
        upload = int(row["expert_upload_bytes_all_agents"])
        downlink = int(row["cc_student_downlink_bytes_all_agents"])
        total = int(row["model_parameter_bytes"])
        _require(upload > 0 and downlink > 0 and total == upload + downlink, "MASC parameter communication differs")

    expert_root = output_root / "expert_sets" / source_run
    for agent, agent_classes in classes.items():
        expert = expert_root / f"agent_{agent}.pt"
        metadata = _read_json(expert.with_suffix(".json"))
        _require(expert.is_file(), f"MASC expert {agent} missing")
        _require(metadata["classes"] == agent_classes, f"MASC expert {agent} classes differ")
        _require((run_dir / f"ra_agent_{agent}.pt").is_file(), f"MASC RA head {agent} missing")
    _require((run_dir / "cc_student.pt").is_file(), "MASC CC student missing")
    return {
        "status": "passed",
        "method": "MASC-complete",
        "run_dir": str(run_dir),
        "agent_count": len(classes),
        "seed": int(config["runtime"]["seed"]),
        "unique_sender_images": expected_total_images,
        "model_parameter_bytes": int(next(iter(by_agent.values()))["model_parameter_bytes"]),
        "metrics": metrics,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("desa", "masc"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--report-json", default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    cli = parse_args(argv)
    config_path = _resolve(cli.config)
    output_root = _resolve(cli.output_root)
    try:
        report = (
            validate_desa(config_path, output_root, cli.rounds)
            if cli.method == "desa"
            else validate_masc(config_path, output_root)
        )
        if cli.report_json:
            atomic_write_json(report, _resolve(cli.report_json))
    except Exception as error:
        print(f"[external-scaling-audit] FAILED: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
