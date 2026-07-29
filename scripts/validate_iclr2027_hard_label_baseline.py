#!/usr/bin/env python3
"""Fail-closed preflight and completion audit for scaling hard-label baselines."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DSDM_ROOT = ROOT / "DSDM"
if str(DSDM_ROOT) not in sys.path:
    sys.path.insert(0, str(DSDM_ROOT))

from agent_data import (  # noqa: E402
    get_agent_class_split,
    get_agent_model_split,
    get_receiver_dir,
)
from config_adapter import build_dsdm_args_from_config, load_config  # noqa: E402
from output_manager import atomic_write_json  # noqa: E402
from packet_consumer import consume_packet_for_training  # noqa: E402
from packet_integrity import file_sha256  # noqa: E402
from run_social_pipeline import build_receiver_args  # noqa: E402
from selection_methods import _fast_repo_provenance  # noqa: E402
from social_output_manager import read_packet_manifest  # noqa: E402
from train import define_model  # noqa: E402


METHODS = {"heuristic", "fast", "full_real"}
CHECKPOINT_VARIANT = "dkp_sl_v1_ablation_fr1_kd0_sc1"


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _run_dir(config: dict) -> Path:
    root = Path(config["project"].get("output_root", "outputs"))
    if not root.is_absolute():
        root = ROOT / root
    return root / config["project"]["run_name"]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_config(expert: dict, baseline: dict, method: str) -> tuple[dict, dict]:
    _require(method in METHODS, f"unsupported method: {method}")
    class_split = get_agent_class_split(baseline)
    model_split = get_agent_model_split(baseline)
    _require(class_split == get_agent_class_split(expert), "expert/baseline class split differs")
    _require(model_split == get_agent_model_split(expert), "expert/baseline model split differs")
    _require(sorted(class_split) == list(range(len(class_split))), "agent IDs are not contiguous")
    flattened = [class_id for classes in class_split.values() for class_id in classes]
    _require(len(flattened) == 100 and sorted(flattened) == list(range(100)), "classes do not cover 0-99 exactly once")
    _require(len({len(classes) for classes in class_split.values()}) == 1, "class split is unbalanced")
    _require(baseline["scaling_protocol"].get("method") == method, "scaling method differs")
    _require(baseline["communication"].get("use_sender_logits") is False, "hard-label baseline enables sender logits")
    _require(baseline["communication"].get("use_generalist_logits") is False, "hard-label baseline enables generalist logits")
    _require(baseline["logits"].get("enabled") is False, "hard-label baseline logits.enabled is true")
    _require(float(baseline["logits"].get("lambda_kd", -1)) == 0.0, "hard-label baseline KD weight is nonzero")
    receiver = baseline["social_learning"]["receiver"]
    _require(receiver.get("loss_switches") == {"fr": True, "kd": False, "supcon": True}, "receiver loss switches differ")
    _require(receiver.get("dkp_variant") == "ablation_fr1_kd0_sc1", "receiver variant differs")
    _require(int(receiver.get("optimizer_steps", -1)) == 3780, "receiver optimizer steps differ")
    per_class = 500 if method == "full_real" else 10
    _require(int(receiver.get("packet_raw_per_class", -1)) == per_class, "packet budget differs")
    _require(int(baseline["distillation"].get("ipc", -1)) == 10, "configured IPC differs")
    return class_split, model_split


def _validate_experts(expert: dict, baseline: dict, class_split: dict, model_split: dict) -> dict[int, str]:
    source_dir = _run_dir(expert)
    target_dir = _run_dir(baseline)
    manifest_path = target_dir / "provenance/expert_reuse_manifest.json"
    _require(manifest_path.is_file(), f"expert reuse manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {int(item["agent_id"]): item for item in manifest.get("agents", [])}
    _require(sorted(records) == sorted(class_split), "expert reuse manifest is incomplete")
    shas = {}
    for agent in sorted(class_split):
        source = source_dir / f"agents/agent_{agent}/checkpoints/expert_model.pt"
        target = target_dir / f"agents/agent_{agent}/checkpoints/expert_model.pt"
        _require(source.is_file() and target.is_file(), f"agent {agent} expert checkpoint missing")
        source_sha = file_sha256(source)
        _require(file_sha256(target) == source_sha, f"agent {agent} expert SHA differs")
        record = records[agent]
        _require(record["model"] == model_split[agent], f"agent {agent} expert model provenance differs")
        _require(record["class_ids"] == class_split[agent], f"agent {agent} expert classes differ")
        _require(record["artifacts"]["expert_model.pt"]["sha256"] == source_sha, f"agent {agent} manifest SHA differs")
        shas[agent] = source_sha
    return shas


def _validate_packets(baseline_path: Path, baseline: dict, method: str, class_split: dict) -> list[dict]:
    args = build_dsdm_args_from_config(baseline, config_path=baseline_path)
    rows = read_packet_manifest(args, method)
    _require(len(rows) == len(class_split), "packet manifest does not cover every sender")
    reports = []
    fast_provenance = _fast_repo_provenance(args) if method == "fast" else None
    per_class = 500 if method == "full_real" else 10
    for row in rows:
        sender = int(row["sender_agent"])
        _require(sender in class_split, f"unexpected packet sender {sender}")
        path = _resolve(row["packet_path"])
        packet = torch.load(path, map_location="cpu", weights_only=False)
        consumed = consume_packet_for_training(args, path)
        labels = packet["labels"].long()
        counts = Counter(int(value) for value in labels.tolist())
        expected_classes = set(class_split[sender])
        _require(set(counts) == expected_classes, f"sender {sender} packet label set differs")
        _require(all(counts[class_id] == per_class for class_id in expected_classes), f"sender {sender} per-class count differs")
        expected_images = len(expected_classes) * per_class
        _require(len(labels) == expected_images, f"sender {sender} packet image count differs")
        _require(int(consumed["raw_images"]) == expected_images, f"sender {sender} raw image count differs")
        _require(int(consumed["num_images"]) == expected_images, f"sender {sender} decoded image count differs")
        _require(int(consumed.get("sender_logit_bytes", 0)) == 0, f"sender {sender} contains sender-logit bytes")
        _require(int(consumed.get("generalist_logit_bytes", 0)) == 0, f"sender {sender} contains generalist-logit bytes")
        _require(not packet.get("has_sender_logits", False) and "sender_logits" not in packet, f"sender {sender} contains sender logits")
        _require(not packet.get("has_generalist_logits", False) and "generalist_logits" not in packet, f"sender {sender} contains generalist logits")
        _require(packet.get("source") == method, f"sender {sender} packet source differs")
        if fast_provenance:
            meta = packet.get("meta", {})
            for key in ("official_commit", "tracked_patch_present", "tracked_patch_sha256"):
                _require(meta.get(key) == fast_provenance[key], f"sender {sender} FAST {key} differs")
        reports.append(
            {
                "sender": sender,
                "packet_sha256": file_sha256(path),
                "raw_images": expected_images,
                "class_count": len(expected_classes),
            }
        )
    _require(sorted(item["sender"] for item in reports) == sorted(class_split), "packet senders are duplicated or missing")
    return reports


def _validate_complete(
    baseline_path: Path,
    baseline: dict,
    method: str,
    class_split: dict,
    model_split: dict,
    expert_shas: dict[int, str],
) -> list[dict]:
    run_dir = _run_dir(baseline)
    results_path = run_dir / "metrics/social_results.csv"
    _require(results_path.is_file(), f"social results missing: {results_path}")
    with results_path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("packet_method") == method
            and row.get("dkp_variant") == "ablation_fr1_kd0_sc1"
        ]
    latest = {int(row["receiver_agent"]): row for row in rows}
    _require(sorted(latest) == sorted(class_split), "completed receiver rows are incomplete or duplicated")
    reports = []
    per_sender = {agent: len(classes) * (500 if method == "full_real" else 10) for agent, classes in class_split.items()}
    for receiver in sorted(class_split):
        row = latest[receiver]
        expected_external = sum(count for sender, count in per_sender.items() if sender != receiver)
        _require(int(row["optimizer_steps"]) == 3780, f"receiver {receiver} optimizer steps differ")
        _require(int(row["external_comm_images"]) == expected_external, f"receiver {receiver} communication images differ")
        _require(int(row["external_comm_logit_bytes"]) == 0, f"receiver {receiver} sender-logit bytes are nonzero")
        _require(int(row["external_comm_generalist_logit_bytes"]) == 0, f"receiver {receiver} generalist-logit bytes are nonzero")
        _require(row["receiver_model"] == model_split[receiver], f"receiver {receiver} model differs")
        _require(row["receiver_init_checkpoint_sha256"] == expert_shas[receiver], f"receiver {receiver} init SHA differs")
        _require(row["fr_teacher_checkpoint_sha256"] == expert_shas[receiver], f"receiver {receiver} FR SHA differs")
        checkpoint = get_receiver_dir(
            build_dsdm_args_from_config(baseline, config_path=baseline_path), receiver
        ) / "checkpoints" / CHECKPOINT_VARIANT / "after_social.pt"
        _require(checkpoint.is_file(), f"receiver {receiver} final checkpoint missing")
        checkpoint_sha = file_sha256(checkpoint)
        _require(row["after_social_checkpoint_sha256"] == checkpoint_sha, f"receiver {receiver} final SHA differs")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        receiver_args = build_receiver_args(
            baseline,
            str(baseline_path),
            receiver,
            packet_method=method,
            init_mode="expert",
        )
        model = define_model(receiver_args, 100)
        model.load_state_dict(state, strict=True)
        model.eval()
        with torch.no_grad():
            output = model(torch.zeros(2, 3, 32, 32))
        _require(tuple(output.shape) == (2, 100), f"receiver {receiver} output shape differs")
        _require(bool(torch.isfinite(output).all()), f"receiver {receiver} output is non-finite")
        metric_values = {
            key: float(row[key])
            for key in ("acc_global_after", "acc_new_after", "acc_expert_after", "forgetting")
        }
        _require(all(math.isfinite(value) for value in metric_values.values()), f"receiver {receiver} metrics are non-finite")
        own_fraction = len(class_split[receiver]) / 100.0
        expected_global = (
            own_fraction * metric_values["acc_expert_after"]
            + (1.0 - own_fraction) * metric_values["acc_new_after"]
        )
        _require(abs(metric_values["acc_global_after"] - expected_global) < 1e-9, f"receiver {receiver} global metric weighting differs")
        reports.append(
            {
                "receiver": receiver,
                "checkpoint_sha256": checkpoint_sha,
                **metric_values,
            }
        )
        del model, state, output
    return reports


def audit(expert_config: str, baseline_config: str, method: str, phase: str) -> dict:
    expert_path = _resolve(expert_config)
    baseline_path = _resolve(baseline_config)
    expert = load_config(expert_path)
    baseline = load_config(baseline_path)
    class_split, model_split = _validate_config(expert, baseline, method)
    expert_shas = _validate_experts(expert, baseline, class_split, model_split)
    packets = _validate_packets(baseline_path, baseline, method, class_split)
    receivers = (
        _validate_complete(baseline_path, baseline, method, class_split, model_split, expert_shas)
        if phase == "complete"
        else []
    )
    return {
        "status": "passed",
        "phase": phase,
        "method": method,
        "run_name": baseline["project"]["run_name"],
        "agent_count": len(class_split),
        "seed": int(baseline["runtime"]["seed"]),
        "unique_sender_images": sum(item["raw_images"] for item in packets),
        "sender_logit_bytes": 0,
        "packets": packets,
        "receivers": receivers,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-config", required=True)
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--phase", choices=("preflight", "complete"), default="preflight")
    parser.add_argument("--report-json", default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    cli = parse_args(argv)
    try:
        report = audit(
            cli.expert_config,
            cli.baseline_config,
            cli.method,
            cli.phase,
        )
        if cli.report_json:
            atomic_write_json(report, _resolve(cli.report_json))
    except Exception as error:
        print(f"[hard-label-baseline-audit] FAILED: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
