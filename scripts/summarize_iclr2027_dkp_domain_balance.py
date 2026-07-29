#!/usr/bin/env python3
"""Validate and summarize the fixed-step seed0 DKP domain-balance diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_adapter import load_config  # noqa: E402
from output_manager import atomic_write_json  # noqa: E402
from packet_integrity import file_sha256  # noqa: E402
from scripts.prepare_iclr2027_dkp_domain_balance import (  # noqa: E402
    BASE_CONFIG,
    CONDITIONS,
    EXPECTED_LOSSES,
    EXPECTED_STEP_MILESTONES,
    EXPECTED_STEPS,
    run_name,
    validate_config,
)


CONFIG_DIR = ROOT / "configs/iclr2027"
EXPERT_RUN = "cifar100_5agent20cls_dkp_cosine_experts_seed0_v1"
METRICS = {
    "global": "acc_global_after",
    "new": "acc_new_after",
    "expert": "acc_expert_after",
}
LOSSES = (
    "loss",
    "loss_cls",
    "loss_ce_local",
    "loss_ce_local_real",
    "loss_ce_local_packet",
    "loss_ce_external",
    "loss_fr",
    "loss_kd",
    "loss_sc",
)


class DomainBalanceSummaryError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise DomainBalanceSummaryError(message)


def _resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _finite(value, context):
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise DomainBalanceSummaryError(f"{context} is not numeric: {value!r}") from exc
    _require(math.isfinite(parsed), f"{context} is non-finite")
    return parsed


def _integer(value, context):
    parsed = int(value)
    _require(str(parsed) == str(value).strip(), f"{context} is not canonical integer text")
    return parsed


def _expert_shas():
    root = ROOT / "outputs" / EXPERT_RUN / "agents"
    result = {}
    for receiver_id in range(5):
        path = root / f"agent_{receiver_id}/checkpoints/expert_model.pt"
        _require(path.is_file(), f"expert checkpoint is missing: {path}")
        result[receiver_id] = file_sha256(path)
    return result


def _result_rows(config, receiver_ids):
    run_dir = _resolve(config["project"].get("output_root", "outputs")) / config["project"]["run_name"]
    path = run_dir / "metrics/social_results.csv"
    _require(path.is_file(), f"result CSV is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        required = {
            "run_name", "protocol", "dkp_variant", "method", "receiver_agent",
            "receiver_model", "local_ce_source", "self_real_images",
            "self_packet_raw_images", "self_packet_decoded_images",
            "self_packet_path", "self_packet_sha256", "external_comm_images",
            "external_comm_logit_bytes", "optimizer_steps", "target_optimizer_steps",
            "scheduler_unit", "scheduler_step_milestones", "prototype_initialized_classes",
            "classifier_type", "prototype_init_mode", "checkpoint_retention",
            "after_social_checkpoint_sha256", "receiver_seed",
            "receiver_init_checkpoint_sha256", "fr_teacher_checkpoint_sha256",
            "receiver_provenance_path", "receiver_provenance_sha256", "training_seconds",
            *METRICS.values(), *LOSSES,
        }
        missing = sorted(required - set(fields))
        _require(not missing, f"result CSV is missing fields {missing}: {path}")
        rows = [row for row in reader if int(row["receiver_agent"]) in receiver_ids]
    actual_ids = sorted(int(row["receiver_agent"]) for row in rows)
    _require(actual_ids == sorted(receiver_ids), f"receiver coverage mismatch in {path}: {actual_ids}")
    _require(len(actual_ids) == len(set(actual_ids)), f"duplicate receiver rows in {path}")
    return path, sorted(rows, key=lambda row: int(row["receiver_agent"]))


def _validate_checkpoint(provenance, expected_sha, context):
    artifacts = provenance.get("checkpoint_artifacts", {})
    _require(set(artifacts) == {"after_social"}, f"{context}: final_only artifacts drifted")
    path = _resolve(artifacts["after_social"]["path"])
    _require(path.is_file(), f"{context}: final checkpoint is missing")
    actual_sha = file_sha256(path)
    _require(actual_sha == expected_sha, f"{context}: final checkpoint SHA mismatch")
    state = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(state, dict) and state, f"{context}: checkpoint state is empty")
    _require(
        all(torch.is_tensor(value) and bool(torch.isfinite(value).all()) for value in state.values()),
        f"{context}: checkpoint contains non-finite or non-tensor state",
    )


def _validate_row(row, condition, receiver_id, expert_sha):
    context = f"{condition}/receiver_{receiver_id}"
    source = CONDITIONS[condition]
    uses_self_packet = source != "real"
    _require(row["run_name"] == run_name(condition), f"{context}: run name mismatch")
    _require(row["protocol"] == "dkp_sl_v1", f"{context}: protocol mismatch")
    _require(row["dkp_variant"] == "ablation_fr1_kd0_sc1", f"{context}: variant mismatch")
    _require(row["method"] == "DKP_SL_ABLATION", f"{context}: method mismatch")
    _require(row["local_ce_source"] == source, f"{context}: local CE source mismatch")
    _require(_integer(row["self_real_images"], context) == 10_000, f"{context}: real count mismatch")
    expected_raw = 200 if uses_self_packet else 0
    expected_decoded = 800 if uses_self_packet else 0
    _require(_integer(row["self_packet_raw_images"], context) == expected_raw, f"{context}: self raw count mismatch")
    _require(_integer(row["self_packet_decoded_images"], context) == expected_decoded, f"{context}: self decoded count mismatch")
    if uses_self_packet:
        packet_path = _resolve(row["self_packet_path"])
        _require(packet_path.is_file(), f"{context}: self packet is missing")
        _require(file_sha256(packet_path) == row["self_packet_sha256"], f"{context}: self packet SHA mismatch")
    else:
        _require(not row["self_packet_path"] and not row["self_packet_sha256"], f"{context}: S unexpectedly loaded self packet")
    _require(_integer(row["external_comm_images"], context) == 800, f"{context}: communication image count mismatch")
    _require(_integer(row["external_comm_logit_bytes"], context) == 0, f"{context}: logits were communicated")
    _require(_integer(row["optimizer_steps"], context) == EXPECTED_STEPS, f"{context}: actual steps mismatch")
    _require(_integer(row["target_optimizer_steps"], context) == EXPECTED_STEPS, f"{context}: target steps mismatch")
    _require(row["scheduler_unit"] == "optimizer_step", f"{context}: scheduler unit mismatch")
    _require(row["scheduler_step_milestones"] == "2457;3213", f"{context}: scheduler milestones mismatch")
    _require(_integer(row["prototype_initialized_classes"], context) == 80, f"{context}: prototype count mismatch")
    _require(row["classifier_type"] == "cosine", f"{context}: classifier mismatch")
    _require(row["prototype_init_mode"] == "cosine_unit_weight_rows", f"{context}: prototype mode mismatch")
    _require(row["checkpoint_retention"] == "final_only", f"{context}: checkpoint retention mismatch")
    _require(_integer(row["receiver_seed"], context) == 100_000 + receiver_id, f"{context}: receiver seed mismatch")
    _require(row["receiver_init_checkpoint_sha256"] == expert_sha, f"{context}: init expert SHA mismatch")
    _require(row["fr_teacher_checkpoint_sha256"] == expert_sha, f"{context}: FR expert SHA mismatch")

    losses = {name: _finite(row[name], f"{context}/{name}") for name in LOSSES}
    _require(abs(losses["loss_kd"]) <= 1e-12, f"{context}: KD loss must be zero")
    expected_total = losses["loss_cls"] + 0.2 * losses["loss_fr"] + 0.1 * losses["loss_sc"]
    _require(abs(losses["loss"] - expected_total) <= 2e-6, f"{context}: total loss identity mismatch")
    if source == "real":
        _require(abs(losses["loss_ce_local_packet"]) <= 1e-12, f"{context}: S used self-packet CE")
        expected_local = losses["loss_ce_local_real"]
    elif source == "packet":
        _require(abs(losses["loss_ce_local_real"]) <= 1e-12, f"{context}: U used real CE")
        expected_local = losses["loss_ce_local_packet"]
    else:
        expected_local = 0.5 * (losses["loss_ce_local_real"] + losses["loss_ce_local_packet"])
    _require(abs(losses["loss_ce_local"] - expected_local) <= 2e-6, f"{context}: local CE identity mismatch")

    provenance_path = _resolve(row["receiver_provenance_path"])
    _require(provenance_path.is_file(), f"{context}: provenance is missing")
    _require(file_sha256(provenance_path) == row["receiver_provenance_sha256"], f"{context}: provenance SHA mismatch")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    _require(provenance.get("local_ce_source") == source, f"{context}: provenance source mismatch")
    _require(provenance.get("loss_switches") == EXPECTED_LOSSES, f"{context}: loss switches mismatch")
    schedule = provenance.get("training_schedule", {})
    _require(schedule.get("target_optimizer_steps") == EXPECTED_STEPS, f"{context}: provenance target mismatch")
    _require(schedule.get("actual_optimizer_steps") == EXPECTED_STEPS, f"{context}: provenance steps mismatch")
    _require(schedule.get("scheduler_step_milestones") == EXPECTED_STEP_MILESTONES, f"{context}: provenance milestones mismatch")
    local_data = provenance.get("local_data", {})
    _require(local_data.get("counted_as_external_communication") is False, f"{context}: self packet counted as communication")
    _require(local_data.get("used_for_kd") is False, f"{context}: self packet entered KD")
    _require(local_data.get("used_for_supcon") is False, f"{context}: self packet entered SupCon")
    _require(int(local_data.get("self_packet_raw_images", -1)) == expected_raw, f"{context}: provenance self raw mismatch")
    _validate_checkpoint(provenance, row["after_social_checkpoint_sha256"], context)

    metrics = {name: _finite(row[field], f"{context}/{name}") for name, field in METRICS.items()}
    _require(all(0.0 <= value <= 100.0 for value in metrics.values()), f"{context}: metric outside [0,100]")
    training_seconds = _finite(row["training_seconds"], f"{context}/training_seconds")
    _require(training_seconds > 0.0, f"{context}: training time must be positive")
    return {
        "receiver_agent": receiver_id,
        "receiver_model": row["receiver_model"],
        "metrics": metrics,
        "loss_means": losses,
        "training_seconds": training_seconds,
        "checkpoint_sha256": row["after_social_checkpoint_sha256"],
    }


def _reference(path):
    path = ROOT / path
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 5:
        return None
    means = {
        name: statistics.fmean(float(row[field]) for row in rows)
        for name, field in METRICS.items()
    }
    return {"path": str(path.resolve()), "sha256": file_sha256(path), "metrics_mean": means}


def build_summary(config_dir, receiver_ids):
    base = load_config(BASE_CONFIG)
    expert_shas = _expert_shas()
    conditions = {}
    for condition in CONDITIONS:
        config_path = Path(config_dir) / f"{run_name(condition)}.yaml"
        config = load_config(config_path)
        validate_config(config, condition, base)
        result_path, rows = _result_rows(config, receiver_ids)
        parsed = [
            _validate_row(row, condition, int(row["receiver_agent"]), expert_shas[int(row["receiver_agent"])])
            for row in rows
        ]
        conditions[condition] = {
            "run_name": run_name(condition),
            "local_ce_source": CONDITIONS[condition],
            "config": str(config_path.resolve()),
            "config_sha256": file_sha256(config_path),
            "social_results_csv": str(result_path.resolve()),
            "social_results_sha256": file_sha256(result_path),
            "receivers": parsed,
            "metrics_mean": {
                name: statistics.fmean(row["metrics"][name] for row in parsed)
                for name in METRICS
            },
            "training_seconds_total": sum(row["training_seconds"] for row in parsed),
        }
    complete = receiver_ids == list(range(5))
    return {
        "status": "complete_diagnostic" if complete else "partial_mechanism_gate",
        "formal_result": False,
        "paper_eligible": False,
        "complete_five_receiver_seed": complete,
        "receiver_ids": receiver_ids,
        "dataset": "cifar100",
        "agents": 5,
        "classes_per_agent": 20,
        "ipc": 10,
        "fixed_training": {
            "optimizer_steps": EXPECTED_STEPS,
            "scheduler_unit": "optimizer_step",
            "scheduler_step_milestones": EXPECTED_STEP_MILESTONES,
            "loss_switches": EXPECTED_LOSSES,
            "lambda_fr": 0.2,
            "lambda_sc": 0.1,
        },
        "conditions": conditions,
        "ranking_by_diagnostic_global": sorted(
            CONDITIONS,
            key=lambda condition: conditions[condition]["metrics_mean"]["global"],
            reverse=True,
        ),
        "selection_note": "diagnostic ranking only; all three conditions continue to five receivers",
        "references": {
            "pat5_same_split": _reference("outputs/cifar100_pat5agent_20cls_ours_seed0_ipc10/metrics/social_results.csv"),
            "current_full": _reference("outputs/cifar100_5agent20cls_dkp_sl_full_ipc10_seed0_v2/metrics/social_results.csv"),
            "current_fr_supcon": _reference("outputs/cifar100_5agent20cls_dkp_ablation_fr1_kd0_sc1_ipc10_seed0_v1/metrics/social_results.csv"),
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default=str(CONFIG_DIR))
    parser.add_argument("--receiver-ids", default="0,1,2,3,4")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    cli = parse_args(argv)
    try:
        receiver_ids = sorted({int(value) for value in cli.receiver_ids.split(",")})
        _require(receiver_ids and all(0 <= value < 5 for value in receiver_ids), "invalid receiver ids")
        output = _resolve(cli.output_json)
        _require(not output.exists(), f"refusing to overwrite summary: {output}")
        summary = build_summary(cli.config_dir, receiver_ids)
        atomic_write_json(summary, output)
    except Exception as exc:
        print(f"ICLR2027 domain-balance summary FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": summary["status"], "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
