#!/usr/bin/env python3
"""Strictly validate and summarize the five-receiver r02 full KD-on run."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_adapter import load_config  # noqa: E402
from output_manager import atomic_write_json  # noqa: E402
from packet_integrity import file_sha256  # noqa: E402
from scripts.prepare_iclr2027_dkp_domain_mix import (  # noqa: E402
    EXPECTED_STEP_MILESTONES,
    EXPECTED_STEPS,
)
from scripts.prepare_iclr2027_dkp_domain_mix_r02 import FRACTION  # noqa: E402
from scripts.prepare_iclr2027_dkp_domain_mix_r02_full import (  # noqa: E402
    RUN_NAME,
    TARGET_CONFIG,
    validate_config,
)
from scripts.summarize_iclr2027_dkp_domain_balance import (  # noqa: E402
    LOSSES,
    METRICS,
    _expert_shas,
    _finite,
    _integer,
    _resolve,
    _result_rows,
    _validate_checkpoint,
)


class R02FullSummaryError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise R02FullSummaryError(message)


def _validate_row(row, receiver_id, expert_sha):
    context = f"r02_full/receiver_{receiver_id}"
    _require(row["run_name"] == RUN_NAME, f"{context}: run name mismatch")
    _require(row["protocol"] == "dkp_sl_v1", f"{context}: protocol mismatch")
    _require(row["dkp_variant"] == "full", f"{context}: variant mismatch")
    _require(row["method"] == "DKP_SL", f"{context}: method mismatch")
    _require(row["local_ce_source"] == "real_packet_mix", f"{context}: local source mismatch")
    _require(math.isclose(_finite(row["local_ce_real_fraction"], context), FRACTION), f"{context}: fraction mismatch")
    _require(_integer(row["self_real_images"], context) == 10_000, f"{context}: real count mismatch")
    _require(_integer(row["self_packet_raw_images"], context) == 200, f"{context}: self raw count mismatch")
    _require(_integer(row["self_packet_decoded_images"], context) == 800, f"{context}: self decoded count mismatch")
    self_packet = _resolve(row["self_packet_path"])
    _require(self_packet.is_file(), f"{context}: self packet missing")
    _require(file_sha256(self_packet) == row["self_packet_sha256"], f"{context}: self packet SHA mismatch")
    _require(_integer(row["external_comm_images"], context) == 800, f"{context}: image count mismatch")
    _require(_integer(row["external_comm_logit_bytes"], context) == 128_000, f"{context}: logit bytes mismatch")
    _require(_integer(row["optimizer_steps"], context) == EXPECTED_STEPS, f"{context}: steps mismatch")
    _require(_integer(row["target_optimizer_steps"], context) == EXPECTED_STEPS, f"{context}: target steps mismatch")
    _require(row["scheduler_unit"] == "optimizer_step", f"{context}: scheduler unit mismatch")
    _require(row["scheduler_step_milestones"] == "2457;3213", f"{context}: milestones mismatch")
    _require(_integer(row["prototype_initialized_classes"], context) == 80, f"{context}: prototype count mismatch")
    _require(row["classifier_type"] == "cosine", f"{context}: classifier mismatch")
    _require(row["prototype_init_mode"] == "cosine_unit_weight_rows", f"{context}: prototype mode mismatch")
    _require(row["checkpoint_retention"] == "final_only", f"{context}: retention mismatch")
    _require(_integer(row["receiver_seed"], context) == 100_000 + receiver_id, f"{context}: seed mismatch")
    _require(row["receiver_init_checkpoint_sha256"] == expert_sha, f"{context}: init SHA mismatch")
    _require(row["fr_teacher_checkpoint_sha256"] == expert_sha, f"{context}: FR SHA mismatch")

    losses = {name: _finite(row[name], f"{context}/{name}") for name in LOSSES}
    _require(losses["loss_kd"] > 0.0, f"{context}: KD loss is not active")
    expected_local = FRACTION * losses["loss_ce_local_real"] + (1.0 - FRACTION) * losses["loss_ce_local_packet"]
    _require(abs(losses["loss_ce_local"] - expected_local) <= 2e-6, f"{context}: local CE identity mismatch")
    expected_cls = 0.2 * losses["loss_ce_local"] + 0.8 * losses["loss_ce_external"]
    _require(abs(losses["loss_cls"] - expected_cls) <= 2e-6, f"{context}: class loss identity mismatch")
    expected_total = losses["loss_cls"] + 0.2 * losses["loss_fr"] + 0.6 * losses["loss_kd"] + 0.1 * losses["loss_sc"]
    _require(abs(losses["loss"] - expected_total) <= 2e-6, f"{context}: total loss identity mismatch")
    _require(abs(_finite(row["loss_sender_kd"], context) - losses["loss_kd"]) <= 1e-9, f"{context}: sender KD identity mismatch")

    provenance_path = _resolve(row["receiver_provenance_path"])
    _require(provenance_path.is_file(), f"{context}: provenance missing")
    _require(file_sha256(provenance_path) == row["receiver_provenance_sha256"], f"{context}: provenance SHA mismatch")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    _require(provenance.get("dkp_variant") == "full", f"{context}: provenance variant mismatch")
    _require(provenance.get("loss_switches") == {"fr": True, "kd": True, "supcon": True}, f"{context}: switches mismatch")
    _require(math.isclose(float(provenance.get("local_ce_real_fraction", -1)), FRACTION), f"{context}: provenance fraction mismatch")
    schedule = provenance.get("training_schedule", {})
    _require(schedule.get("actual_optimizer_steps") == EXPECTED_STEPS, f"{context}: provenance steps mismatch")
    _require(schedule.get("scheduler_step_milestones") == EXPECTED_STEP_MILESTONES, f"{context}: provenance milestones mismatch")
    local_data = provenance.get("local_data", {})
    _require(local_data.get("counted_as_external_communication") is False, f"{context}: self packet communication leak")
    _require(local_data.get("used_for_kd") is False, f"{context}: self packet entered KD")
    _require(local_data.get("used_for_supcon") is False, f"{context}: self packet entered SupCon")
    _validate_checkpoint(provenance, row["after_social_checkpoint_sha256"], context)

    metrics = {name: _finite(row[field], f"{context}/{name}") for name, field in METRICS.items()}
    _require(all(0.0 <= value <= 100.0 for value in metrics.values()), f"{context}: metric outside [0,100]")
    seconds = _finite(row["training_seconds"], f"{context}/seconds")
    _require(seconds > 0.0, f"{context}: training time invalid")
    return {
        "receiver_agent": receiver_id,
        "receiver_model": row["receiver_model"],
        "metrics": metrics,
        "loss_means": losses,
        "training_seconds": seconds,
        "checkpoint_sha256": row["after_social_checkpoint_sha256"],
    }


def build_summary():
    config = load_config(TARGET_CONFIG)
    validate_config(config)
    result_path, rows = _result_rows(config, list(range(5)))
    expert_shas = _expert_shas()
    parsed = [_validate_row(row, int(row["receiver_agent"]), expert_shas[int(row["receiver_agent"])]) for row in rows]
    reference_path = ROOT / "outputs/diagnostics/iclr2027_cifar100_5agent20cls_ipc10_seed0_domain_mix_with_r02_v1.json"
    _require(reference_path.is_file(), "paired KD-off summary is missing")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))["conditions"]["r02"]["metrics_mean"]
    means = {name: statistics.fmean(row["metrics"][name] for row in parsed) for name in METRICS}
    condition = {
        "run_name": RUN_NAME,
        "local_ce_source": "real_packet_mix",
        "local_ce_real_fraction": FRACTION,
        "config": str(Path(TARGET_CONFIG).resolve()),
        "config_sha256": file_sha256(TARGET_CONFIG),
        "social_results_csv": str(result_path.resolve()),
        "social_results_sha256": file_sha256(result_path),
        "receivers": parsed,
        "metrics_mean": means,
        "training_seconds_total": sum(row["training_seconds"] for row in parsed),
    }
    return {
        "status": "complete_diagnostic",
        "formal_result": False,
        "paper_eligible": False,
        "complete_five_receiver_seed": True,
        "receiver_ids": list(range(5)),
        "condition": "r02_full",
        "run_name": RUN_NAME,
        "conditions": {"r02_full": condition},
        "metrics_mean": means,
        "paired_r02_kd_off_metrics_mean": reference,
        "delta_full_minus_kd_off": {name: means[name] - reference[name] for name in METRICS},
        "training_seconds_total": condition["training_seconds_total"],
        "optimizer_steps_per_receiver": EXPECTED_STEPS,
        "raw_external_images_per_receiver": 800,
        "transmitted_logit_bytes_per_receiver": 128_000,
        "adaptive_after_completed_seed0_curve": True,
        "requires_independent_seed_preregistration": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    cli = parse_args(argv)
    try:
        output = _resolve(cli.output_json)
        _require(not output.exists(), f"refusing to overwrite summary: {output}")
        summary = build_summary()
        atomic_write_json(summary, output)
    except Exception as exc:
        print(f"ICLR2027 r02 full summary FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": summary["status"], "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
