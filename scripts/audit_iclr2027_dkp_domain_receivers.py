#!/usr/bin/env python3
"""Strict-load every final receiver referenced by a complete domain diagnostic summary."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_data import build_agent_args  # noqa: E402
from config_adapter import load_config  # noqa: E402
from output_manager import atomic_write_json  # noqa: E402
from packet_integrity import file_sha256  # noqa: E402
from scripts.audit_iclr2027_dkp_loss_ablation_final_receivers import (  # noqa: E402
    validate_cosine_receiver_checkpoint,
)


class DomainReceiverAuditError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise DomainReceiverAuditError(message)


def _resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _load_json(path, context):
    path = _resolve(path)
    _require(path.is_file(), f"{context} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DomainReceiverAuditError(f"cannot load {context}: {path}") from exc
    _require(isinstance(payload, dict), f"{context} is not a JSON object")
    return path, payload


def audit_summary(summary_path):
    summary_path, summary = _load_json(summary_path, "domain diagnostic summary")
    _require(summary.get("status") == "complete_diagnostic", "summary is not complete_diagnostic")
    _require(summary.get("formal_result") is False, "summary must remain non-formal")
    _require(summary.get("paper_eligible") is False, "summary must remain paper-ineligible")
    _require(summary.get("complete_five_receiver_seed") is True, "summary lacks complete five-receiver coverage")
    _require(summary.get("receiver_ids") == list(range(5)), "summary receiver ids drifted")
    conditions = summary.get("conditions")
    _require(isinstance(conditions, dict) and conditions, "summary has no conditions")

    condition_reports = {}
    for condition, condition_summary in conditions.items():
        config_path = _resolve(condition_summary.get("config", ""))
        _require(config_path.is_file(), f"{condition}: config is missing")
        _require(file_sha256(config_path) == condition_summary.get("config_sha256"), f"{condition}: config SHA mismatch")
        config = load_config(config_path)
        run_name = config.get("project", {}).get("run_name")
        _require(run_name == condition_summary.get("run_name"), f"{condition}: run name mismatch")
        _require(config.get("project", {}).get("paper_eligible") is False, f"{condition}: config became paper-eligible")
        receivers = condition_summary.get("receivers")
        _require(isinstance(receivers, list) and len(receivers) == 5, f"{condition}: receiver coverage is incomplete")
        by_id = {int(row["receiver_agent"]): row for row in receivers}
        _require(sorted(by_id) == list(range(5)) and len(by_id) == 5, f"{condition}: receiver ids drifted")
        run_dir = _resolve(config["project"].get("output_root", "outputs")) / run_name
        variant = str(config.get("social_learning", {}).get("receiver", {}).get("dkp_variant", ""))
        _require(variant, f"{condition}: receiver variant is missing")
        reports = []
        for receiver_id in range(5):
            row = by_id[receiver_id]
            args = build_agent_args(config, config_path, receiver_id)
            _require(str(args.classifier_type).lower() == "cosine", f"{condition}/receiver_{receiver_id}: non-cosine args")
            _require(int(args.nclass) == 100, f"{condition}/receiver_{receiver_id}: output universe drifted")
            checkpoint = (
                run_dir
                / f"social_learning/receiver_agent_{receiver_id}/checkpoints/"
                f"dkp_sl_v1_{variant}/after_social.pt"
            )
            checkpoint_report = validate_cosine_receiver_checkpoint(
                checkpoint,
                args,
                expected_sha256=row.get("checkpoint_sha256"),
                random_seed=10_000 + receiver_id,
            )
            reports.append(
                {
                    "receiver_agent": receiver_id,
                    "receiver_model": row.get("receiver_model"),
                    "metrics": row.get("metrics"),
                    "checkpoint": checkpoint_report,
                }
            )
        condition_reports[condition] = {
            "run_name": run_name,
            "config": str(config_path),
            "config_sha256": file_sha256(config_path),
            "receivers": reports,
        }

    return {
        "status": "passed",
        "formal_result": False,
        "paper_eligible": False,
        "summary": str(summary_path),
        "summary_sha256": file_sha256(summary_path),
        "condition_count": len(condition_reports),
        "receiver_checkpoint_count": 5 * len(condition_reports),
        "conditions": condition_reports,
        "checks": {
            "strict_state_dict_load": True,
            "finite_state_parameters_buffers_output_features": True,
            "cosine_classifier_unique_and_positive_scale": True,
            "output_shape": [2, 100],
            "get_feature_contract": True,
            "checkpoint_sha256_matches_summary": True,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--report-json", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    cli = parse_args(argv)
    try:
        output = _resolve(cli.report_json)
        _require(not output.exists(), f"refusing to overwrite audit report: {output}")
        report = audit_summary(cli.summary_json)
        atomic_write_json(report, output)
    except Exception as exc:
        print(f"ICLR2027 domain receiver audit FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "passed", "report": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
