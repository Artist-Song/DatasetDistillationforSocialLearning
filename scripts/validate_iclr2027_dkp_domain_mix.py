#!/usr/bin/env python3
"""Fail-closed preflight for the seed0 packet-heavy local-CE fraction curve."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_adapter import load_config  # noqa: E402
from output_manager import atomic_write_json  # noqa: E402
from packet_integrity import file_sha256  # noqa: E402
from scripts.prepare_iclr2027_dkp_domain_mix import (  # noqa: E402
    BASE_CONFIG,
    EXPECTED_LOSSES,
    EXPECTED_STEP_MILESTONES,
    EXPECTED_STEPS,
    MIX_CONDITIONS,
    run_name,
    validate_config,
)
from scripts.validate_iclr2027_dkp_domain_balance import (  # noqa: E402
    EXPERT_RUN,
    _source_experts,
    _validate_packets,
    _validate_receivers,
    _validate_reused_experts,
)


CONFIG_DIR = ROOT / "configs/iclr2027"


class DomainMixPreflightError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise DomainMixPreflightError(message)


def _resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def run_preflight(config_dir=CONFIG_DIR, device="cuda"):
    config_dir = _resolve(config_dir)
    base = load_config(BASE_CONFIG)
    sources = _source_experts()

    base_rows, reference, base_packets = _validate_packets(
        base,
        Path(BASE_CONFIG),
        sources,
    )
    _require(len(base_rows) == 5, "completed U reference manifest is incomplete")

    conditions = {}
    for condition, fraction in MIX_CONDITIONS.items():
        config_path = config_dir / f"{run_name(condition)}.yaml"
        _require(config_path.is_file(), f"mix config is missing: {config_path}")
        config = load_config(config_path)
        validate_config(config, condition, base)
        _validate_reused_experts(config, sources)
        rows, _, packets = _validate_packets(
            config,
            config_path,
            sources,
            reference,
        )
        receivers = _validate_receivers(config, config_path, rows, sources, device)
        for receiver in receivers:
            receiver["local_ce_real_fraction"] = fraction
        conditions[condition] = {
            "run_name": run_name(condition),
            "local_ce_source": "real_packet_mix",
            "local_ce_real_fraction": fraction,
            "config": str(config_path.resolve()),
            "config_sha256": file_sha256(config_path),
            "loss_switches": EXPECTED_LOSSES,
            "optimizer_steps": EXPECTED_STEPS,
            "scheduler_step_milestones": EXPECTED_STEP_MILESTONES,
            "packets": packets,
            "receivers": receivers,
        }

    return {
        "status": "passed",
        "formal_result": False,
        "paper_eligible": False,
        "protocol": "dkp_sl_v1",
        "expert_source_run": EXPERT_RUN,
        "expert_provenance": sources,
        "packet_reference_run": base["project"]["run_name"],
        "packet_reference": base_packets,
        "conditions": conditions,
        "cross_condition_and_u_reference_packet_tensors_identical": True,
        "sender_logits_disabled": True,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default=str(CONFIG_DIR))
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--report-json", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    cli = parse_args(argv)
    try:
        output = _resolve(cli.report_json)
        _require(not output.exists(), f"refusing to overwrite preflight report: {output}")
        report = run_preflight(cli.config_dir, cli.device)
        atomic_write_json(report, output)
    except Exception as exc:
        print(f"ICLR2027 domain-mix preflight FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "passed", "report": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
