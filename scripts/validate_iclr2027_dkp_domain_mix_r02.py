#!/usr/bin/env python3
"""Run the strict combined U/curve/r02 preflight without mutating completed artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from output_manager import atomic_write_json  # noqa: E402
from scripts import prepare_iclr2027_dkp_domain_mix as mix_protocol  # noqa: E402
from scripts.prepare_iclr2027_dkp_domain_mix_r02 import (  # noqa: E402
    CONDITION,
    FRACTION,
    TARGET_CONFIG,
    validate_config,
)
from scripts.validate_iclr2027_dkp_domain_mix import run_preflight  # noqa: E402


class R02PreflightError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise R02PreflightError(message)


def _resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def combined_preflight(config_dir, device):
    validate_config(mix_protocol._load(TARGET_CONFIG))
    old = dict(mix_protocol.MIX_CONDITIONS)
    try:
        mix_protocol.MIX_CONDITIONS[CONDITION] = FRACTION
        report = run_preflight(config_dir, device)
    finally:
        mix_protocol.MIX_CONDITIONS.clear()
        mix_protocol.MIX_CONDITIONS.update(old)
    report["adaptive_refinement"] = {
        "condition": CONDITION,
        "local_ce_real_fraction": FRACTION,
        "selected_after_completed_curve": True,
        "formal_hyperparameter_selection": False,
    }
    _require(CONDITION in report["conditions"], "combined preflight omitted r02")
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default=str(ROOT / "configs/iclr2027"))
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--report-json", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    cli = parse_args(argv)
    try:
        output = _resolve(cli.report_json)
        _require(not output.exists(), f"refusing to overwrite preflight report: {output}")
        report = combined_preflight(cli.config_dir, cli.device)
        atomic_write_json(report, output)
    except Exception as exc:
        print(f"ICLR2027 r02 preflight FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "passed", "report": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
