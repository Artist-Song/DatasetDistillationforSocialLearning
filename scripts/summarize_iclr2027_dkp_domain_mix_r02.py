#!/usr/bin/env python3
"""Build the strict combined five-point curve summary after r02 completes."""

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
from scripts.summarize_iclr2027_dkp_domain_mix import build_summary  # noqa: E402


class R02SummaryError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise R02SummaryError(message)


def _resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def combined_summary(config_dir):
    validate_config(mix_protocol._load(TARGET_CONFIG))
    old = dict(mix_protocol.MIX_CONDITIONS)
    try:
        mix_protocol.MIX_CONDITIONS[CONDITION] = FRACTION
        summary = build_summary(config_dir, list(range(5)))
    finally:
        mix_protocol.MIX_CONDITIONS.clear()
        mix_protocol.MIX_CONDITIONS.update(old)
    _require(CONDITION in summary["conditions"], "combined summary omitted r02")
    summary["adaptive_refinement"] = {
        "condition": CONDITION,
        "local_ce_real_fraction": FRACTION,
        "selected_after_completed_curve": True,
        "formal_hyperparameter_selection": False,
        "requires_independent_seed_preregistration": True,
    }
    summary["selection_note"] = (
        "single-seed internal diagnostic only; r02 was selected after viewing the completed curve "
        "and is not a preregistered or formal hyperparameter result"
    )
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default=str(ROOT / "configs/iclr2027"))
    parser.add_argument("--output-json", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    cli = parse_args(argv)
    try:
        output = _resolve(cli.output_json)
        _require(not output.exists(), f"refusing to overwrite summary: {output}")
        summary = combined_summary(cli.config_dir)
        atomic_write_json(summary, output)
    except Exception as exc:
        print(f"ICLR2027 r02 summary FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": summary["status"], "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
