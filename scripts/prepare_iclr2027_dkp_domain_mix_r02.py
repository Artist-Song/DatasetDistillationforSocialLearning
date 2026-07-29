#!/usr/bin/env python3
"""Prepare the post-curve r02 DKP domain-mix confirmation config."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_adapter import build_dsdm_args_from_config  # noqa: E402
from scripts.prepare_iclr2027_dkp_domain_mix import (  # noqa: E402
    BASE_CONFIG,
    EXPECTED_LOSSES,
    EXPECTED_STEP_MILESTONES,
    EXPECTED_STEPS,
)


CONDITION = "r02"
FRACTION = 0.02
RUN_NAME = "cifar100_5agent20cls_dkp_domain_mix_r02_steps3780_ipc10_seed0_v1"
TARGET_CONFIG = ROOT / "configs/iclr2027" / f"{RUN_NAME}.yaml"


class R02ConfigError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise R02ConfigError(message)


def _load(path):
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"config is not a mapping: {path}")
    return payload


def build_config(base_path=BASE_CONFIG):
    base = _load(base_path)
    _require(
        base.get("social_learning", {}).get("receiver", {}).get("local_ce_source") == "packet",
        "base must be the completed U packet-only config",
    )
    config = copy.deepcopy(base)
    config["project"].update(
        {
            "run_name": RUN_NAME,
            "stage": "iclr2027_dkp_domain_mix_postcurve_diagnostic",
            "comparability_group": "cifar100_5agent20cls_iclr2027_dkp_domain_balance_seed0_ipc10",
            "paper_eligible": False,
            "protocol_status": "planned_diagnostic",
        }
    )
    receiver = config["social_learning"]["receiver"]
    receiver.update(
        {
            "local_ce_source": "real_packet_mix",
            "local_ce_real_fraction": FRACTION,
            "optimizer_steps": EXPECTED_STEPS,
            "self_packet_batch_size": 64,
            "scheduler_unit": "optimizer_step",
            "scheduler_step_milestones": list(EXPECTED_STEP_MILESTONES),
            "checkpoint_retention": "final_only",
        }
    )
    config.pop("domain_balance_diagnostic", None)
    config["domain_mix_diagnostic"] = {
        "role": "postcurve_adaptive_seed0_internal_confirmation_not_formal_result",
        "condition": CONDITION,
        "local_ce_source": "real_packet_mix",
        "local_ce_real_fraction": FRACTION,
        "target_optimizer_steps": EXPECTED_STEPS,
        "scheduler_unit": "optimizer_step",
        "scheduler_step_milestones": list(EXPECTED_STEP_MILESTONES),
        "losses": dict(EXPECTED_LOSSES),
        "self_packet_counted_as_external_communication": False,
        "all_fractions_cover_all_five_receivers": True,
        "adaptive_after_completed_fraction_curve": True,
        "formal_hyperparameter_selection": False,
    }
    return config


def validate_config(config, base_path=BASE_CONFIG):
    expected = build_config(base_path)
    _require(config == expected, "r02 config differs from the exact generated protocol")
    args = build_dsdm_args_from_config(config)
    _require(args.receiver_local_ce_source == "real_packet_mix", "parsed local CE source mismatch")
    _require(math.isclose(args.receiver_local_ce_real_fraction, FRACTION), "parsed fraction mismatch")
    _require(args.receiver_optimizer_steps == EXPECTED_STEPS, "parsed optimizer steps mismatch")
    _require(args.receiver_scheduler_step_milestones == EXPECTED_STEP_MILESTONES, "parsed milestones mismatch")
    _require(args.dkp_loss_switches == EXPECTED_LOSSES, "parsed loss switches mismatch")
    return args


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default=str(BASE_CONFIG))
    parser.add_argument("--target-config", default=str(TARGET_CONFIG))
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    cli = parse_args(argv)
    config = build_config(cli.base_config)
    validate_config(config, cli.base_config)
    path = Path(cli.target_config)
    if path.exists():
        if _load(path) != config:
            raise FileExistsError(f"refusing to replace different config: {path}")
    elif cli.check:
        raise FileNotFoundError(f"expected config is missing: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(json.dumps({"status": "passed", "config": str(path.resolve())}, indent=2))


if __name__ == "__main__":
    main()
