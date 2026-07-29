#!/usr/bin/env python3
"""Prepare the paired KD-on full DKP-SL confirmation at r02."""

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
    EXPECTED_STEP_MILESTONES,
    EXPECTED_STEPS,
)
from scripts.prepare_iclr2027_dkp_domain_mix_r02 import (  # noqa: E402
    FRACTION,
    TARGET_CONFIG as R02_CONFIG,
)


RUN_NAME = "cifar100_5agent20cls_dkp_domain_mix_r02_full_steps3780_ipc10_seed0_v2"
TARGET_CONFIG = ROOT / "configs/iclr2027" / f"{RUN_NAME}.yaml"
EXPECTED_LOSSES = {"fr": True, "kd": True, "supcon": True}


class R02FullConfigError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise R02FullConfigError(message)


def _load(path):
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"config is not a mapping: {path}")
    return payload


def build_config(base_path=R02_CONFIG):
    base = _load(base_path)
    diagnostic = base.get("domain_mix_diagnostic", {})
    _require(diagnostic.get("adaptive_after_completed_fraction_curve") is True, "base is not exact r02")
    _require(math.isclose(float(diagnostic.get("local_ce_real_fraction", -1)), FRACTION), "base fraction drifted")
    config = copy.deepcopy(base)
    config["project"].update(
        {
            "run_name": RUN_NAME,
            "stage": "iclr2027_dkp_domain_mix_r02_full_kd_diagnostic",
            "paper_eligible": False,
            "protocol_status": "planned_diagnostic",
        }
    )
    receiver = config["social_learning"]["receiver"]
    receiver.update(
        {
            "dkp_variant": "full",
            "loss_switches": dict(EXPECTED_LOSSES),
            "local_ce_source": "real_packet_mix",
            "local_ce_real_fraction": FRACTION,
            "optimizer_steps": EXPECTED_STEPS,
            "scheduler_unit": "optimizer_step",
            "scheduler_step_milestones": list(EXPECTED_STEP_MILESTONES),
            "checkpoint_retention": "final_only",
        }
    )
    config["communication"]["use_sender_logits"] = True
    config["logits"].update({"enabled": True, "lambda_kd": 0.6, "temperature": 2.0})
    config["loss_ablation"].update(
        {
            "variant_id": "fr1_kd1_sc1",
            "switches": dict(EXPECTED_LOSSES),
        }
    )
    config["domain_mix_diagnostic"].update(
        {
            "role": "postcurve_adaptive_r02_kd_on_paired_confirmation_not_formal_result",
            "condition": "r02_full",
            "losses": dict(EXPECTED_LOSSES),
            "paired_reference_run": base["project"]["run_name"],
            "paired_difference": "sender_class_kd_enabled",
            "sender_logits_required": True,
            "formal_hyperparameter_selection": False,
        }
    )
    return config


def validate_config(config, base_path=R02_CONFIG):
    expected = build_config(base_path)
    _require(config == expected, "r02 full config differs from the exact generated protocol")
    args = build_dsdm_args_from_config(config)
    _require(args.dkp_variant == "full", "parsed variant mismatch")
    _require(args.dkp_loss_switches == EXPECTED_LOSSES, "parsed loss switches mismatch")
    _require(args.receiver_local_ce_source == "real_packet_mix", "parsed local CE source mismatch")
    _require(math.isclose(args.receiver_local_ce_real_fraction, FRACTION), "parsed fraction mismatch")
    _require(args.receiver_optimizer_steps == EXPECTED_STEPS, "parsed steps mismatch")
    _require(args.receiver_scheduler_step_milestones == EXPECTED_STEP_MILESTONES, "parsed milestones mismatch")
    _require(bool(args.use_sender_logits), "parsed sender-logit switch is off")
    _require(math.isclose(float(config["logits"]["lambda_kd"]), 0.6), "lambda_KD mismatch")
    _require(math.isclose(float(config["logits"]["temperature"]), 2.0), "KD temperature mismatch")
    _require(config["loss_ablation"].get("variant_id") == "fr1_kd1_sc1", "loss-ablation variant mismatch")
    _require(config["loss_ablation"].get("switches") == EXPECTED_LOSSES, "loss-ablation switches mismatch")
    return args


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default=str(R02_CONFIG))
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
