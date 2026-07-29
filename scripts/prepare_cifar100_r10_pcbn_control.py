#!/usr/bin/env python3
"""Derive the R10 PCBN control from the completed pure-DSDM recovery config."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PURE_CONFIG = ROOT / (
    "configs/fullclass_dsdm/"
    "fullclass_resnet10_standard_modelbest_e0200_ipc10_seed0_recovery.yaml"
)
CALIBRATION = ROOT / "experiments/diagnostics/cifar100_r10_pcbn_weight_calibration_seed0.json"
OUTPUT_CONFIG = ROOT / (
    "configs/fullclass_dsdm/"
    "fullclass_resnet10_standard_modelbest_e0200_ipc10_seed0_recovery_pcbn.yaml"
)


def weight_token(weight: float) -> str:
    return f"{weight:g}".replace(".", "p")


def main() -> None:
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    if calibration.get("status") != "passed":
        raise ValueError("R10 PCBN calibration did not pass")
    selection = calibration["selection"]
    if float(selection["target_total_loss_share"]) != 0.075:
        raise ValueError("R10 PCBN calibration target changed unexpectedly")
    weight = float(selection["selected_weight"])

    cfg = copy.deepcopy(yaml.safe_load(PURE_CONFIG.read_text(encoding="utf-8")))
    pure_run = str(cfg["project"]["run_name"])
    run_name = f"{pure_run}_pcbn_w{weight_token(weight)}"
    cfg["project"].update(
        {
            "stage": "fullclass_dsdm_model_specific_pcbn_control",
            "run_name": run_name,
        }
    )
    cfg["distillation"]["pcbn"] = {
        "enabled": True,
        "weight": weight,
        "layers": "all",
        "normalize_layers": True,
    }
    cfg["fullclass_pool"]["pcbn_control"] = {
        "pure_run": pure_run,
        "calibration": str(CALIBRATION.relative_to(ROOT)),
        "weight_selection_rule": selection["rule"],
        "target_total_loss_share": float(selection["target_total_loss_share"]),
        "calibrated_initial_total_loss_share": float(selection["realized_mean_total_loss_share"]),
    }

    OUTPUT_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CONFIG.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(OUTPUT_CONFIG)
    print(run_name)
    print(weight)


if __name__ == "__main__":
    main()
