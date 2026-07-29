#!/usr/bin/env python3
"""Prepare controlled full-class R10 PCBN weight-sweep configs."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/fullclass_dsdm"
REFERENCE_CONFIG = CONFIG_ROOT / (
    "fullclass_resnet10_standard_modelbest_e0200_ipc10_seed0_recovery_pcbn.yaml"
)
WEIGHTS = (1300, 2100, 3400)


def main() -> None:
    reference = yaml.safe_load(REFERENCE_CONFIG.read_text(encoding="utf-8"))
    pure_run = reference["fullclass_pool"]["pcbn_control"]["pure_run"]
    reference_run = reference["project"]["run_name"]
    for weight in WEIGHTS:
        cfg = copy.deepcopy(reference)
        run_name = f"{pure_run}_pcbn_w{weight}"
        cfg["project"].update(
            {
                "stage": "fullclass_dsdm_model_specific_pcbn_weight_sweep",
                "run_name": run_name,
            }
        )
        cfg["distillation"]["pcbn"]["weight"] = float(weight)
        cfg["fullclass_pool"]["pcbn_control"] = {
            "pure_run": pure_run,
            "reference_pcbn_run": reference_run,
            "weight": float(weight),
            "layers": "all",
            "normalize_layers": True,
            "weight_sweep_role": "conditional_upper_boundary" if weight == 3400 else "primary_bracket",
            "loss_calibration": "experiments/diagnostics/cifar100_r10_pcbn_weight_calibration_seed0.json",
            "gradient_calibration": "experiments/diagnostics/cifar100_r10_pcbn_gradient_diagnostic_seed0.json",
            "protocol": "same_seed_init_guides_dsdm_recipe_only_weight_changes",
        }
        output = CONFIG_ROOT / (
            f"fullclass_resnet10_standard_modelbest_e0200_ipc10_seed0_recovery_pcbn_w{weight}.yaml"
        )
        output.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(output)


if __name__ == "__main__":
    main()
