#!/usr/bin/env python3
"""Prepare clean recovery configs after the model-specific ResNet queue was interrupted."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "fullclass_dsdm"
OUTPUT_ROOT = ROOT / "outputs"

MODELS = ("resnet10_standard", "resnet18_standard")


def build_config(model: str) -> dict:
    source = CONFIG_ROOT / f"fullclass_{model}_modelbest_e0200_ipc10_seed0.yaml"
    with source.open("r", encoding="utf-8") as handle:
        cfg = copy.deepcopy(yaml.safe_load(handle))

    original_run = cfg["project"]["run_name"]
    recovery_run = f"{original_run}_recovery"
    cfg["project"].update(
        {
            "run_name": recovery_run,
            "comparability_group": "cifar100_fullclass_dsdm_model_specific_e0200_seed0",
        }
    )
    cfg["fullclass_pool"]["recovery_reason"] = "clean_restart_after_operator_interrupt"

    if model == "resnet10_standard":
        guide = cfg["model_pool"]["models"][model]["guide_training"]
        guide["source_root"] = str(OUTPUT_ROOT / original_run)
        cfg["fullclass_pool"]["guide_protocol"] = "reuse_byte_identical_interrupted_run_guides"
    return cfg


def main() -> None:
    for model in MODELS:
        config = build_config(model)
        path = CONFIG_ROOT / f"fullclass_{model}_modelbest_e0200_ipc10_seed0_recovery.yaml"
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        print(path)


if __name__ == "__main__":
    main()
