#!/usr/bin/env python3
"""Prepare controlled single-trajectory DSDM pool comparisons."""

from __future__ import annotations

from pathlib import Path

import yaml

try:
    from scripts.prepare_fullclass_dsdm import MODEL_IDS, build_config
except ModuleNotFoundError:
    from prepare_fullclass_dsdm import MODEL_IDS, build_config


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_EPOCHS = list(range(20, 201, 20))
TRAJECTORY_MODELS = ("conv3", "conv4", "alexnet")


def output_path(model_name: str) -> Path:
    return (
        ROOT
        / "configs"
        / "fullclass_dsdm"
        / f"fullclass_{model_name}_trajectorye0020to0200step0020_ipc10_seed0.yaml"
    )


def build_trajectory_config(model_name: str = "conv3") -> dict:
    if model_name not in TRAJECTORY_MODELS:
        raise ValueError(f"Unsupported trajectory model: {model_name}")
    cfg = build_config(model_name)
    run_name = (
        f"cifar100_fullclass_dsdm_{model_name}_"
        "trajectorye0020to0200step0020_ipc10_seed0"
    )
    model_id = MODEL_IDS[model_name]
    guide = cfg["model_pool"]["models"][model_id]["guide_training"]
    guide.update(
        {
            "num_models": len(CHECKPOINT_EPOCHS),
            "max_epochs": 200,
            "snapshot_epochs": [200],
            "selected_epoch": 200,
            "training_style": "dsdm_single_trajectory",
            "pool_design": "single_trajectory_epoch_snapshots",
            "trajectory_count": 1,
            "trajectory_checkpoint_epochs": CHECKPOINT_EPOCHS,
        }
    )
    cfg["project"].update(
        {
            "stage": "fullclass_dsdm_pool_ablation",
            "run_name": run_name,
            "comparability_group": "cifar100_fullclass_dsdm_dsdmguidee0200_seed0",
        }
    )
    cfg["fullclass_pool"].update(
        {
            "guide_protocol": "dsdm_single_trajectory_checkpoints_e20_to_e200_step20",
            "guide_pool_design": "single_trajectory_epoch_snapshots",
            "guide_trajectory_count": 1,
            "guide_checkpoint_epochs": CHECKPOINT_EPOCHS,
            "controlled_against": (
                f"cifar100_fullclass_dsdm_{model_name}_"
                "dsdmguidee0200_ipc10_seed0"
            ),
        }
    )
    return cfg


def main() -> None:
    for model_name in TRAJECTORY_MODELS:
        path = output_path(model_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                build_trajectory_config(model_name),
                handle,
                sort_keys=False,
                allow_unicode=False,
            )
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
