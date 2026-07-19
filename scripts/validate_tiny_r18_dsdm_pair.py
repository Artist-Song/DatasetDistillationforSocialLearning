#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DSDM_ROOT = ROOT / "DSDM"
for path in (ROOT, DSDM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent_data import build_agent_args, get_agent_train_dataset, get_test_dataset
from config_adapter import load_config
from evaluation_schedule import resolve_evaluation_iterations
from pcbn import PCBNRegularizer
from train import define_model


EXPECTED_SCHEDULE = [100, 500, 1000, 2000, 3000, 5000, 7500, 10000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight the paired Tiny-ImageNet ResNet-18 DSDM runs")
    parser.add_argument("--pure", default="configs/tinyimagenet_r18_agent0_dsdm_ipc10_seed0.yaml")
    parser.add_argument("--pcbn", default="configs/tinyimagenet_r18_agent0_dsdm_pcbn_ipc10_seed0.yaml")
    return parser.parse_args()


def comparison_payload(cfg: dict) -> dict:
    payload = copy.deepcopy(cfg)
    payload["project"].pop("run_name", None)
    payload["project"].pop("stage", None)
    payload["distillation"].pop("pcbn", None)
    return payload


def main() -> None:
    cli = parse_args()
    pure_path = (ROOT / cli.pure).resolve()
    pcbn_path = (ROOT / cli.pcbn).resolve()
    pure_cfg = load_config(pure_path)
    pcbn_cfg = load_config(pcbn_path)
    if comparison_payload(pure_cfg) != comparison_payload(pcbn_cfg):
        raise ValueError("Paired configs differ outside project identity and PCBN settings")

    pure_args = build_agent_args(pure_cfg, pure_path, 0)
    pcbn_args = build_agent_args(pcbn_cfg, pcbn_path, 0)
    expected_classes = list(range(50))
    for name, args in (("pure", pure_args), ("pcbn", pcbn_args)):
        if args.dataset != "tinyimagenet" or args.nclass != 200 or args.size != 64:
            raise ValueError(f"{name}: invalid Tiny-ImageNet dimensions")
        if args.active_class_ids != expected_classes:
            raise ValueError(f"{name}: agent 0 must preserve global labels 0-49")
        if (args.idx_from, args.idx_to) != (5, -1):
            raise ValueError(f"{name}: ResNet-18 must use penultimate feature index 5")
        if args.pretrained_model_number != 10 or args.pretrained_epochs != 100:
            raise ValueError(
                f"{name}: expected 10 shared guides x 100 epochs, found "
                f"{args.pretrained_model_number} x {args.pretrained_epochs}"
            )
        schedule = resolve_evaluation_iterations(args.niter, args.evaluate_iterations, args.evaluate_iter)
        if schedule != EXPECTED_SCHEDULE:
            raise ValueError(f"{name}: unexpected evaluation schedule: {schedule}")

    integrity_path = (ROOT / pure_cfg["dataset"]["integrity_report"]).resolve()
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    if not integrity.get("passed"):
        raise ValueError(f"Tiny-ImageNet integrity gate failed: {integrity_path}")

    train_dataset = get_agent_train_dataset(pure_args, 0, normalize=False)
    val_dataset = get_test_dataset(pure_args)
    if len(train_dataset) != 25_000 or len(val_dataset) != 9_993:
        raise ValueError(f"Unexpected clean split sizes: train={len(train_dataset)} val={len(val_dataset)}")
    if set(train_dataset.targets) != set(expected_classes):
        raise ValueError("Sender-local training labels were remapped or filtered incorrectly")

    model = define_model(pure_args, 200).eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    with torch.no_grad():
        output_shape = tuple(model(torch.randn(2, 3, 64, 64)).shape)
        feature_shape = tuple(model.get_feature(torch.randn(2, 3, 64, 64), 5, -1)[0].shape)
    if parameter_count != 11_271_432 or output_shape != (2, 200) or feature_shape != (2, 512):
        raise ValueError(
            f"Unexpected ResNet-18 contract: params={parameter_count} output={output_shape} feature={feature_shape}"
        )

    pure_regularizer = PCBNRegularizer(pure_args)
    pcbn_regularizer = PCBNRegularizer(pcbn_args)
    if pure_regularizer.enabled or not pcbn_regularizer.enabled:
        raise ValueError("PCBN enable flags are not a pure off/on pair")
    hooked = pcbn_regularizer.attach(model)
    pcbn_regularizer.close()
    if hooked != 20:
        raise ValueError(f"Expected 20 ResNet-18 BN hooks, found {hooked}")

    print(
        json.dumps(
            {
                "status": "passed",
                "train_images_agent0": len(train_dataset),
                "clean_validation_images_global": len(val_dataset),
                "global_labels": [0, 49],
                "classifier_outputs": 200,
                "parameters": parameter_count,
                "feature_index": 5,
                "feature_shape": feature_shape,
                "guide_pool": {"models": pure_args.pretrained_model_number, "epochs": pure_args.pretrained_epochs},
                "evaluation_iterations": EXPECTED_SCHEDULE,
                "pcbn_hooks": hooked,
                "pcbn_weight": pcbn_args.pcbn_weight,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
