#!/usr/bin/env python3
from __future__ import annotations

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


PURE_CONFIG = ROOT / "configs/tinyimagenet_r18_all200_dsdm_ipc10_seed0.yaml"
PCBN_CONFIG = ROOT / "configs/tinyimagenet_r18_all200_dsdm_pcbn_ipc10_seed0.yaml"
EXPECTED_SCHEDULE = [100, 500, 1000, 2000, 3000, 5000, 7500, 10000]


def comparable(cfg: dict) -> dict:
    payload = copy.deepcopy(cfg)
    payload["project"].pop("run_name", None)
    payload["project"].pop("stage", None)
    payload["distillation"].pop("pcbn", None)
    return payload


def main() -> None:
    pure_cfg = load_config(PURE_CONFIG)
    pcbn_cfg = load_config(PCBN_CONFIG)
    if comparable(pure_cfg) != comparable(pcbn_cfg):
        raise ValueError("All-200 paired configs differ outside project identity and PCBN")

    expected_classes = list(range(200))
    pure_args = build_agent_args(pure_cfg, PURE_CONFIG, 0)
    pcbn_args = build_agent_args(pcbn_cfg, PCBN_CONFIG, 0)
    for name, args in (("pure", pure_args), ("pcbn", pcbn_args)):
        if args.dataset != "tinyimagenet" or args.nclass != 200 or args.size != 64:
            raise ValueError(f"{name}: invalid Tiny-ImageNet dimensions")
        if args.active_class_ids != expected_classes:
            raise ValueError(f"{name}: expected global classes 0-199")
        if args.ipc != 10 or args.factor != 2 or args.ipc * len(expected_classes) != 2000:
            raise ValueError(f"{name}: invalid IPC-10 image budget")
        if (args.idx_from, args.idx_to) != (5, -1):
            raise ValueError(f"{name}: invalid ResNet-18 feature index")
        if args.pretrained_model_number != 10 or args.pretrained_epochs != 100:
            raise ValueError(f"{name}: expected 10 guides x 100 epochs")
        schedule = resolve_evaluation_iterations(args.niter, args.evaluate_iterations, args.evaluate_iter)
        if schedule != EXPECTED_SCHEDULE:
            raise ValueError(f"{name}: unexpected evaluation schedule: {schedule}")

    train_dataset = get_agent_train_dataset(pure_args, 0, normalize=False)
    val_dataset = get_test_dataset(pure_args)
    if len(train_dataset) != 100_000 or len(val_dataset) != 9_993:
        raise ValueError(f"Unexpected split sizes: train={len(train_dataset)} val={len(val_dataset)}")
    if set(train_dataset.targets) != set(expected_classes):
        raise ValueError("All-200 labels were remapped or filtered incorrectly")

    model = define_model(pure_args, 200).eval()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    with torch.no_grad():
        output_shape = tuple(model(torch.randn(2, 3, 64, 64)).shape)
        feature_shape = tuple(model.get_feature(torch.randn(2, 3, 64, 64), 5, -1)[0].shape)
    if parameters != 11_271_432 or output_shape != (2, 200) or feature_shape != (2, 512):
        raise ValueError("Unexpected standard Tiny-ImageNet ResNet-18 contract")

    pure_regularizer = PCBNRegularizer(pure_args)
    pcbn_regularizer = PCBNRegularizer(pcbn_args)
    if pure_regularizer.enabled or not pcbn_regularizer.enabled:
        raise ValueError("PCBN flags are not an off/on pair")
    hooks = pcbn_regularizer.attach(model)
    pcbn_regularizer.close()
    if hooks != 20:
        raise ValueError(f"Expected 20 BN hooks, found {hooks}")

    print(json.dumps({
        "status": "passed",
        "classes": [0, 199],
        "train_images": len(train_dataset),
        "clean_validation_images": len(val_dataset),
        "ipc": pure_args.ipc,
        "raw_synthetic_images": pure_args.ipc * len(expected_classes),
        "decoded_training_images": pure_args.ipc * len(expected_classes) * pure_args.factor ** 2,
        "guide_pool": {"models": 10, "epochs": 100},
        "parameters": parameters,
        "feature_index": 5,
        "evaluation_iterations": EXPECTED_SCHEDULE,
        "pcbn_hooks": hooks,
        "pcbn_weight": pcbn_args.pcbn_weight,
    }, indent=2))


if __name__ == "__main__":
    main()
