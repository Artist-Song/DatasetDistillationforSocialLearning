#!/usr/bin/env python3
"""Validate R10 PCBN weight candidates against the w960 reference."""

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

from agent_data import build_agent_args  # noqa: E402
from config_adapter import load_config  # noqa: E402
from evaluation_schedule import resolve_evaluation_iterations  # noqa: E402
from pcbn import PCBNRegularizer  # noqa: E402
from train import define_model  # noqa: E402


REFERENCE = ROOT / (
    "configs/fullclass_dsdm/"
    "fullclass_resnet10_standard_modelbest_e0200_ipc10_seed0_recovery_pcbn.yaml"
)
GRADIENT_DIAGNOSTIC = ROOT / "experiments/diagnostics/cifar100_r10_pcbn_gradient_diagnostic_seed0.json"
EXPECTED_SCHEDULE = [100, 500, 1000, 2000, 3000, 5000, 7500, 10000]
ALLOWED_WEIGHTS = {1300.0, 2100.0, 3400.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weight", type=float, required=True)
    return parser.parse_args()


def comparable(cfg: dict) -> dict:
    payload = copy.deepcopy(cfg)
    payload["project"].pop("run_name", None)
    payload["project"].pop("stage", None)
    payload["distillation"]["pcbn"].pop("weight", None)
    payload.get("fullclass_pool", {}).pop("pcbn_control", None)
    return payload


def main() -> None:
    cli = parse_args()
    if cli.weight not in ALLOWED_WEIGHTS:
        raise ValueError(f"Unsupported PCBN sweep weight: {cli.weight}")
    config_path = cli.config if cli.config.is_absolute() else ROOT / cli.config
    reference_cfg = load_config(REFERENCE)
    candidate_cfg = load_config(config_path)
    if comparable(reference_cfg) != comparable(candidate_cfg):
        raise ValueError("PCBN sweep config differs outside run identity, weight, and sweep provenance")

    reference_args = build_agent_args(reference_cfg, REFERENCE, 0)
    candidate_args = build_agent_args(candidate_cfg, config_path, 0)
    if float(reference_args.pcbn_weight) != 960.0 or float(candidate_args.pcbn_weight) != cli.weight:
        raise ValueError("Resolved PCBN weights do not match the sweep contract")
    for name, args in (("reference", reference_args), ("candidate", candidate_args)):
        if args.dataset != "cifar100" or args.nclass != 100 or args.active_class_ids != list(range(100)):
            raise ValueError(f"{name}: invalid full-class CIFAR-100 contract")
        if args.net_type != "resnet_cifar_standard" or args.depth != 10:
            raise ValueError(f"{name}: expected standard CIFAR ResNet-10")
        if (args.idx_from, args.idx_to) != (5, -1):
            raise ValueError(f"{name}: expected penultimate feature index 5")
        if args.ipc != 10 or args.factor != 2 or args.lr_img != 0.01 or args.niter != 10000:
            raise ValueError(f"{name}: DSDM recipe changed")
        if args.guide_model_mode != "eval" or not args.freeze_guide_parameters:
            raise ValueError(f"{name}: PCBN sweep requires eval/frozen guides")
        if args.pretrained_model_number != 10 or args.pretrained_epochs != 200:
            raise ValueError(f"{name}: guide pool changed")
        if resolve_evaluation_iterations(args.niter, args.evaluate_iterations, args.evaluate_iter) != EXPECTED_SCHEDULE:
            raise ValueError(f"{name}: evaluation schedule changed")
    if reference_args.guide_source_root != candidate_args.guide_source_root:
        raise ValueError("PCBN sweep does not reuse the reference guide source")

    diagnostic = json.loads(GRADIENT_DIAGNOSTIC.read_text(encoding="utf-8"))
    if diagnostic.get("status") != "passed" or cli.weight not in diagnostic["protocol"]["weights"]:
        raise ValueError("Candidate weight lacks a passed gradient diagnostic")

    model = define_model(candidate_args, 100).eval()
    regularizer = PCBNRegularizer(candidate_args)
    hooks = regularizer.attach(model)
    selected_layers = list(regularizer.layer_names)
    regularizer.close()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    with torch.no_grad():
        output_shape = tuple(model(torch.randn(2, 3, 32, 32)).shape)
        feature_shape = tuple(model.get_feature(torch.randn(2, 3, 32, 32), 5, -1)[0].shape)
    if parameters != 4_949_412 or output_shape != (2, 100) or feature_shape != (2, 512) or hooks != 12:
        raise ValueError("Unexpected standard R10/PCBN contract")

    print(
        json.dumps(
            {
                "status": "passed",
                "config": str(config_path.relative_to(ROOT)),
                "run_name": candidate_cfg["project"]["run_name"],
                "weight": cli.weight,
                "guide_source_root": candidate_args.guide_source_root,
                "evaluation_iterations": EXPECTED_SCHEDULE,
                "parameters": parameters,
                "feature_shape": list(feature_shape),
                "pcbn_hook_count": hooks,
                "pcbn_layers": selected_layers,
                "gradient_diagnostic": str(GRADIENT_DIAGNOSTIC.relative_to(ROOT)),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
