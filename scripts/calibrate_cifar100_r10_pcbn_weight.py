#!/usr/bin/env python3
"""Calibrate a conservative PCBN weight for the full-class CIFAR-100 R10 pair."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DSDM_ROOT = ROOT / "DSDM"
for path in (ROOT, DSDM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent_data import build_agent_args  # noqa: E402
from config_adapter import load_config  # noqa: E402
from DSDM import ClassMemDataLoader, Synthesizer, diffaug, load_resized_data, matchloss  # noqa: E402
from output_manager import atomic_write_json  # noqa: E402
from pcbn import PCBNRegularizer  # noqa: E402
from train import define_model  # noqa: E402


PURE_CONFIG = ROOT / (
    "configs/fullclass_dsdm/"
    "fullclass_resnet10_standard_modelbest_e0200_ipc10_seed0_recovery.yaml"
)
PURE_RUN = ROOT / "outputs/cifar100_fullclass_dsdm_resnet10_standard_modelbest_e0200_ipc10_seed0_recovery"
DEFAULT_OUTPUT = ROOT / "experiments/diagnostics/cifar100_r10_pcbn_weight_calibration_seed0.json"
SAMPLED_CLASSES = (0, 20, 40, 60, 80, 99)
SAMPLED_GUIDES = (0, 5)
TARGET_SHARES = (0.05, 0.075, 0.10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dicts_equal(left_path: Path, right_path: Path) -> bool:
    left = torch.load(left_path, map_location="cpu", weights_only=True)
    right = torch.load(right_path, map_location="cpu", weights_only=True)
    if left.keys() != right.keys():
        return False
    return all(torch.equal(left[key], right[key]) for key in left)


def rounded_weight(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid calibrated PCBN weight: {value}")
    exponent = math.floor(math.log10(value))
    quantum = 10 ** (exponent - 1)
    return float(round(value / quantum) * quantum)


def weight_for_share(base_mean: float, raw_pcbn_mean: float, target_share: float) -> float:
    return (target_share / (1.0 - target_share)) * base_mean / raw_pcbn_mean


def main() -> None:
    cli = parse_args()
    if cli.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA calibration requested but CUDA is unavailable")
    device = torch.device(cli.device)

    cfg = load_config(PURE_CONFIG)
    args = build_agent_args(cfg, PURE_CONFIG, 0)
    if args.net_type != "resnet_cifar_standard" or args.depth != 10:
        raise ValueError("Calibration requires the standard-width CIFAR ResNet-10")
    if args.active_class_ids != list(range(100)) or args.ipc != 10 or args.factor != 2:
        raise ValueError("Calibration requires the full CIFAR-100 IPC=10/factor=2 protocol")
    if (args.idx_from, args.idx_to) != (5, -1) or args.lr_img != 0.01:
        raise ValueError("Calibration config does not match the completed R10 DSDM run")

    manifest_path = PURE_RUN / "agents/agent_0/checkpoints/dsdm_pretrained/pretrained_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_models = {int(item["model_idx"]): item for item in manifest["models"]}
    missing = [index for index in SAMPLED_GUIDES if index not in manifest_models]
    if missing:
        raise ValueError(f"Guide manifest is incomplete for calibration: {missing}")

    seed_everything(int(args.seed))
    trainset, _ = load_resized_data(args)
    loader = ClassMemDataLoader(trainset, batch_size=args.batch_real, device=device)
    synset = Synthesizer(args, args.nclass, args.nch, args.size, args.size, device=device)
    synset.init(loader, init_type=args.init)
    augmentation, _ = diffaug(args, device=device)

    regularizer_args = copy.copy(args)
    regularizer_args.pcbn_enabled = True
    regularizer_args.pcbn_weight = 1.0
    regularizer_args.pcbn_layers = "all"
    regularizer_args.pcbn_normalize_layers = True

    samples = []
    hook_count = None
    guide_evidence = []
    for guide_index in SAMPLED_GUIDES:
        item = manifest_models[guide_index]
        guide_path = Path(item["dst"])
        if not guide_path.is_absolute():
            guide_path = ROOT / guide_path
        if not guide_path.exists():
            raise FileNotFoundError(guide_path)
        actual_hash = sha256(guide_path)
        source_path = Path(item["src"])
        if not source_path.is_absolute():
            source_path = ROOT / source_path
        if actual_hash != item["dst_sha256"] or sha256(source_path) != item["src_sha256"]:
            raise ValueError(f"Guide hash mismatch: {guide_path}")
        if not state_dicts_equal(source_path, guide_path):
            raise ValueError(f"Guide state changed during DSDM mapping: {guide_path}")
        guide_evidence.append(
            {
                "model_idx": guide_index,
                "path": str(guide_path.relative_to(ROOT)),
                "source_file_sha256": item["src_sha256"],
                "mapped_file_sha256": actual_hash,
                "state_dict_exact_match": True,
            }
        )

        model = define_model(args, args.nclass).to(device).eval()
        model.load_state_dict(torch.load(guide_path, map_location=device, weights_only=True))
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        regularizer = PCBNRegularizer(regularizer_args)
        current_hooks = regularizer.attach(model)
        if hook_count is None:
            hook_count = current_hooks
        elif hook_count != current_hooks:
            raise ValueError("PCBN hook count changed between guides")

        for class_id in SAMPLED_CLASSES:
            sample_seed = int(args.seed) + 10_000 * guide_index + class_id
            seed_everything(sample_seed)
            real_images, _ = loader.class_sample(class_id)
            syn_images, _ = synset.sample(class_id, max_size=args.batch_syn_max)
            real_count = int(real_images.shape[0])
            augmented = augmentation(torch.cat([real_images, syn_images]))
            real_aug = augmented[:real_count]
            syn_aug = augmented[real_count:]

            base_loss = matchloss(args, real_aug, syn_aug, model)
            raw_pcbn_loss = regularizer.loss(model, real_aug, syn_aug)
            base_value = float(base_loss.detach().item())
            raw_value = float(raw_pcbn_loss.detach().item())
            if not math.isfinite(base_value) or not math.isfinite(raw_value) or raw_value <= 0:
                raise FloatingPointError(
                    f"Non-finite calibration loss guide={guide_index} class={class_id}: "
                    f"base={base_value} pcbn={raw_value}"
                )
            samples.append(
                {
                    "guide_index": guide_index,
                    "class_id": class_id,
                    "sample_seed": sample_seed,
                    "base_dsdm_loss": base_value,
                    "raw_normalized_pcbn_loss": raw_value,
                    "raw_pcbn_to_base_ratio": raw_value / base_value,
                }
            )
            del base_loss, raw_pcbn_loss, augmented, real_aug, syn_aug

        regularizer.close()
        del regularizer, model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    base_values = [sample["base_dsdm_loss"] for sample in samples]
    raw_values = [sample["raw_normalized_pcbn_loss"] for sample in samples]
    base_mean = statistics.fmean(base_values)
    raw_mean = statistics.fmean(raw_values)
    candidates = {}
    for target_share in TARGET_SHARES:
        raw_weight = weight_for_share(base_mean, raw_mean, target_share)
        candidates[f"{target_share:.3f}"] = {
            "target_total_loss_share": target_share,
            "unrounded_weight": raw_weight,
            "two_significant_digit_weight": rounded_weight(raw_weight),
        }

    selected_target = 0.075
    selected_weight = candidates[f"{selected_target:.3f}"]["two_significant_digit_weight"]
    weighted_pcbn_mean = selected_weight * raw_mean
    realized_share = weighted_pcbn_mean / (base_mean + weighted_pcbn_mean)
    if not 0.06 <= realized_share <= 0.09:
        raise ValueError(f"Rounded PCBN weight misses conservative target: share={realized_share}")

    model = define_model(args, args.nclass).eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    with torch.no_grad():
        output_shape = tuple(model(torch.randn(2, 3, 32, 32)).shape)
        feature_shape = tuple(model.get_feature(torch.randn(2, 3, 32, 32), 5, -1)[0].shape)
    if parameter_count != 4_949_412 or output_shape != (2, 100) or feature_shape != (2, 512):
        raise ValueError(
            f"Unexpected R10 contract params={parameter_count} output={output_shape} feature={feature_shape}"
        )
    if hook_count != 12:
        raise ValueError(f"Expected 12 standard R10 BN hooks, found {hook_count}")

    best_manifest_path = PURE_RUN / "agents/agent_0/synthetic/best_manifest.json"
    best_manifest = json.loads(best_manifest_path.read_text(encoding="utf-8"))
    payload = {
        "status": "passed",
        "purpose": "initial-loss-scale calibration; not an accuracy hyperparameter sweep",
        "pure_config": str(PURE_CONFIG.relative_to(ROOT)),
        "pure_run": str(PURE_RUN.relative_to(ROOT)),
        "pure_best_accuracy": float(best_manifest["best_acc"]),
        "pure_best_iteration": int(best_manifest["iteration"]),
        "device": str(device),
        "protocol": {
            "dataset": "cifar100",
            "class_ids": [0, 99],
            "raw_ipc": int(args.ipc),
            "factor": int(args.factor),
            "initialization": str(args.init),
            "backbone": "resnet10_standard",
            "parameters": parameter_count,
            "feature_index": int(args.idx_from),
            "feature_shape": list(feature_shape),
            "pcbn_layers": "all",
            "pcbn_hook_count": hook_count,
            "pcbn_normalize_layers": True,
            "sampled_classes": list(SAMPLED_CLASSES),
            "sampled_guides": list(SAMPLED_GUIDES),
        },
        "guide_evidence": guide_evidence,
        "aggregate": {
            "sample_count": len(samples),
            "base_dsdm_loss_mean": base_mean,
            "base_dsdm_loss_median": statistics.median(base_values),
            "raw_normalized_pcbn_loss_mean": raw_mean,
            "raw_normalized_pcbn_loss_median": statistics.median(raw_values),
        },
        "candidate_weights": candidates,
        "selection": {
            "rule": "two-significant-digit weight targeting 7.5% of initial total loss",
            "target_total_loss_share": selected_target,
            "selected_weight": selected_weight,
            "realized_mean_total_loss_share": realized_share,
            "weighted_pcbn_loss_mean": weighted_pcbn_mean,
        },
        "samples": samples,
    }
    output_path = cli.output if cli.output.is_absolute() else ROOT / cli.output
    atomic_write_json(payload, output_path)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
