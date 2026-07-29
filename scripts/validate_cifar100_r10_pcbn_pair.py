#!/usr/bin/env python3
"""Validate that the full-class R10 pair differs only by PCBN and run identity."""

from __future__ import annotations

import copy
import hashlib
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


PURE_CONFIG = ROOT / (
    "configs/fullclass_dsdm/"
    "fullclass_resnet10_standard_modelbest_e0200_ipc10_seed0_recovery.yaml"
)
PCBN_CONFIG = ROOT / (
    "configs/fullclass_dsdm/"
    "fullclass_resnet10_standard_modelbest_e0200_ipc10_seed0_recovery_pcbn.yaml"
)
CALIBRATION = ROOT / "experiments/diagnostics/cifar100_r10_pcbn_weight_calibration_seed0.json"
PURE_RUN = ROOT / "outputs/cifar100_fullclass_dsdm_resnet10_standard_modelbest_e0200_ipc10_seed0_recovery"
EXPECTED_SCHEDULE = [100, 500, 1000, 2000, 3000, 5000, 7500, 10000]


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


def comparable(cfg: dict) -> dict:
    payload = copy.deepcopy(cfg)
    payload["project"].pop("run_name", None)
    payload["project"].pop("stage", None)
    payload["distillation"].pop("pcbn", None)
    payload.get("fullclass_pool", {}).pop("pcbn_control", None)
    return payload


def main() -> None:
    pure_cfg = load_config(PURE_CONFIG)
    pcbn_cfg = load_config(PCBN_CONFIG)
    if comparable(pure_cfg) != comparable(pcbn_cfg):
        raise ValueError("R10 paired configs differ outside run identity and PCBN provenance")

    pure_args = build_agent_args(pure_cfg, PURE_CONFIG, 0)
    pcbn_args = build_agent_args(pcbn_cfg, PCBN_CONFIG, 0)
    for name, args in (("pure", pure_args), ("pcbn", pcbn_args)):
        if args.dataset != "cifar100" or args.nclass != 100 or args.size != 32:
            raise ValueError(f"{name}: invalid CIFAR-100 dimensions")
        if args.active_class_ids != list(range(100)):
            raise ValueError(f"{name}: global labels 0-99 are not preserved")
        if args.net_type != "resnet_cifar_standard" or args.depth != 10:
            raise ValueError(f"{name}: expected standard-width CIFAR ResNet-10")
        if (args.idx_from, args.idx_to) != (5, -1):
            raise ValueError(f"{name}: expected penultimate feature index 5")
        if args.ipc != 10 or args.factor != 2 or args.lr_img != 0.01 or args.niter != 10000:
            raise ValueError(f"{name}: DSDM recipe changed")
        if args.pretrained_model_number != 10 or args.pretrained_epochs != 200:
            raise ValueError(f"{name}: expected 10 independent epoch-200 guides")
        schedule = resolve_evaluation_iterations(args.niter, args.evaluate_iterations, args.evaluate_iter)
        if schedule != EXPECTED_SCHEDULE:
            raise ValueError(f"{name}: unexpected evaluation schedule {schedule}")
    if pure_args.guide_source_root != pcbn_args.guide_source_root:
        raise ValueError("R10 pair does not reuse the same guide source root")

    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    selected_weight = float(calibration["selection"]["selected_weight"])
    if calibration.get("status") != "passed" or float(pcbn_args.pcbn_weight) != selected_weight:
        raise ValueError("PCBN config weight does not match the passed calibration")

    pure_regularizer = PCBNRegularizer(pure_args)
    pcbn_regularizer = PCBNRegularizer(pcbn_args)
    if pure_regularizer.enabled or not pcbn_regularizer.enabled:
        raise ValueError("PCBN flags are not an off/on pair")

    model = define_model(pure_args, 100).eval()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    with torch.no_grad():
        output_shape = tuple(model(torch.randn(2, 3, 32, 32)).shape)
        feature_shape = tuple(model.get_feature(torch.randn(2, 3, 32, 32), 5, -1)[0].shape)
    hooks = pcbn_regularizer.attach(model)
    pcbn_regularizer.close()
    if parameters != 4_949_412 or output_shape != (2, 100) or feature_shape != (2, 512) or hooks != 12:
        raise ValueError(
            f"Unexpected R10/PCBN contract params={parameters} output={output_shape} "
            f"feature={feature_shape} hooks={hooks}"
        )

    manifest_path = PURE_RUN / "agents/agent_0/checkpoints/dsdm_pretrained/pretrained_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(manifest.get("models", [])) != 10:
        raise ValueError("Completed pure R10 guide manifest is incomplete")
    for item in manifest["models"]:
        src = Path(item["src"])
        dst = Path(item["dst"])
        if not src.is_absolute():
            src = ROOT / src
        if not dst.is_absolute():
            dst = ROOT / dst
        if sha256(src) != item["src_sha256"] or sha256(dst) != item["dst_sha256"]:
            raise ValueError(f"Guide artifact hash changed: model {item['model_idx']}")
        if not state_dicts_equal(src, dst):
            raise ValueError(f"Pure R10 guide state changed during mapping: model {item['model_idx']}")

    best_manifest = json.loads(
        (PURE_RUN / "agents/agent_0/synthetic/best_manifest.json").read_text(encoding="utf-8")
    )
    integrity = json.loads((PURE_RUN / "metrics/packet_integrity_dsdm.json").read_text(encoding="utf-8"))
    if best_manifest.get("pcbn_enabled") or float(best_manifest.get("best_acc", -1)) != 38.86:
        raise ValueError("Completed pure R10 reference is not the expected 38.86% run")
    summary = integrity.get("summary", {})
    if (
        summary.get("total_raw_images") != 1000
        or summary.get("total_train_images") != 4000
        or set(summary.get("per_class_raw_images", {}).values()) != {10}
        or integrity.get("warnings")
    ):
        raise ValueError("Completed pure R10 packet integrity did not pass")

    print(
        json.dumps(
            {
                "status": "passed",
                "pure_run": pure_cfg["project"]["run_name"],
                "pcbn_run": pcbn_cfg["project"]["run_name"],
                "pure_best_accuracy": float(best_manifest["best_acc"]),
                "pure_best_iteration": int(best_manifest["iteration"]),
                "parameters": parameters,
                "classifier_outputs": output_shape[-1],
                "feature_index": 5,
                "feature_shape": list(feature_shape),
                "guide_count": len(manifest["models"]),
                "guide_source_root": str(pure_args.guide_source_root),
                "guide_pairing": "exact_state_dict_match; serialization hashes retained separately",
                "evaluation_iterations": EXPECTED_SCHEDULE,
                "pcbn_hooks": hooks,
                "pcbn_weight": selected_weight,
                "calibrated_initial_total_loss_share": float(
                    calibration["selection"]["realized_mean_total_loss_share"]
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
