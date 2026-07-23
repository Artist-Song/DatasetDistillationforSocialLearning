#!/usr/bin/env python3
"""Validate and summarize teacher-maturity DSDM image/logit quality runs."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from output_manager import atomic_write_json
from scripts.prepare_teacher_quality_protocol import MODELS


LEGACY_IMAGE_REFERENCES = {
    "conv3": 58.75,
    "conv4": 66.90,
    "alexnet": 58.15,
    "resnet10_standard": 55.45,
    "resnet18_standard": 55.95,
}


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--model", action="append", choices=sorted(MODELS), dest="models")
    parser.add_argument("--max-image-regression", type=float, default=2.0)
    parser.add_argument("--min-teacher-test-accuracy", type=float, default=70.0)
    parser.add_argument("--min-logit-label-agreement", type=float, default=0.70)
    return parser.parse_args()


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _finite_number(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _config_path(name, epoch, smoke):
    root = ROOT / "configs" / ("teacher_quality_smoke" if smoke else "teacher_quality")
    return root / f"packet_{name}_guidee{int(epoch):04d}_seed0_ipc10.yaml"


def _inspect_candidate(name, spec, epoch, smoke):
    config_path = _config_path(name, epoch, smoke)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    run_name = config["project"]["run_name"]
    agent_id = int(spec["agent"])
    agent_dir = ROOT / "outputs" / run_name / "agents" / f"agent_{agent_id}"
    packet_path = agent_dir / "packets" / "dsdm_packet.pt"
    best_path = agent_dir / "synthetic" / "best_manifest.json"
    quality_path = agent_dir / "metrics" / "logit_quality_dsdm.json"
    pretrained_manifest_path = (
        agent_dir / "checkpoints" / "dsdm_pretrained" / f"e{int(epoch):04d}" / "pretrained_manifest.json"
    )
    required = [packet_path, best_path, quality_path, pretrained_manifest_path]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        return None, missing

    packet = torch.load(packet_path, map_location="cpu")
    best = _read_json(best_path)
    quality = _read_json(quality_path)
    pretrained_manifest = _read_json(pretrained_manifest_path)
    images = packet.get("images")
    labels = packet.get("labels")
    logits = packet.get("sender_logits")
    class_ids = [int(value) for value in packet.get("class_ids", [])]
    expected_classes = [int(value) for value in config["agents"]["class_split"][f"agent_{agent_id}"]]
    expected_raw_images = int(config["distillation"]["ipc"]) * len(expected_classes)
    expected_iterations = int(config["distillation"]["niter"])
    teacher = quality.get("teacher", {})

    checks = {
        "images_tensor": torch.is_tensor(images),
        "images_finite": torch.is_tensor(images) and bool(torch.isfinite(images).all().item()),
        "raw_image_count": torch.is_tensor(images) and int(images.shape[0]) == expected_raw_images,
        "labels_aligned": torch.is_tensor(labels) and torch.is_tensor(images) and len(labels) == len(images),
        "class_ids_exact": class_ids == expected_classes,
        "logits_tensor": torch.is_tensor(logits),
        "logits_finite": torch.is_tensor(logits) and bool(torch.isfinite(logits).all().item()),
        "logit_class_dim": torch.is_tensor(logits) and int(logits.shape[1]) == len(expected_classes),
        "logit_rows_recorded": torch.is_tensor(logits)
        and int(logits.shape[0]) == int(packet.get("sender_logit_num_images", -1)),
        "condense_complete": bool(packet.get("meta", {}).get("condense_complete", False)),
        "completed_iterations": int(packet.get("meta", {}).get("completed_iterations", -1))
        == expected_iterations,
        "guide_epoch_provenance": int(packet.get("meta", {}).get("guide_epoch", -1)) == int(epoch),
        "guide_manifest_provenance": packet.get("meta", {}).get("guide_pretrained_manifest")
        == str(pretrained_manifest_path.relative_to(ROOT)),
        "guide_checkpoint_count": len(pretrained_manifest.get("models", []))
        == int(config["model_pool"]["models"][spec["model_id"]]["guide_training"]["num_models"]),
        "guide_checkpoints_hashed": all(
            len(str(item.get("src_sha256", ""))) == 64 and len(str(item.get("dst_sha256", ""))) == 64
            for item in pretrained_manifest.get("models", [])
        ),
        "best_accuracy_finite": _finite_number(best.get("best_acc")),
        "quality_metrics_finite": all(
            _finite_number(quality.get(field))
            for field in (
                "packet_label_agreement",
                "mean_entropy",
                "mean_top1_margin",
                "mean_abs_logit",
                "max_abs_logit",
            )
        ),
        "teacher_role": teacher.get("role") == "fully_converged_agent_expert_and_logit_teacher",
        "teacher_not_test_selected": teacher.get("test_used_for_selection") is False,
        "teacher_checkpoint_hashed": len(str(teacher.get("checkpoint_sha256", ""))) == 64,
    }
    structural_pass = all(checks.values())
    row = {
        "model": name,
        "agent_id": agent_id,
        "guide_epoch": int(epoch),
        "run_name": run_name,
        "structural_pass": structural_pass,
        "failed_checks": ",".join(key for key, value in checks.items() if not value),
        "best_image_accuracy": float(best["best_acc"]) if _finite_number(best.get("best_acc")) else float("nan"),
        "best_iteration": int(best.get("iteration", -1)),
        "raw_image_count": int(images.shape[0]) if torch.is_tensor(images) else -1,
        "decoded_logit_rows": int(logits.shape[0]) if torch.is_tensor(logits) else -1,
        "logit_dim": int(logits.shape[1]) if torch.is_tensor(logits) and logits.ndim == 2 else -1,
        "teacher_selected_epoch": teacher.get("selected_epoch"),
        "teacher_validation_accuracy": teacher.get("best_validation_accuracy"),
        "teacher_test_accuracy_report_only": teacher.get("official_test_accuracy_report_only"),
        "packet_label_agreement": quality.get("packet_label_agreement"),
        "mean_entropy": quality.get("mean_entropy"),
        "mean_top1_margin": quality.get("mean_top1_margin"),
        "mean_abs_logit": quality.get("mean_abs_logit"),
        "max_abs_logit": quality.get("max_abs_logit"),
        "packet_path": str(packet_path.relative_to(ROOT)),
    }
    return row, []


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["model"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = _parse_args()
    selected_models = args.models or list(MODELS)
    rows = []
    missing = []
    for name in selected_models:
        spec = MODELS[name]
        epochs = [1, 2] if args.smoke else spec["guide_epochs"]
        for epoch in epochs:
            row, candidate_missing = _inspect_candidate(name, spec, epoch, args.smoke)
            if row is not None:
                rows.append(row)
            missing.extend(candidate_missing)

    selections = []
    failures = []
    for name in selected_models:
        candidates = [row for row in rows if row["model"] == name and row["structural_pass"]]
        if not candidates:
            failures.append(f"{name}: no structurally valid candidate")
            continue
        selected = max(candidates, key=lambda row: row["best_image_accuracy"])
        reference = LEGACY_IMAGE_REFERENCES[name]
        image_gate = selected["best_image_accuracy"] >= reference - args.max_image_regression
        teacher_gate = _finite_number(selected["teacher_test_accuracy_report_only"]) and (
            float(selected["teacher_test_accuracy_report_only"]) >= args.min_teacher_test_accuracy
        )
        agreement_gate = _finite_number(selected["packet_label_agreement"]) and (
            float(selected["packet_label_agreement"]) >= args.min_logit_label_agreement
        )
        selection = {
            **selected,
            "legacy_image_reference": reference,
            "legacy_reference_kind": (
                "same_backbone_pat5_seed0" if not name.startswith("resnet") else "compact_resnet_pat5_floor_only"
            ),
            "image_quality_gate": image_gate,
            "teacher_quality_gate": teacher_gate,
            "logit_agreement_gate": agreement_gate,
            "performance_gate_applied": not args.smoke,
        }
        selections.append(selection)
        if not args.smoke:
            if not image_gate:
                failures.append(
                    f"{name}: image accuracy {selected['best_image_accuracy']:.2f} below "
                    f"reference gate {reference - args.max_image_regression:.2f}"
                )
            if not teacher_gate:
                failures.append(f"{name}: mature teacher accuracy below {args.min_teacher_test_accuracy:.2f}")
            if not agreement_gate:
                failures.append(f"{name}: packet/logit agreement below {args.min_logit_label_agreement:.3f}")

    if missing and not args.allow_incomplete:
        failures.append(f"missing {len(missing)} required artifacts")
    structural_failures = [row for row in rows if not row["structural_pass"]]
    if structural_failures:
        failures.extend(
            f"{row['model']}@{row['guide_epoch']}: {row['failed_checks']}" for row in structural_failures
        )

    suffix = "_smoke" if args.smoke else ""
    output_dir = ROOT / "outputs" / f"teacher_quality_seed0_summary{suffix}"
    _write_csv(output_dir / "candidates.csv", rows)
    _write_csv(output_dir / "selected.csv", selections)
    payload = {
        "protocol": "cifar100_pat5_teacher_maturity_seed0",
        "smoke": bool(args.smoke),
        "complete": not missing,
        "passed": not failures,
        "thresholds": {
            "max_image_regression": args.max_image_regression,
            "min_teacher_test_accuracy": args.min_teacher_test_accuracy,
            "min_logit_label_agreement": args.min_logit_label_agreement,
        },
        "missing": missing,
        "failures": failures,
        "candidates": rows,
        "selected": selections,
    }
    atomic_write_json(payload, output_dir / "summary.json")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
