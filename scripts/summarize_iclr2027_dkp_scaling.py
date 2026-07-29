#!/usr/bin/env python3
"""Strictly audit and summarize one complete DKP scaling seed."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_data import get_receiver_dir  # noqa: E402
from config_adapter import build_dsdm_args_from_config, load_config  # noqa: E402
from output_manager import atomic_write_json  # noqa: E402
from packet_integrity import file_sha256, resolve_strict_dkp_contract  # noqa: E402
from run_social_pipeline import build_receiver_args  # noqa: E402


class ScalingSummaryError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise ScalingSummaryError(message)


def _resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _finite(row, field, context):
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ScalingSummaryError(f"{context}: {field} is missing or non-numeric") from error
    _require(math.isfinite(value), f"{context}: {field} is non-finite")
    return value


def _latest_receiver_rows(path, agent_count):
    _require(path.is_file(), f"social results are missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    latest = {}
    for row in rows:
        if row.get("method") == "DKP_SL" and row.get("dkp_variant") == "full":
            latest[int(row["receiver_agent"])] = row
    _require(sorted(latest) == list(range(agent_count)), "social results do not cover every receiver exactly")
    return latest


def _feature_tensors(result):
    result = result[0] if isinstance(result, tuple) else result
    if torch.is_tensor(result):
        return [result]
    _require(isinstance(result, (list, tuple)) and result, "get_feature returned no tensors")
    _require(all(torch.is_tensor(value) for value in result), "get_feature returned a non-tensor")
    return list(result)


def _audit_checkpoint(config, config_path, run_dir, receiver_id, row):
    args = build_receiver_args(config, str(config_path), receiver_id, packet_method="dsdm", init_mode="expert")
    checkpoint_dir = get_receiver_dir(args, receiver_id) / "checkpoints/dkp_sl_v1_full"
    checkpoint_path = checkpoint_dir / "after_social.pt"
    _require(checkpoint_path.is_file(), f"receiver {receiver_id}: final checkpoint is missing")
    checkpoint_sha = file_sha256(checkpoint_path)
    _require(
        row.get("after_social_checkpoint_sha256") == checkpoint_sha,
        f"receiver {receiver_id}: result/checkpoint SHA mismatch",
    )
    retained_pt = sorted(path.name for path in checkpoint_dir.glob("*.pt"))
    _require(retained_pt == ["after_social.pt"], f"receiver {receiver_id}: final_only retention drifted")
    provenance_path = _resolve(row.get("receiver_provenance_path", ""))
    _require(provenance_path.is_file(), f"receiver {receiver_id}: provenance is missing")
    _require(
        file_sha256(provenance_path) == row.get("receiver_provenance_sha256"),
        f"receiver {receiver_id}: provenance SHA mismatch",
    )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    balance = provenance.get("classification_balance", {})
    expected_local_weight = len(args.active_class_ids) / 100.0
    _require(
        math.isclose(float(balance.get("ce_local_weight", -1)), expected_local_weight, abs_tol=1e-12),
        f"receiver {receiver_id}: provenance local CE weight drifted",
    )
    _require(
        math.isclose(float(balance.get("ce_external_weight", -1)), 1.0 - expected_local_weight, abs_tol=1e-12),
        f"receiver {receiver_id}: provenance external CE weight drifted",
    )

    dsdm_root = ROOT / "DSDM"
    if str(dsdm_root) not in sys.path:
        sys.path.append(str(dsdm_root))
    from train import define_model
    from models.cosine_classifier import get_cosine_classifier

    model = define_model(args, 100)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    _require(state and all(torch.is_tensor(value) for value in state.values()), f"receiver {receiver_id}: invalid state_dict")
    for name, value in state.items():
        if value.is_floating_point() or value.is_complex():
            _require(bool(torch.isfinite(value).all()), f"receiver {receiver_id}: non-finite tensor {name}")
    model.load_state_dict(state, strict=True)
    classifier = get_cosine_classifier(model)
    _require(int(classifier.out_features) == 100, f"receiver {receiver_id}: classifier output drifted")
    _require(
        math.isfinite(float(classifier.scale.detach())),
        f"receiver {receiver_id}: cosine scale is non-finite",
    )
    model.eval()
    generator = torch.Generator().manual_seed(900_000 + receiver_id)
    images = torch.randn(2, int(args.nch), int(args.size), int(args.size), generator=generator)
    with torch.no_grad():
        logits = model(images)
        features = _feature_tensors(model.get_feature(images, int(args.idx_from), int(args.idx_to)))
    _require(tuple(logits.shape) == (2, 100), f"receiver {receiver_id}: output shape drifted")
    _require(bool(torch.isfinite(logits).all()), f"receiver {receiver_id}: output is non-finite")
    _require(all(bool(torch.isfinite(value).all()) for value in features), f"receiver {receiver_id}: feature is non-finite")
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "provenance": str(provenance_path),
        "provenance_sha256": row["receiver_provenance_sha256"],
        "output_shape": [2, 100],
        "feature_shapes": [list(value.shape) for value in features],
    }


def build_summary(config_path):
    config_path = _resolve(config_path)
    config = load_config(config_path)
    contract = resolve_strict_dkp_contract(config)
    run_dir = _resolve(config["project"].get("output_root", "outputs")) / config["project"]["run_name"]
    config_snapshot = run_dir / "config/main.yaml"
    _require(config_snapshot.is_file(), "immutable config snapshot is missing")
    _require(file_sha256(config_snapshot) == file_sha256(config_path), "config snapshot SHA differs")
    social_path = run_dir / "metrics/social_results.csv"
    rows = _latest_receiver_rows(social_path, contract.agent_count)

    local_weight = contract.classes_per_agent / 100.0
    external_weight = 1.0 - local_weight
    expected_external_raw = (contract.agent_count - 1) * contract.raw_per_sender
    sender_logit_bytes = contract.decoded_per_sender * contract.classes_per_agent * 2
    expected_external_logit_bytes = (contract.agent_count - 1) * sender_logit_bytes
    receiver_reports = []
    for receiver_id, row in sorted(rows.items()):
        context = f"receiver {receiver_id}"
        _require(row.get("protocol") == "dkp_sl_v1", f"{context}: receiver protocol drifted")
        _require(row.get("local_ce_source") == "real_packet_mix", f"{context}: local CE source drifted")
        _require(math.isclose(_finite(row, "local_ce_real_fraction", context), 0.02, abs_tol=1e-12), f"{context}: r02 fraction drifted")
        _require(int(row["optimizer_steps"]) == 3780, f"{context}: optimizer step count drifted")
        _require(int(row["target_optimizer_steps"]) == 3780, f"{context}: target step count drifted")
        _require(int(row["external_comm_images"]) == expected_external_raw, f"{context}: raw communication count drifted")
        _require(int(row["external_comm_logit_bytes"]) == expected_external_logit_bytes, f"{context}: logit bytes drifted")
        _require(int(row["self_packet_raw_images"]) == contract.raw_per_sender, f"{context}: self raw count drifted")
        _require(int(row["self_packet_decoded_images"]) == contract.decoded_per_sender, f"{context}: self decoded count drifted")

        losses = {
            name: _finite(row, name, context)
            for name in (
                "loss",
                "loss_cls",
                "loss_ce_local",
                "loss_ce_local_real",
                "loss_ce_local_packet",
                "loss_ce_external",
                "loss_fr",
                "loss_kd",
                "loss_sc",
            )
        }
        expected_local = 0.02 * losses["loss_ce_local_real"] + 0.98 * losses["loss_ce_local_packet"]
        expected_cls = local_weight * losses["loss_ce_local"] + external_weight * losses["loss_ce_external"]
        expected_total = losses["loss_cls"] + 0.2 * losses["loss_fr"] + 0.6 * losses["loss_kd"] + 0.1 * losses["loss_sc"]
        _require(abs(losses["loss_ce_local"] - expected_local) <= 2e-6, f"{context}: local CE identity mismatch")
        _require(abs(losses["loss_cls"] - expected_cls) <= 2e-6, f"{context}: class CE identity mismatch")
        _require(abs(losses["loss"] - expected_total) <= 2e-6, f"{context}: total loss identity mismatch")

        metrics = {
            "global": _finite(row, "acc_global_after", context),
            "new": _finite(row, "acc_new_after", context),
            "expert": _finite(row, "acc_expert_after", context),
        }
        expected_global = local_weight * metrics["expert"] + external_weight * metrics["new"]
        _require(abs(metrics["global"] - expected_global) <= 1e-6, f"{context}: metric identity mismatch")
        checkpoint = _audit_checkpoint(config, config_path, run_dir, receiver_id, row)
        receiver_reports.append(
            {
                "receiver_agent": receiver_id,
                "receiver_model": row["receiver_model"],
                "metrics": metrics,
                "loss_means": losses,
                "training_seconds": _finite(row, "training_seconds", context),
                **checkpoint,
            }
        )

    metrics_mean = {
        metric: statistics.fmean(report["metrics"][metric] for report in receiver_reports)
        for metric in ("global", "new", "expert")
    }
    return {
        "status": "complete_seed",
        "formal_result": False,
        "paper_eligible": False,
        "run_name": config["project"]["run_name"],
        "protocol": contract.protocol,
        "seed": int(config["runtime"]["seed"]),
        "agent_count": contract.agent_count,
        "classes_per_agent": contract.classes_per_agent,
        "receiver_ids": list(range(contract.agent_count)),
        "metrics_mean": metrics_mean,
        "ce_weights": {"local": local_weight, "external": external_weight},
        "communication": {
            "unique_sender_raw_images": contract.agent_count * contract.raw_per_sender,
            "unique_sender_logit_bytes": contract.agent_count * sender_logit_bytes,
            "external_raw_images_per_receiver": expected_external_raw,
            "external_logit_bytes_per_receiver": expected_external_logit_bytes,
            "receiver_incidence_raw_images": contract.agent_count * expected_external_raw,
            "receiver_incidence_logit_bytes": contract.agent_count * expected_external_logit_bytes,
        },
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "social_results_csv": str(social_path),
        "social_results_sha256": file_sha256(social_path),
        "receivers": receiver_reports,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    return parser.parse_args(argv)


def main(argv=None):
    cli = parse_args(argv)
    try:
        summary = build_summary(cli.config)
        config = load_config(_resolve(cli.config))
        run_dir = _resolve(config["project"].get("output_root", "outputs")) / config["project"]["run_name"]
        output = _resolve(cli.output) if cli.output else run_dir / "metrics/scaling_seed_summary.json"
        atomic_write_json(summary, output)
    except Exception as error:
        print(f"[dkp-scaling-summary] FAILED: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
