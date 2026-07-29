#!/usr/bin/env python3
"""Fail-closed final-checkpoint audit for the six DKP loss-ablation runs."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DSDM_ROOT = ROOT / "DSDM"
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))
if str(DSDM_ROOT) not in sys.path:
    sys.path.append(str(DSDM_ROOT))

from config_adapter import load_config  # noqa: E402
from scripts.audit_iclr2027_dkp_final_receivers import (  # noqa: E402
    _default_model_builder,
)
from scripts.prepare_iclr2027_dkp_loss_ablation import (  # noqa: E402
    MISSING_COMBINATIONS,
    variant_id,
)
from scripts.summarize_iclr2027_dkp_first_round import (  # noqa: E402
    EXPECTED_RECEIVERS,
    LOSS_FIELDS,
    load_complete_experts,
)
from scripts.summarize_iclr2027_dkp_loss_ablation import (  # noqa: E402
    ALL_SWITCHES,
    DEFAULT_CONFIG_DIR,
    DEFAULT_EXPERT_CONFIG,
    DEFAULT_OUTPUT as DEFAULT_SUMMARY,
    default_config_paths,
    validate_configs,
    validate_social_run,
)
from scripts.validate_iclr2027_cosine_experts import (  # noqa: E402
    resolve_run_dir,
    sha256_file,
)
from social_output_manager import SOCIAL_RESULT_FIELDS  # noqa: E402


DEFAULT_REPORT = (
    ROOT
    / "outputs/diagnostics/"
    "iclr2027_cifar100_5agent20cls_ipc10_seed0_loss_ablation_final_checkpoint_audit_v1.json"
)
EXPECTED_MODELS = {
    0: "convnet3w1",
    1: "convnet4w15",
    2: "alexnet",
    3: "resnet10_standard",
    4: "resnet18_standard",
}
CONDITION_SWITCHES = {
    variant_id(*switches): switches for switches in MISSING_COMBINATIONS
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
LOSS_TOLERANCE = 1e-6


class LossAblationFinalAuditError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise LossAblationFinalAuditError(message)


def _resolve(path):
    resolved = Path(path)
    return resolved.resolve() if resolved.is_absolute() else (ROOT / resolved).resolve()


def _load_json(path, context):
    path = Path(path)
    _require(path.is_file(), f"{context} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LossAblationFinalAuditError(f"cannot read {context}: {path}") from exc
    _require(isinstance(payload, dict), f"{context} is not a JSON object: {path}")
    return payload


def _finite_float(value, context):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise LossAblationFinalAuditError(
            f"{context} is not numeric: {value!r}"
        ) from exc
    _require(math.isfinite(result), f"{context} is not finite: {result}")
    return result


def _integer(value, context):
    result = _finite_float(value, context)
    _require(result.is_integer(), f"{context} is not an integer: {result}")
    return int(result)


def _same(actual, expected, context, tolerance=1e-9):
    actual_value = _finite_float(actual, context)
    expected_value = _finite_float(expected, f"{context} expected value")
    _require(
        math.isclose(actual_value, expected_value, rel_tol=0.0, abs_tol=tolerance),
        f"{context} mismatch: {actual_value} != {expected_value}",
    )
    return actual_value


def _valid_sha256(value, context):
    digest = str(value).strip().lower()
    _require(bool(SHA256_PATTERN.fullmatch(digest)), f"{context} is not a SHA-256 digest")
    return digest


def _tensor_is_finite(tensor):
    if tensor.is_sparse:
        tensor = tensor.coalesce().values()
    if tensor.is_floating_point() or tensor.is_complex():
        return bool(torch.isfinite(tensor).all())
    return True


def _feature_tensors(result):
    feature_result = result[0] if isinstance(result, tuple) else result
    if torch.is_tensor(feature_result):
        return [feature_result]
    _require(
        isinstance(feature_result, (list, tuple)),
        "get_feature returned an unsupported object",
    )
    tensors = list(feature_result)
    _require(tensors, "get_feature returned no tensors")
    _require(all(torch.is_tensor(value) for value in tensors), "get_feature returned non-tensor data")
    return tensors


def validate_cosine_receiver_checkpoint(
    checkpoint_path,
    args,
    expected_sha256=None,
    model_builder=None,
    random_seed=0,
):
    """Strict-load one final receiver and verify its cosine head and feature path."""
    checkpoint_path = Path(checkpoint_path).resolve()
    _require(checkpoint_path.is_file(), f"receiver checkpoint is missing: {checkpoint_path}")
    _require(str(getattr(args, "classifier_type", "")).lower() == "cosine", "receiver args are not cosine")
    _require(int(getattr(args, "nclass", 100)) == 100, "receiver output universe is not 100 classes")
    actual_sha = sha256_file(checkpoint_path)
    if expected_sha256 is not None:
        expected_sha = _valid_sha256(expected_sha256, "expected receiver checkpoint SHA")
        _require(actual_sha == expected_sha, f"receiver checkpoint SHA mismatch: {checkpoint_path}")

    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise LossAblationFinalAuditError(
            f"cannot load receiver checkpoint: {checkpoint_path}"
        ) from exc
    _require(isinstance(state, dict) and state, "receiver checkpoint is not a non-empty state dict")
    _require(all(isinstance(name, str) for name in state), "receiver checkpoint has a non-string key")
    _require(all(torch.is_tensor(value) for value in state.values()), "receiver state contains non-tensors")
    nonfinite = [name for name, value in state.items() if not _tensor_is_finite(value)]
    _require(not nonfinite, f"receiver checkpoint contains non-finite tensors: {nonfinite[:5]}")
    state_tensor_count = len(state)
    state_numel = sum(value.numel() for value in state.values())

    model = (model_builder or _default_model_builder)(args)
    _require(isinstance(model, torch.nn.Module), "model builder did not return torch.nn.Module")
    try:
        incompatible = model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise LossAblationFinalAuditError(
            f"strict checkpoint load failed: {checkpoint_path}: {exc}"
        ) from exc
    _require(
        not incompatible.missing_keys and not incompatible.unexpected_keys,
        "strict checkpoint load returned incompatible keys",
    )
    del state

    cosine_heads = [
        module
        for module in model.modules()
        if bool(getattr(module, "is_cosine_classifier", False))
    ]
    _require(len(cosine_heads) == 1, f"runtime model has {len(cosine_heads)} cosine heads")
    classifier = cosine_heads[0]
    _require(int(classifier.out_features) == 100, "runtime cosine head does not output 100 classes")
    _require(getattr(classifier, "bias", None) is None, "runtime cosine head unexpectedly has bias")
    scale = _finite_float(classifier.scale.detach().cpu().item(), "runtime cosine scale")
    _require(scale > 0.0, "runtime cosine scale is not positive")

    model.eval()
    model_nonfinite = [
        name
        for name, value in list(model.named_parameters()) + list(model.named_buffers())
        if not _tensor_is_finite(value.detach())
    ]
    _require(not model_nonfinite, f"loaded model contains non-finite tensors: {model_nonfinite[:5]}")

    generator = torch.Generator(device="cpu").manual_seed(int(random_seed))
    sample = torch.randn(
        2,
        int(args.nch),
        int(args.size),
        int(args.size),
        generator=generator,
    )
    with torch.no_grad():
        output = model(sample)
        _require(torch.is_tensor(output), "receiver output is not a tensor")
        _require(tuple(output.shape) == (2, 100), f"receiver output shape is {tuple(output.shape)}")
        _require(_tensor_is_finite(output), "receiver output contains non-finite values")
        feature_result = model.get_feature(
            sample,
            int(args.idx_from),
            int(getattr(args, "idx_to", -1)),
        )
    features = _feature_tensors(feature_result)
    for index, feature in enumerate(features):
        _require(feature.ndim >= 1 and feature.shape[0] == 2, f"feature {index} has invalid batch shape")
        _require(_tensor_is_finite(feature), f"feature {index} contains non-finite values")

    report = {
        "path": str(checkpoint_path),
        "sha256": actual_sha,
        "bytes": checkpoint_path.stat().st_size,
        "state_tensor_count": state_tensor_count,
        "state_numel": state_numel,
        "classifier_type": "cosine",
        "classifier_in_features": int(classifier.in_features),
        "classifier_out_features": int(classifier.out_features),
        "classifier_bias": False,
        "cosine_scale": scale,
        "output_shape": list(output.shape),
        "feature_shapes": [list(feature.shape) for feature in features],
    }
    del model, classifier, cosine_heads, output, features, feature_result, sample
    gc.collect()
    return report


def _read_exact_rows(path, context):
    path = Path(path)
    _require(path.is_file(), f"{context} social CSV is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames == SOCIAL_RESULT_FIELDS, f"{context} social CSV schema drifted")
        rows = list(reader)
    _require(len(rows) == 5, f"{context} social CSV has {len(rows)} rows; expected five")
    by_id = {_integer(row.get("receiver_agent"), f"{context} receiver id"): row for row in rows}
    _require(sorted(by_id) == EXPECTED_RECEIVERS and len(by_id) == 5, f"{context} receiver rows are incomplete")
    return path.resolve(), by_id


def _validate_loss_row(row, switches, context):
    fr, kd, sc = switches
    losses = {field: _finite_float(row.get(field), f"{context} {field}") for field in LOSS_FIELDS}
    _same(
        losses["loss_cls"],
        0.2 * losses["loss_ce_local"] + 0.8 * losses["loss_ce_external"],
        f"{context} class-loss identity",
        tolerance=LOSS_TOLERANCE,
    )
    for enabled, field in ((fr, "loss_fr"), (kd, "loss_kd"), (sc, "loss_sc")):
        if not enabled:
            _same(losses[field], 0.0, f"{context} inactive {field}", tolerance=1e-12)
    expected_total = losses["loss_cls"]
    expected_total += (0.2 if fr else 0.0) * losses["loss_fr"]
    expected_total += (0.6 if kd else 0.0) * losses["loss_kd"]
    expected_total += (0.1 if sc else 0.0) * losses["loss_sc"]
    _same(losses["loss"], expected_total, f"{context} total-loss identity", tolerance=LOSS_TOLERANCE)
    _same(row.get("loss_sender_kd"), losses["loss_kd"], f"{context} sender-KD identity")
    _same(row.get("loss_generalist_kd"), 0.0, f"{context} generalist-KD loss")
    return losses


def _validate_receiver_artifacts(
    run_dir,
    cfg,
    args,
    row,
    validated_row,
    switches,
    expert_sha,
    model_builder,
):
    condition = variant_id(*switches)
    receiver_id = int(args.agent_id)
    context = f"{condition} receiver {receiver_id}"
    variant = f"ablation_{condition}"
    _require(row.get("run_name") == cfg["project"]["run_name"], f"{context} run_name mismatch")
    _require(row.get("receiver_model") == EXPECTED_MODELS[receiver_id], f"{context} model mismatch")
    _require(row.get("classifier_type") == "cosine", f"{context} result head is not cosine")
    _require(row.get("dkp_variant") == variant, f"{context} variant mismatch")
    _require(row.get("checkpoint_retention") == "final_only", f"{context} is not final_only")
    _require(_integer(row.get("optimizer_steps"), f"{context} steps") == 9_420, f"{context} step count mismatch")
    _require(_integer(row.get("external_comm_images"), f"{context} images") == 800, f"{context} image count mismatch")
    _require(
        _integer(row.get("prototype_initialized_classes"), f"{context} prototypes") == 80,
        f"{context} prototype count mismatch",
    )
    expected_bytes = 128_000 if switches[1] else 0
    _require(
        _integer(row.get("external_comm_logit_bytes"), f"{context} logit bytes") == expected_bytes,
        f"{context} logit-byte count mismatch",
    )
    _require(row.get("receiver_init_checkpoint_sha256") == expert_sha, f"{context} init SHA mismatch")
    _require(row.get("fr_teacher_checkpoint_sha256") == expert_sha, f"{context} FR SHA mismatch")
    losses = _validate_loss_row(row, switches, context)

    metrics = {
        "global": _finite_float(row.get("acc_global_after"), f"{context} global accuracy"),
        "new": _finite_float(row.get("acc_new_after"), f"{context} new accuracy"),
        "expert": _finite_float(row.get("acc_expert_after"), f"{context} expert accuracy"),
    }
    for key, value in metrics.items():
        _require(0.0 <= value <= 100.0, f"{context} {key} accuracy is outside [0,100]")
    _same(
        row.get("forgetting"),
        _finite_float(row.get("acc_expert_before"), f"{context} expert-before") - metrics["expert"],
        f"{context} forgetting identity",
        tolerance=LOSS_TOLERANCE,
    )

    checkpoint_dir = (
        Path(run_dir)
        / f"social_learning/receiver_agent_{receiver_id}/checkpoints/dkp_sl_v1_{variant}"
    )
    final_path = checkpoint_dir / "after_social.pt"
    provenance_path = checkpoint_dir / "receiver_provenance.json"
    _require(checkpoint_dir.is_dir(), f"{context} checkpoint directory is missing")
    retained = {path.resolve() for path in checkpoint_dir.rglob("*") if path.is_file()}
    _require(
        retained == {final_path.resolve(), provenance_path.resolve()},
        f"{context} final_only directory contains unexpected artifacts",
    )
    final_sha = sha256_file(final_path)
    _require(final_sha == _valid_sha256(row.get("after_social_checkpoint_sha256"), f"{context} result final SHA"), f"{context} result final SHA mismatch")
    declared_provenance = _resolve(row.get("receiver_provenance_path", ""))
    _require(declared_provenance == provenance_path.resolve(), f"{context} provenance path mismatch")
    provenance_sha = sha256_file(provenance_path)
    _require(
        provenance_sha == _valid_sha256(row.get("receiver_provenance_sha256"), f"{context} provenance SHA"),
        f"{context} provenance SHA mismatch",
    )

    provenance = _load_json(provenance_path, f"{context} provenance")
    expected_switches = {"fr": switches[0], "kd": switches[1], "supcon": switches[2]}
    _require(provenance.get("protocol") == "dkp_sl_v1", f"{context} provenance protocol mismatch")
    _require(provenance.get("dkp_variant") == variant, f"{context} provenance variant mismatch")
    _require(provenance.get("loss_switches") == expected_switches, f"{context} provenance switches mismatch")
    _require(_integer(provenance.get("receiver_agent"), f"{context} provenance id") == receiver_id, f"{context} provenance id mismatch")
    _require(provenance.get("receiver_model") == args.model_name, f"{context} provenance model mismatch")
    _require(_integer(provenance.get("receiver_seed"), f"{context} provenance seed") == 100_000 + receiver_id, f"{context} provenance seed mismatch")
    _require(provenance.get("receiver_init_checkpoint_sha256") == expert_sha, f"{context} provenance init SHA mismatch")
    _require(provenance.get("fr_teacher_checkpoint_sha256") == expert_sha, f"{context} provenance FR SHA mismatch")
    _require(provenance.get("classifier_type") == "cosine", f"{context} provenance head mismatch")
    _require(provenance.get("checkpoint_retention") == "final_only", f"{context} provenance retention mismatch")

    prototype = provenance.get("prototype_initialization", {})
    _require(prototype.get("classifier_type") == "cosine", f"{context} prototype head mismatch")
    _require(prototype.get("mode") == "cosine_unit_weight_rows", f"{context} prototype mode mismatch")
    _require(prototype.get("local_rows_preserved") is True, f"{context} local prototype rows were not preserved")
    _same(prototype.get("external_weight_norm_min"), 1.0, f"{context} prototype minimum norm", tolerance=1e-6)
    _same(prototype.get("external_weight_norm_max"), 1.0, f"{context} prototype maximum norm", tolerance=1e-6)
    expected_external = sorted(set(range(100)) - set(int(value) for value in args.active_class_ids))
    initialized = [_integer(value, f"{context} initialized class") for value in prototype.get("initialized_classes", [])]
    _require(initialized == expected_external, f"{context} initialized class order mismatch")

    artifacts = provenance.get("checkpoint_artifacts", {})
    _require(set(artifacts) == {"after_social"}, f"{context} provenance retained extra checkpoints")
    artifact = artifacts["after_social"]
    _require(_resolve(artifact.get("path", "")) == final_path.resolve(), f"{context} provenance final path mismatch")
    _require(artifact.get("sha256") == final_sha, f"{context} provenance final SHA mismatch")

    statistics = provenance.get("statistics", {})
    _require(_integer(statistics.get("optimizer_steps"), f"{context} provenance steps") == 9_420, f"{context} provenance steps mismatch")
    _require(_integer(statistics.get("external_comm_images"), f"{context} provenance images") == 800, f"{context} provenance images mismatch")
    _require(_integer(statistics.get("external_comm_logit_bytes"), f"{context} provenance bytes") == expected_bytes, f"{context} provenance bytes mismatch")
    _same(statistics.get("training_seconds"), row.get("training_seconds"), f"{context} provenance training time", tolerance=1e-7)
    metrics_after = statistics.get("metrics_after", {})
    for key, csv_key in (
        ("acc_global", "acc_global_after"),
        ("acc_new", "acc_new_after"),
        ("acc_expert", "acc_expert_after"),
    ):
        _same(metrics_after.get(key), row.get(csv_key), f"{context} provenance {key}", tolerance=1e-7)
    metrics_before = statistics.get("metrics_before", {})
    for key, csv_key in (("acc_global", "acc_global_before"), ("acc_expert", "acc_expert_before")):
        _same(metrics_before.get(key), row.get(csv_key), f"{context} provenance {key}", tolerance=1e-7)
    provenance_losses = statistics.get("loss_means", {})
    for field in LOSS_FIELDS:
        _same(provenance_losses.get(field), losses[field], f"{context} provenance {field}", tolerance=1e-7)

    for metric in ("global", "new", "expert"):
        _same(validated_row["metrics"][metric], metrics[metric], f"{context} strict CSV {metric}")
    for field in LOSS_FIELDS:
        _same(validated_row["losses"][field], losses[field], f"{context} strict CSV {field}")

    checkpoint_report = validate_cosine_receiver_checkpoint(
        final_path,
        args,
        expected_sha256=final_sha,
        model_builder=model_builder,
        random_seed=40_000 + receiver_id,
    )
    checkpoint_report.update(
        {
            "receiver_agent": receiver_id,
            "receiver_model": args.model_name,
            "receiver_provenance": str(provenance_path.resolve()),
            "receiver_provenance_sha256": provenance_sha,
        }
    )
    return {
        "metrics": metrics,
        "losses": losses,
        "training_seconds": _finite_float(row.get("training_seconds"), f"{context} training time"),
        "optimizer_steps": 9_420,
        "raw_external_images": 800,
        "logit_bytes": expected_bytes,
        "prototype_initialized_classes": 80,
        "checkpoint": checkpoint_report,
    }


def _validate_summary(summary_path, contexts):
    summary_path = _resolve(summary_path)
    summary = _load_json(summary_path, "loss-ablation summary")
    _require(summary.get("status") == "complete_diagnostic", "summary status is not complete_diagnostic")
    _require(summary.get("formal_result") is False, "summary incorrectly claims a formal result")
    _require(summary.get("paper_eligible") is False, "summary incorrectly claims paper eligibility")
    _require(summary.get("receiver_ids") == EXPECTED_RECEIVERS, "summary receiver ids are incomplete")
    _require(_integer(summary.get("seed"), "summary seed") == 0, "summary seed mismatch")
    _require(_integer(summary.get("agents"), "summary agents") == 5, "summary agent count mismatch")
    _require(_integer(summary.get("classes_per_agent"), "summary classes") == 20, "summary class count mismatch")
    _require(_integer(summary.get("ipc"), "summary IPC") == 10, "summary IPC mismatch")
    expected_order = [variant_id(*switches) for switches in ALL_SWITCHES]
    _require(summary.get("condition_order") == expected_order, "summary 2^3 condition order is incomplete")

    sources = summary.get("sources", {})
    expert_path = contexts["expert_config"]
    _require(_resolve(sources.get("expert_config", "")) == expert_path, "summary expert config path mismatch")
    _require(sources.get("expert_config_sha256") == sha256_file(expert_path), "summary expert config SHA mismatch")
    for condition, context in contexts["conditions"].items():
        source = sources.get(condition, {})
        _require(_resolve(source.get("config", "")) == context["config_source"], f"summary {condition} config path mismatch")
        _require(source.get("config_sha256") == sha256_file(context["config_source"]), f"summary {condition} config SHA mismatch")
        _require(_resolve(source.get("run_dir", "")) == context["run_dir"], f"summary {condition} run path mismatch")
        _require(_resolve(source.get("social_results_csv", "")) == context["social_results_csv"], f"summary {condition} CSV path mismatch")
        _require(source.get("social_results_sha256") == sha256_file(context["social_results_csv"]), f"summary {condition} CSV SHA mismatch")

    entries = summary.get("per_receiver")
    _require(isinstance(entries, list) and len(entries) == 5, "summary lacks five receiver entries")
    by_id = {_integer(entry.get("receiver_agent"), "summary receiver id"): entry for entry in entries}
    _require(sorted(by_id) == EXPECTED_RECEIVERS and len(by_id) == 5, "summary receiver ids are incomplete")
    for receiver_id in EXPECTED_RECEIVERS:
        entry = by_id[receiver_id]
        _require(entry.get("receiver_model") == EXPECTED_MODELS[receiver_id], f"summary receiver {receiver_id} model mismatch")
        for condition, context in contexts["conditions"].items():
            source = context["receivers"][receiver_id]
            target = entry.get("conditions", {}).get(condition, {})
            for metric in ("global", "new", "expert"):
                _same(target.get(metric), source["metrics"][metric], f"summary {condition} receiver {receiver_id} {metric}")
            for field in LOSS_FIELDS:
                _same(target.get("loss_mean", {}).get(field), source["losses"][field], f"summary {condition} receiver {receiver_id} {field}")
            for key in (
                "training_seconds",
                "optimizer_steps",
                "raw_external_images",
                "logit_bytes",
                "prototype_initialized_classes",
            ):
                _same(target.get(key), source[key], f"summary {condition} receiver {receiver_id} {key}")

    aggregates = summary.get("mean_over_five_receivers", {})
    for condition, context in contexts["conditions"].items():
        target = aggregates.get(condition, {})
        switches = CONDITION_SWITCHES[condition]
        _require(
            target.get("switches")
            == {"fr": switches[0], "kd": switches[1], "supcon": switches[2]},
            f"summary {condition} switch set mismatch",
        )
        receivers = [context["receivers"][receiver_id] for receiver_id in EXPECTED_RECEIVERS]
        for metric in ("global", "new", "expert"):
            expected = sum(row["metrics"][metric] for row in receivers) / 5.0
            _same(target.get("metrics_mean", {}).get(metric), expected, f"summary {condition} mean {metric}")
        for field in LOSS_FIELDS:
            expected = sum(row["losses"][field] for row in receivers) / 5.0
            _same(target.get("loss_mean", {}).get(field), expected, f"summary {condition} mean {field}")
        resources = target.get("resources", {})
        for key in ("training_seconds", "optimizer_steps", "raw_external_images", "logit_bytes"):
            values = [row[key] for row in receivers]
            _same(resources.get(f"{key}_total"), sum(values), f"summary {condition} total {key}")
            _same(resources.get(f"{key}_mean"), sum(values) / 5.0, f"summary {condition} mean {key}")
    return {
        "path": str(summary_path),
        "sha256": sha256_file(summary_path),
        "status": summary["status"],
    }


def audit_loss_ablation_final_receivers(
    expert_config=DEFAULT_EXPERT_CONFIG,
    config_paths=None,
    expert_run_dir=None,
    run_dirs=None,
    summary_path=DEFAULT_SUMMARY,
    model_builder=None,
):
    """Audit all 30 final-only checkpoints without writing an artifact."""
    config_paths = default_config_paths() if config_paths is None else dict(config_paths)
    run_dirs = {} if run_dirs is None else dict(run_dirs)
    expert_path, expert_cfg, expert_args, validated_configs = validate_configs(
        expert_config,
        config_paths,
    )
    resolved_expert_run = resolve_run_dir(expert_cfg, override=expert_run_dir)
    experts = load_complete_experts(resolved_expert_run, expert_args, min_local_test_accuracy=20.0)
    builder = model_builder or _default_model_builder
    contexts = {"expert_config": expert_path.resolve(), "conditions": {}}

    for switches in MISSING_COMBINATIONS:
        condition = variant_id(*switches)
        source_path, source_cfg, _source_args = validated_configs[condition]
        run_dir = resolve_run_dir(source_cfg, override=run_dirs.get(condition))
        _require(run_dir.name == source_cfg["project"]["run_name"], f"{condition} run directory name mismatch")
        _require(run_dir.is_dir(), f"{condition} run directory is missing: {run_dir}")
        snapshot_path = run_dir / "config/main.yaml"
        _require(snapshot_path.is_file(), f"{condition} immutable config snapshot is missing")
        _require(sha256_file(snapshot_path) == sha256_file(source_path), f"{condition} config snapshot differs from source")
        snapshot_cfg = load_config(snapshot_path)
        _require(snapshot_cfg == source_cfg, f"{condition} parsed config snapshot differs from source")
        _, _, args_by_agent = validated_configs[condition]

        for receiver_id in EXPECTED_RECEIVERS:
            reused_expert = run_dir / f"agents/agent_{receiver_id}/checkpoints/expert_model.pt"
            _require(reused_expert.is_file(), f"{condition} agent {receiver_id} reused expert is missing")
            _require(
                sha256_file(reused_expert) == experts[receiver_id]["checkpoint_sha256"],
                f"{condition} agent {receiver_id} reused expert SHA mismatch",
            )

        social_path, rows = _read_exact_rows(run_dir / "metrics/social_results.csv", condition)
        validated_social = validate_social_run(
            run_dir,
            snapshot_cfg,
            args_by_agent,
            experts,
            switches,
        )
        expected_final_paths = {
            (
                run_dir
                / f"social_learning/receiver_agent_{receiver_id}/checkpoints/"
                f"dkp_sl_v1_ablation_{condition}/after_social.pt"
            ).resolve()
            for receiver_id in EXPECTED_RECEIVERS
        }
        discovered = {
            path.resolve()
            for path in run_dir.glob("social_learning/receiver_agent_*/checkpoints/**/after_social.pt")
        }
        _require(discovered == expected_final_paths, f"{condition} final checkpoint layout is incomplete")

        receiver_reports = {}
        checkpoint_reports = []
        for receiver_id in EXPECTED_RECEIVERS:
            report = _validate_receiver_artifacts(
                run_dir,
                snapshot_cfg,
                args_by_agent[receiver_id],
                rows[receiver_id],
                validated_social["receivers"][receiver_id],
                switches,
                experts[receiver_id]["checkpoint_sha256"],
                builder,
            )
            receiver_reports[receiver_id] = report
            checkpoint_reports.append(report["checkpoint"])
        contexts["conditions"][condition] = {
            "switches": {"fr": switches[0], "kd": switches[1], "supcon": switches[2]},
            "run_name": source_cfg["project"]["run_name"],
            "run_dir": run_dir.resolve(),
            "config_source": source_path.resolve(),
            "config_snapshot": snapshot_path.resolve(),
            "social_results_csv": social_path,
            "receivers": receiver_reports,
            "checkpoints": checkpoint_reports,
        }

    summary_report = _validate_summary(summary_path, contexts)
    return {
        "status": "passed",
        "formal_result": False,
        "paper_eligible": False,
        "scope": "ICLR 2027 seed0 six-condition cosine DKP loss-ablation final checkpoint diagnostic",
        "classifier": "cosine",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "receiver_ids": EXPECTED_RECEIVERS,
        "condition_count": 6,
        "checkpoint_count": 30,
        "summary": summary_report,
        "expert_run": {
            "path": str(Path(resolved_expert_run).resolve()),
            "checkpoint_sha256": {
                str(receiver_id): experts[receiver_id]["checkpoint_sha256"]
                for receiver_id in EXPECTED_RECEIVERS
            },
        },
        "runs": {
            condition: {
                "run_name": context["run_name"],
                "run_dir": str(context["run_dir"]),
                "switches": context["switches"],
                "config_source": str(context["config_source"]),
                "config_source_sha256": sha256_file(context["config_source"]),
                "config_snapshot": str(context["config_snapshot"]),
                "config_snapshot_sha256": sha256_file(context["config_snapshot"]),
                "social_results_csv": str(context["social_results_csv"]),
                "social_results_sha256": sha256_file(context["social_results_csv"]),
                "checkpoint_retention": "final_only",
                "checkpoints": context["checkpoints"],
            }
            for condition, context in contexts["conditions"].items()
        },
    }


build_audit = audit_loss_ablation_final_receivers


def _report_path(requested=None):
    diagnostics_dir = (ROOT / "outputs/diagnostics").resolve()
    output = DEFAULT_REPORT.resolve() if requested is None else _resolve(requested)
    _require(output.suffix.lower() == ".json", "loss-ablation audit output must be JSON")
    _require(output.parent == diagnostics_dir, "loss-ablation audit must stay directly in outputs/diagnostics")
    _require(not output.exists(), f"refusing to overwrite loss-ablation audit: {output}")
    return output


def _write_json_exclusive(payload, path):
    """Publish a complete JSON report without any overwrite race."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        raise LossAblationFinalAuditError(
            f"refusing to overwrite loss-ablation audit: {path}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-config", default=str(DEFAULT_EXPERT_CONFIG))
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--expert-run-dir")
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--output-json", default=str(DEFAULT_REPORT))
    return parser.parse_args(argv)


def main(argv=None):
    cli = parse_args(argv)
    try:
        output = _report_path(cli.output_json)
        report = audit_loss_ablation_final_receivers(
            expert_config=cli.expert_config,
            config_paths=default_config_paths(cli.config_dir),
            expert_run_dir=cli.expert_run_dir,
            summary_path=cli.summary_json,
        )
        _write_json_exclusive(report, output)
        print(json.dumps({"status": "passed", "output": str(output)}, indent=2))
        return 0
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, csv.Error) as exc:
        print(f"ICLR2027 DKP loss-ablation final-receiver audit FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
