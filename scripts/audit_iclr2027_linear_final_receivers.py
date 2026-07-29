#!/usr/bin/env python3
"""Fail-closed audit for the ten matched-linear DKP receiver checkpoints."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
import sys
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

from agent_data import build_agent_args  # noqa: E402
from config_adapter import load_config  # noqa: E402
from output_manager import atomic_write_json  # noqa: E402
from scripts.prepare_iclr2027_linear_head_ablation import (  # noqa: E402
    CE_ONLY_RUN,
    EXPERT_RUN,
    FULL_RUN,
)
from scripts.summarize_iclr2027_dkp_first_round import (  # noqa: E402
    EXPECTED_RECEIVERS,
    LOSS_FIELDS,
    validate_social_run,
)
from scripts.validate_iclr2027_cosine_experts import (  # noqa: E402
    resolve_run_dir,
    sha256_file,
)
from scripts.validate_iclr2027_linear_communication import (  # noqa: E402
    CE_CONFIG,
    EXPERT_CONFIG,
    FULL_CONFIG,
    validate_config_contract,
)
from social_output_manager import SOCIAL_RESULT_FIELDS  # noqa: E402


DEFAULT_EXPERT_CONFIG = ROOT / EXPERT_CONFIG
DEFAULT_CE_CONFIG = ROOT / CE_CONFIG
DEFAULT_FULL_CONFIG = ROOT / FULL_CONFIG
DEFAULT_CE_RUN = ROOT / "outputs" / CE_ONLY_RUN
DEFAULT_FULL_RUN = ROOT / "outputs" / FULL_RUN
DEFAULT_SUMMARY = DEFAULT_FULL_RUN / "metrics/linear_head_seed0_summary.json"
DEFAULT_REPORT_NAME = "linear_final_receiver_checkpoint_audit.json"
EXPECTED_MODELS = {
    0: "convnet3w1",
    1: "convnet4w15",
    2: "alexnet",
    3: "resnet10_standard",
    4: "resnet18_standard",
}
EXPECTED_RUNS = {"ce_only": CE_ONLY_RUN, "full": FULL_RUN}
SUMMARY_KEYS = {"ce_only": "ce_only", "full": "full_dkp_sl"}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class LinearFinalReceiverAuditError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise LinearFinalReceiverAuditError(message)


def _resolve(path):
    resolved = Path(path)
    return resolved.resolve() if resolved.is_absolute() else (ROOT / resolved).resolve()


def _load_json(path, context):
    path = Path(path)
    _require(path.is_file(), f"{context} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LinearFinalReceiverAuditError(f"cannot read {context}: {path}") from exc
    _require(isinstance(payload, dict), f"{context} is not a JSON object: {path}")
    return payload


def _finite_float(value, context):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise LinearFinalReceiverAuditError(f"{context} is not numeric: {value!r}") from exc
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
    normalized = str(value).strip().lower()
    _require(bool(SHA256_PATTERN.fullmatch(normalized)), f"{context} is not a SHA-256 digest")
    return normalized


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


def _default_model_builder(args):
    from train import define_model

    return define_model(args, 100)


def validate_linear_receiver_checkpoint(
    checkpoint_path,
    args,
    expected_sha256=None,
    model_builder=None,
    random_seed=0,
):
    """Strict-load one final linear receiver and verify state, output, and features."""
    checkpoint_path = Path(checkpoint_path).resolve()
    _require(checkpoint_path.is_file(), f"receiver checkpoint is missing: {checkpoint_path}")
    _require(str(getattr(args, "classifier_type", "")).lower() == "linear", "receiver args are not linear")
    _require(int(getattr(args, "nclass", 100)) == 100, "receiver output universe is not 100 classes")
    actual_sha = sha256_file(checkpoint_path)
    if expected_sha256 is not None:
        expected_sha = _valid_sha256(expected_sha256, "expected receiver checkpoint SHA")
        _require(actual_sha == expected_sha, f"receiver checkpoint SHA mismatch: {checkpoint_path}")

    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise LinearFinalReceiverAuditError(
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
        raise LinearFinalReceiverAuditError(
            f"strict checkpoint load failed: {checkpoint_path}: {exc}"
        ) from exc
    _require(
        not incompatible.missing_keys and not incompatible.unexpected_keys,
        "strict checkpoint load returned incompatible keys",
    )
    del state

    from models.cosine_classifier import get_output_classifier, get_output_classifier_type

    _require(get_output_classifier_type(model) == "linear", "runtime receiver head is not linear")
    classifier = get_output_classifier(model)
    _require(isinstance(classifier, torch.nn.Linear), "runtime receiver head is not nn.Linear")
    _require(int(classifier.out_features) == 100, "runtime linear head does not output 100 classes")
    _require(classifier.bias is not None, "runtime linear head has no bias")

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
        feature_result = model.get_feature(sample, int(args.idx_from), int(args.idx_to))
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
        "classifier_type": "linear",
        "classifier_in_features": int(classifier.in_features),
        "classifier_out_features": int(classifier.out_features),
        "classifier_bias": True,
        "output_shape": list(output.shape),
        "feature_shapes": [list(feature.shape) for feature in features],
    }
    del model, classifier, output, features, feature_result, sample
    gc.collect()
    return report


def _load_configs(expert_config, ce_config, full_config, ce_run_dir, full_run_dir):
    paths = {
        "expert": _resolve(expert_config),
        "ce_only": _resolve(ce_config),
        "full": _resolve(full_config),
    }
    for role, path in paths.items():
        _require(path.is_file(), f"{role} config is missing: {path}")
    configs = {role: load_config(path) for role, path in paths.items()}
    try:
        validate_config_contract(configs["expert"], configs["ce_only"], configs["full"])
    except RuntimeError as exc:
        raise LinearFinalReceiverAuditError(f"matched-linear config contract failed: {exc}") from exc

    runs = {
        "ce_only": resolve_run_dir(configs["ce_only"], override=ce_run_dir),
        "full": resolve_run_dir(configs["full"], override=full_run_dir),
    }
    snapshots = {}
    args_by_variant = {}
    for variant, run_dir in runs.items():
        run_dir = Path(run_dir).resolve()
        _require(run_dir.name == EXPECTED_RUNS[variant], f"{variant} run directory name drifted")
        _require(run_dir.is_dir(), f"{variant} run directory is missing: {run_dir}")
        snapshot_path = run_dir / "config/main.yaml"
        _require(snapshot_path.is_file(), f"{variant} immutable config snapshot is missing")
        _require(
            sha256_file(snapshot_path) == sha256_file(paths[variant]),
            f"{variant} config snapshot differs from the configured source",
        )
        snapshot = load_config(snapshot_path)
        snapshots[variant] = {
            "run_dir": run_dir,
            "config_path": snapshot_path.resolve(),
            "config_sha256": sha256_file(snapshot_path),
            "config": snapshot,
        }
        args_by_variant[variant] = {
            agent_id: build_agent_args(snapshot, snapshot_path, agent_id)
            for agent_id in EXPECTED_RECEIVERS
        }
        for agent_id, args in args_by_variant[variant].items():
            _require(args.model_name == EXPECTED_MODELS[agent_id], f"{variant} agent {agent_id} model drifted")
            _require(args.classifier_type == "linear", f"{variant} agent {agent_id} head is not linear")
            _require(int(args.nclass) == 100, f"{variant} agent {agent_id} output dim drifted")

    try:
        validate_config_contract(configs["expert"], snapshots["ce_only"]["config"], snapshots["full"]["config"])
    except RuntimeError as exc:
        raise LinearFinalReceiverAuditError(f"run config snapshot contract failed: {exc}") from exc
    return paths, configs, snapshots, args_by_variant


def _summary_source(summary, path_key, sha_key, expected_path, context):
    sources = summary.get("sources", {})
    _require(path_key in sources and sha_key in sources, f"summary is missing {context} source fields")
    declared = _resolve(sources[path_key])
    expected = Path(expected_path).resolve()
    _require(declared == expected, f"summary {context} path mismatch")
    declared_sha = _valid_sha256(sources[sha_key], f"summary {context} SHA")
    _require(sha256_file(declared) == declared_sha, f"summary {context} declared SHA mismatch")
    return {"path": str(declared), "sha256": declared_sha}


def _load_summary(summary_path, paths, snapshots):
    summary_path = _resolve(summary_path)
    summary = _load_json(summary_path, "matched-linear summary")
    _require(summary.get("status") == "complete_diagnostic", "linear summary is not complete_diagnostic")
    _require(summary.get("formal_result") is False, "linear summary incorrectly claims a formal result")
    _require(summary.get("paper_eligible") is False, "linear summary incorrectly claims paper eligibility")
    _require(summary.get("classifier") == "linear", "linear summary classifier marker drifted")
    _require(summary.get("dataset") == "cifar100", "linear summary dataset drifted")
    _require(_integer(summary.get("seed"), "summary seed") == 0, "linear summary seed drifted")
    _require(_integer(summary.get("agents"), "summary agents") == 5, "linear summary agent count drifted")
    _require(_integer(summary.get("classes_per_agent"), "summary classes") == 20, "summary classes drifted")
    _require(_integer(summary.get("ipc"), "summary IPC") == 10, "linear summary IPC drifted")
    _require(summary.get("receiver_ids") == EXPECTED_RECEIVERS, "linear summary receiver ids are incomplete")

    source_reports = {
        "expert_config": _summary_source(
            summary, "expert_config", "expert_config_sha256", paths["expert"], "expert config"
        ),
        "ce_only_config": _summary_source(
            summary, "ce_only_config", "ce_only_config_sha256", paths["ce_only"], "CE config"
        ),
        "full_config": _summary_source(
            summary, "full_config", "full_config_sha256", paths["full"], "full config"
        ),
        "ce_only_social_results": _summary_source(
            summary,
            "ce_only_social_results",
            "ce_only_social_results_sha256",
            snapshots["ce_only"]["run_dir"] / "metrics/social_results.csv",
            "CE social CSV",
        ),
        "full_social_results": _summary_source(
            summary,
            "full_social_results",
            "full_social_results_sha256",
            snapshots["full"]["run_dir"] / "metrics/social_results.csv",
            "full social CSV",
        ),
    }
    expert_run = _resolve(summary.get("sources", {}).get("expert_run_dir", ""))
    _require(expert_run.name == EXPERT_RUN, "linear summary expert run path drifted")

    entries = summary.get("per_receiver")
    _require(isinstance(entries, list) and len(entries) == 5, "linear summary lacks five receiver entries")
    by_id = {_integer(entry.get("receiver_agent"), "summary receiver id"): entry for entry in entries}
    _require(sorted(by_id) == EXPECTED_RECEIVERS and len(by_id) == 5, "summary receiver ids are incomplete")
    experts = {}
    for receiver_id in EXPECTED_RECEIVERS:
        entry = by_id[receiver_id]
        _require(entry.get("receiver_model") == EXPECTED_MODELS[receiver_id], "summary model assignment drifted")
        local = entry.get("local_linear_expert", {})
        accuracy = _finite_float(local.get("expert"), f"receiver {receiver_id} local expert accuracy")
        _require(0.0 <= accuracy <= 100.0, "summary local expert accuracy is outside [0,100]")
        expert_sha = _valid_sha256(local.get("checkpoint_sha256"), f"receiver {receiver_id} expert SHA")
        _require(_integer(local.get("selected_epoch"), "selected epoch") > 0, "selected epoch is invalid")
        experts[receiver_id] = {
            "official_test_accuracy_report_only": accuracy,
            "checkpoint_sha256": expert_sha,
        }
    return summary_path, summary, by_id, experts, source_reports


def _read_exact_social_rows(path, variant):
    path = Path(path)
    _require(path.is_file(), f"{variant} social CSV is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames == SOCIAL_RESULT_FIELDS, f"{variant} social CSV schema drifted")
        rows = list(reader)
    _require(len(rows) == 5, f"{variant} social CSV has {len(rows)} rows; expected five")
    by_id = {_integer(row.get("receiver_agent"), f"{variant} receiver id"): row for row in rows}
    _require(sorted(by_id) == EXPECTED_RECEIVERS and len(by_id) == 5, f"{variant} receiver rows are incomplete")
    return by_id


def _validate_social_csv(snapshot, args_by_agent, experts, variant):
    path = snapshot["run_dir"] / "metrics/social_results.csv"
    rows = _read_exact_social_rows(path, variant)
    try:
        validated = validate_social_run(
            snapshot["run_dir"], snapshot["config"], args_by_agent, experts, variant
        )
    except RuntimeError as exc:
        raise LinearFinalReceiverAuditError(f"{variant} social CSV failed strict validation: {exc}") from exc
    validated["rows"] = rows
    return validated


def _validate_linear_provenance(snapshot, args, row, variant, receiver_id, expert_sha):
    context = f"linear {variant} receiver {receiver_id}"
    _require(row.get("classifier_type") == "linear", f"{context} result head is not linear")
    _require(row.get("prototype_init_mode") == "linear_local_row_norm_bias_mean", f"{context} mode drifted")
    alpha = _finite_float(row.get("prototype_alpha"), f"{context} alpha")
    beta = _finite_float(row.get("prototype_beta"), f"{context} beta")
    norm_min = _finite_float(row.get("prototype_weight_norm_min"), f"{context} minimum norm")
    norm_max = _finite_float(row.get("prototype_weight_norm_max"), f"{context} maximum norm")
    _require(alpha > 0.0, f"{context} alpha is not positive")
    _same(norm_min, alpha, f"{context} minimum norm", tolerance=1e-6)
    _same(norm_max, alpha, f"{context} maximum norm", tolerance=1e-6)
    _require(row.get("checkpoint_retention") == "final_only", f"{context} is not final_only")
    _require(row.get("receiver_init_checkpoint_sha256") == expert_sha, f"{context} init SHA drifted")
    _require(row.get("fr_teacher_checkpoint_sha256") == expert_sha, f"{context} FR SHA drifted")

    checkpoint_dir = (
        snapshot["run_dir"]
        / f"social_learning/receiver_agent_{receiver_id}/checkpoints/dkp_sl_v1_{variant}"
    )
    final_path = checkpoint_dir / "after_social.pt"
    provenance_path = checkpoint_dir / "receiver_provenance.json"
    _require(checkpoint_dir.is_dir(), f"{context} checkpoint directory is missing")
    retained_states = sorted(path.resolve() for path in checkpoint_dir.rglob("*.pt"))
    _require(retained_states == [final_path.resolve()], f"{context} retained non-final checkpoints")
    _require(final_path.is_file(), f"{context} after_social checkpoint is missing")
    final_sha = sha256_file(final_path)
    _require(final_sha == row.get("after_social_checkpoint_sha256"), f"{context} result SHA mismatch")
    declared_provenance = _resolve(row.get("receiver_provenance_path", ""))
    _require(declared_provenance == provenance_path.resolve(), f"{context} provenance path drifted")
    provenance_sha = sha256_file(provenance_path)
    _require(provenance_sha == row.get("receiver_provenance_sha256"), f"{context} provenance SHA mismatch")

    provenance = _load_json(provenance_path, f"{context} provenance")
    _require(provenance.get("protocol") == "dkp_sl_v1", f"{context} provenance protocol drifted")
    _require(provenance.get("dkp_variant") == variant, f"{context} provenance variant drifted")
    _require(_integer(provenance.get("receiver_agent"), f"{context} provenance agent") == receiver_id, "provenance id drifted")
    _require(provenance.get("receiver_model") == args.model_name, f"{context} provenance model drifted")
    _require(provenance.get("classifier_type") == "linear", f"{context} provenance head drifted")
    _require(provenance.get("checkpoint_retention") == "final_only", f"{context} provenance retention drifted")
    _require(
        provenance.get("loss_switches")
        == snapshot["config"]["social_learning"]["receiver"]["loss_switches"],
        f"{context} provenance loss switches drifted",
    )
    _require(provenance.get("receiver_init_checkpoint_sha256") == expert_sha, "provenance init SHA drifted")
    _require(provenance.get("fr_teacher_checkpoint_sha256") == expert_sha, "provenance FR SHA drifted")

    prototype = provenance.get("prototype_initialization", {})
    _require(prototype.get("classifier_type") == "linear", f"{context} prototype head drifted")
    _require(prototype.get("mode") == "linear_local_row_norm_bias_mean", f"{context} prototype mode drifted")
    _same(prototype.get("alpha"), alpha, f"{context} provenance alpha", tolerance=1e-7)
    _same(prototype.get("beta"), beta, f"{context} provenance beta", tolerance=1e-7)
    _same(prototype.get("external_weight_norm_min"), alpha, f"{context} provenance min norm", tolerance=1e-6)
    _same(prototype.get("external_weight_norm_max"), alpha, f"{context} provenance max norm", tolerance=1e-6)
    expected_external = sorted(set(range(100)) - set(int(value) for value in args.active_class_ids))
    initialized = [_integer(value, f"{context} initialized class") for value in prototype.get("initialized_classes", [])]
    _require(initialized == expected_external, f"{context} initialized-class provenance drifted")

    artifacts = provenance.get("checkpoint_artifacts", {})
    _require(set(artifacts) == {"after_social"}, f"{context} provenance retained redundant checkpoints")
    artifact = artifacts["after_social"]
    _require(_resolve(artifact.get("path", "")) == final_path.resolve(), f"{context} final path drifted")
    _require(artifact.get("sha256") == final_sha, f"{context} final provenance SHA drifted")

    statistics = provenance.get("statistics", {})
    _require(
        _integer(statistics.get("optimizer_steps"), f"{context} provenance steps")
        == _integer(row.get("optimizer_steps"), f"{context} result steps"),
        f"{context} optimizer-step provenance drifted",
    )
    _require(
        _integer(statistics.get("external_comm_images"), f"{context} provenance images")
        == _integer(row.get("external_comm_images"), f"{context} result images"),
        f"{context} communication-image provenance drifted",
    )
    _require(
        _integer(statistics.get("external_comm_logit_bytes"), f"{context} provenance bytes")
        == _integer(row.get("external_comm_logit_bytes"), f"{context} result bytes"),
        f"{context} logit-byte provenance drifted",
    )
    metrics_after = statistics.get("metrics_after", {})
    for key, csv_key in (
        ("acc_global", "acc_global_after"),
        ("acc_new", "acc_new_after"),
        ("acc_expert", "acc_expert_after"),
    ):
        _same(metrics_after.get(key), row.get(csv_key), f"{context} provenance {key}", tolerance=1e-7)
    loss_means = statistics.get("loss_means", {})
    for field in LOSS_FIELDS:
        _same(loss_means.get(field), row.get(field), f"{context} provenance {field}", tolerance=1e-7)

    return {
        "prototype_alpha": alpha,
        "prototype_beta": beta,
        "prototype_weight_norm_min": norm_min,
        "prototype_weight_norm_max": norm_max,
        "after_social_checkpoint": str(final_path.resolve()),
        "after_social_checkpoint_sha256": final_sha,
        "receiver_provenance": str(provenance_path.resolve()),
        "receiver_provenance_sha256": provenance_sha,
    }


def _validate_summary_values(summary, summary_entries, social, provenance):
    for receiver_id in EXPECTED_RECEIVERS:
        entry = summary_entries[receiver_id]
        for variant in ("ce_only", "full"):
            target = entry.get(SUMMARY_KEYS[variant], {})
            validated = social[variant]["receivers"][receiver_id]
            for metric in ("global", "new", "expert"):
                _same(target.get(metric), validated["metrics"][metric], f"summary {variant} receiver {receiver_id} {metric}")
            for loss in LOSS_FIELDS:
                _same(
                    target.get("loss_mean", {}).get(loss),
                    validated["losses"][loss],
                    f"summary {variant} receiver {receiver_id} {loss}",
                )
            for key in ("training_seconds", "optimizer_steps", "raw_external_images", "logit_bytes", "prototype_initialized_classes"):
                _same(target.get(key), validated[key], f"summary {variant} receiver {receiver_id} {key}")
            summary_prototype = target.get("linear_prototype", {})
            audited_prototype = provenance[variant][receiver_id]
            for key in (
                "prototype_alpha",
                "prototype_beta",
                "prototype_weight_norm_min",
                "prototype_weight_norm_max",
            ):
                _same(summary_prototype.get(key), audited_prototype[key], f"summary {variant} receiver {receiver_id} {key}")
            _require(
                summary_prototype.get("after_social_checkpoint_sha256")
                == audited_prototype["after_social_checkpoint_sha256"],
                f"summary {variant} receiver {receiver_id} final SHA drifted",
            )
            _require(
                _resolve(summary_prototype.get("receiver_provenance", ""))
                == Path(audited_prototype["receiver_provenance"]).resolve(),
                f"summary {variant} receiver {receiver_id} provenance path drifted",
            )
            _require(
                summary_prototype.get("receiver_provenance_sha256")
                == audited_prototype["receiver_provenance_sha256"],
                f"summary {variant} receiver {receiver_id} provenance SHA drifted",
            )

    aggregates = summary.get("mean_over_five_receivers", {})
    local_expected = sum(
        _finite_float(summary_entries[receiver_id]["local_linear_expert"]["expert"], "local expert")
        for receiver_id in EXPECTED_RECEIVERS
    ) / 5.0
    _same(aggregates.get("local_linear_expert", {}).get("expert"), local_expected, "summary local expert mean")
    for variant in ("ce_only", "full"):
        aggregate = aggregates.get(SUMMARY_KEYS[variant], {})
        receivers = [social[variant]["receivers"][receiver_id] for receiver_id in EXPECTED_RECEIVERS]
        for metric in ("global", "new", "expert"):
            expected = sum(row["metrics"][metric] for row in receivers) / 5.0
            _same(aggregate.get("metrics_mean", {}).get(metric), expected, f"summary {variant} mean {metric}")
        for loss in LOSS_FIELDS:
            expected = sum(row["losses"][loss] for row in receivers) / 5.0
            _same(aggregate.get("loss_mean", {}).get(loss), expected, f"summary {variant} mean {loss}")
        resources = aggregate.get("resources", {})
        for key in ("training_seconds", "optimizer_steps", "raw_external_images", "logit_bytes"):
            values = [row[key] for row in receivers]
            _same(resources.get(f"{key}_total"), sum(values), f"summary {variant} total {key}")
            _same(resources.get(f"{key}_mean"), sum(values) / 5.0, f"summary {variant} mean {key}")


def audit_linear_receiver_checkpoints(
    expert_config=DEFAULT_EXPERT_CONFIG,
    ce_config=DEFAULT_CE_CONFIG,
    full_config=DEFAULT_FULL_CONFIG,
    ce_run_dir=None,
    full_run_dir=None,
    summary_path=None,
    model_builder=None,
):
    """Audit both five-receiver linear runs without writing any artifact."""
    paths, _configs, snapshots, args_by_variant = _load_configs(
        expert_config,
        ce_config,
        full_config,
        ce_run_dir,
        full_run_dir,
    )
    resolved_summary = summary_path or (
        snapshots["full"]["run_dir"] / "metrics/linear_head_seed0_summary.json"
    )
    summary_path, summary, summary_entries, experts, source_reports = _load_summary(
        resolved_summary, paths, snapshots
    )
    social = {
        variant: _validate_social_csv(
            snapshots[variant], args_by_variant[variant], experts, variant
        )
        for variant in ("ce_only", "full")
    }

    provenance = {"ce_only": {}, "full": {}}
    checkpoint_reports = {"ce_only": [], "full": []}
    builder = model_builder or _default_model_builder
    for variant in ("ce_only", "full"):
        expected_final_paths = {
            (
                snapshots[variant]["run_dir"]
                / f"social_learning/receiver_agent_{receiver_id}/checkpoints/dkp_sl_v1_{variant}/after_social.pt"
            ).resolve()
            for receiver_id in EXPECTED_RECEIVERS
        }
        discovered = {
            path.resolve()
            for path in snapshots[variant]["run_dir"].glob(
                "social_learning/receiver_agent_*/checkpoints/**/after_social.pt"
            )
        }
        _require(discovered == expected_final_paths, f"{variant} final checkpoint layout is incomplete")
        for receiver_id in EXPECTED_RECEIVERS:
            args = args_by_variant[variant][receiver_id]
            row = social[variant]["rows"][receiver_id]
            provenance_report = _validate_linear_provenance(
                snapshots[variant],
                args,
                row,
                variant,
                receiver_id,
                experts[receiver_id]["checkpoint_sha256"],
            )
            checkpoint_report = validate_linear_receiver_checkpoint(
                provenance_report["after_social_checkpoint"],
                args,
                expected_sha256=provenance_report["after_social_checkpoint_sha256"],
                model_builder=builder,
                random_seed=30_000 + receiver_id,
            )
            checkpoint_report.update(
                {
                    "receiver_agent": receiver_id,
                    "receiver_model": args.model_name,
                    "receiver_provenance": provenance_report["receiver_provenance"],
                    "receiver_provenance_sha256": provenance_report[
                        "receiver_provenance_sha256"
                    ],
                }
            )
            provenance[variant][receiver_id] = provenance_report
            checkpoint_reports[variant].append(checkpoint_report)

    _validate_summary_values(summary, summary_entries, social, provenance)
    return {
        "status": "passed",
        "formal_result": False,
        "paper_eligible": False,
        "scope": "ICLR 2027 seed0 matched-linear CE/full final receiver checkpoint diagnostic",
        "classifier": "linear",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "receiver_ids": EXPECTED_RECEIVERS,
        "checkpoint_count": 10,
        "summary": {
            "path": str(summary_path),
            "sha256": sha256_file(summary_path),
            "status": summary["status"],
            "sources": source_reports,
        },
        "runs": {
            variant: {
                "run_name": EXPECTED_RUNS[variant],
                "run_dir": str(snapshots[variant]["run_dir"]),
                "config_snapshot": str(snapshots[variant]["config_path"]),
                "config_snapshot_sha256": snapshots[variant]["config_sha256"],
                "social_results_csv": social[variant]["social_results_csv"],
                "social_results_sha256": social[variant]["social_results_sha256"],
                "checkpoint_retention": "final_only",
                "checkpoints": checkpoint_reports[variant],
            }
            for variant in ("ce_only", "full")
        },
    }


build_audit = audit_linear_receiver_checkpoints


def _report_path(full_run_dir, requested=None):
    metrics_dir = Path(full_run_dir).resolve() / "metrics"
    output = metrics_dir / DEFAULT_REPORT_NAME if requested is None else _resolve(requested)
    _require(output.suffix.lower() == ".json", "linear receiver audit output must be JSON")
    _require(output.parent == metrics_dir.resolve(), "linear receiver audit must stay in full-run metrics")
    _require(not output.exists(), f"refusing to overwrite linear receiver audit: {output}")
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-config", default=str(DEFAULT_EXPERT_CONFIG))
    parser.add_argument("--ce-config", default=str(DEFAULT_CE_CONFIG))
    parser.add_argument("--full-config", default=str(DEFAULT_FULL_CONFIG))
    parser.add_argument("--ce-run-dir")
    parser.add_argument("--full-run-dir")
    parser.add_argument("--summary-json")
    parser.add_argument("--output-json")
    return parser.parse_args(argv)


def main(argv=None):
    cli = parse_args(argv)
    try:
        full_config = load_config(_resolve(cli.full_config))
        full_run = resolve_run_dir(full_config, override=cli.full_run_dir)
        output = _report_path(full_run, cli.output_json)
        report = audit_linear_receiver_checkpoints(
            expert_config=cli.expert_config,
            ce_config=cli.ce_config,
            full_config=cli.full_config,
            ce_run_dir=cli.ce_run_dir,
            full_run_dir=cli.full_run_dir,
            summary_path=cli.summary_json,
        )
        atomic_write_json(report, output)
        print(json.dumps({"status": "passed", "output": str(output)}, indent=2))
        return 0
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, csv.Error) as exc:
        print(f"ICLR2027 linear final-receiver audit FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
