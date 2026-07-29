#!/usr/bin/env python3
"""Fail-closed audit for the ten completed ICLR 2027 DKP receiver checkpoints."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_data import build_agent_args, get_agent_class_split, get_agent_model_split  # noqa: E402
from config_adapter import build_dsdm_args_from_config, load_config  # noqa: E402
from output_manager import atomic_write_json  # noqa: E402


EXPECTED_RECEIVERS = list(range(5))
EXPECTED_MODELS = {
    0: "convnet3w1",
    1: "convnet4w15",
    2: "alexnet",
    3: "resnet10_standard",
    4: "resnet18_standard",
}
EXPECTED_RUN_NAMES = {
    "ce_only": "cifar100_5agent20cls_dkp_ce_only_ipc10_seed0_v2",
    "full": "cifar100_5agent20cls_dkp_sl_full_ipc10_seed0_v2",
}
EXPERT_RUN_NAME = "cifar100_5agent20cls_dkp_cosine_experts_seed0_v1"
DEFAULT_CE_RUN = ROOT / "outputs" / EXPECTED_RUN_NAMES["ce_only"]
DEFAULT_FULL_RUN = ROOT / "outputs" / EXPECTED_RUN_NAMES["full"]
DEFAULT_EXPERT_RUN = ROOT / "outputs" / EXPERT_RUN_NAME
DEFAULT_REPORT_NAME = "final_receiver_checkpoint_audit.json"
# Immutable header written by the two completed v2 runs. Keep this audit
# contract independent from schema extensions used by future runs.
V2_SOCIAL_RESULT_FIELDS = [
    "run_name",
    "protocol",
    "dkp_variant",
    "receiver_agent",
    "receiver_model",
    "expert_classes",
    "packet_method",
    "method",
    "init_mode",
    "self_data_mode",
    "self_real_per_class",
    "use_fr",
    "lambda_fr",
    "lambda_sc",
    "supcon_temperature",
    "use_logits",
    "communication_mode",
    "use_generalist_logits",
    "kd_mix_beta",
    "lambda_kd",
    "kd_temperature",
    "ipc",
    "self_real_images",
    "external_comm_images",
    "external_comm_logit_bytes",
    "external_comm_generalist_logit_bytes",
    "acc_global_before",
    "acc_expert_before",
    "acc_global_after",
    "acc_expert_after",
    "acc_new_after",
    "forgetting",
    "loss_cls",
    "loss",
    "loss_ce_local",
    "loss_ce_external",
    "loss_fr",
    "loss_kd",
    "loss_sc",
    "loss_sender_kd",
    "loss_generalist_kd",
    "receiver_augment",
    "freeze_bn_stats",
    "optimizer_steps",
    "training_seconds",
    "prototype_initialized_classes",
    "receiver_seed",
    "receiver_init_checkpoint_sha256",
    "fr_teacher_checkpoint_sha256",
    "time",
]
LOSS_IDENTITY_ABS_TOLERANCE = 1e-7
LOSS_FIELDS = (
    "loss",
    "loss_cls",
    "loss_ce_local",
    "loss_ce_external",
    "loss_fr",
    "loss_kd",
    "loss_sc",
    "loss_sender_kd",
    "loss_generalist_kd",
)


class FinalReceiverAuditError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise FinalReceiverAuditError(message)


def _resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _sha256(path):
    path = Path(path)
    _require(path.is_file(), f"artifact is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path, context):
    path = Path(path)
    _require(path.is_file(), f"{context} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalReceiverAuditError(f"cannot read {context}: {path}") from exc
    _require(isinstance(payload, dict), f"{context} is not a JSON object: {path}")
    return payload


def _finite_float(value, context):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FinalReceiverAuditError(f"{context} is not numeric: {value!r}") from exc
    _require(math.isfinite(result), f"{context} is not finite: {result}")
    return result


def _integer(value, context):
    result = _finite_float(value, context)
    _require(result.is_integer(), f"{context} is not an integer: {result}")
    return int(result)


def _boolean(value, context):
    normalized = str(value).strip().lower()
    _require(normalized in {"true", "false"}, f"{context} is not boolean: {value!r}")
    return normalized == "true"


def _same(actual, expected, context, tolerance=1e-9):
    actual = _finite_float(actual, context)
    expected = _finite_float(expected, f"{context} expected value")
    _require(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance),
        f"{context} mismatch: {actual} != {expected}",
    )
    return actual


def _normalized_json(value):
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _validate_snapshot(run_dir, variant):
    run_dir = _resolve(run_dir)
    _require(run_dir.is_dir(), f"{variant} run directory is missing: {run_dir}")
    snapshot_path = run_dir / "config" / "main.yaml"
    _require(snapshot_path.is_file(), f"{variant} config snapshot is missing: {snapshot_path}")
    cfg = load_config(snapshot_path)

    expected_run_name = EXPECTED_RUN_NAMES[variant]
    _require(run_dir.name == expected_run_name, f"{variant} run directory name is not the expected v2 run")
    _require(cfg.get("project", {}).get("run_name") == expected_run_name, f"{variant} snapshot run_name mismatch")
    _require(cfg.get("project", {}).get("paper_eligible") is False, f"{variant} must remain diagnostic")
    _require(str(cfg.get("dataset", {}).get("name", "")).lower() == "cifar100", f"{variant} dataset mismatch")
    _require(_integer(cfg.get("dataset", {}).get("num_classes"), f"{variant} class count") == 100, f"{variant} output class count is not 100")

    class_split = get_agent_class_split(cfg)
    model_split = get_agent_model_split(cfg)
    _require(sorted(class_split) == EXPECTED_RECEIVERS, f"{variant} class split is incomplete")
    _require(model_split == EXPECTED_MODELS, f"{variant} model assignment mismatch: {model_split}")
    flattened = [int(class_id) for agent_id in EXPECTED_RECEIVERS for class_id in class_split[agent_id]]
    _require(all(len(class_split[agent_id]) == 20 for agent_id in EXPECTED_RECEIVERS), f"{variant} does not have 20 classes per agent")
    _require(len(flattened) == len(set(flattened)) == 100, f"{variant} class sets are not disjoint")
    _require(sorted(flattened) == list(range(100)), f"{variant} class sets do not cover global 0-99")

    receiver_cfg = cfg.get("social_learning", {}).get("receiver", {})
    communication_cfg = cfg.get("communication", {})
    logits_cfg = cfg.get("logits", {})
    _require(receiver_cfg.get("protocol") == "dkp_sl_v1", f"{variant} receiver protocol mismatch")
    _require(receiver_cfg.get("dkp_variant") == variant, f"{variant} DKP variant mismatch")
    _require(communication_cfg.get("strict_packet_validation") is True, f"{variant} strict packets are disabled")
    require_logits = variant == "full"
    _require(communication_cfg.get("use_sender_logits") is require_logits, f"{variant} sender-logit flag mismatch")
    _require(_integer(receiver_cfg.get("epochs"), f"{variant} receiver epochs") == 60, f"{variant} receiver epochs changed")
    _require(_integer(receiver_cfg.get("local_batch_size"), f"{variant} local batch size") == 64, f"{variant} local batch size changed")
    _same(receiver_cfg.get("lambda_fr"), 0.2 if require_logits else 0.0, f"{variant} lambda_FR")
    _same(receiver_cfg.get("lambda_sc"), 0.1 if require_logits else 0.0, f"{variant} lambda_SC")
    _same(logits_cfg.get("lambda_kd"), 0.6 if require_logits else 0.0, f"{variant} lambda_KD")

    args_by_agent = {}
    for agent_id in EXPECTED_RECEIVERS:
        args = build_agent_args(cfg, snapshot_path, agent_id)
        _require(str(args.classifier_type).lower() == "cosine", f"{variant} agent {agent_id} is not cosine")
        _require(_integer(args.nclass, f"{variant} agent {agent_id} output classes") == 100, f"{variant} agent {agent_id} output is not 100-dimensional")
        args_by_agent[agent_id] = args

    resolved_path = run_dir / "config" / "social_resolved_args.json"
    resolved = _load_json(resolved_path, f"{variant} resolved args")
    rebuilt = _normalized_json(vars(build_dsdm_args_from_config(cfg, config_path=snapshot_path)))
    recorded_config = _resolve(resolved.get("config_path", ""))
    _require(recorded_config.is_file(), f"{variant} resolved config_path is missing: {recorded_config}")
    _require(_sha256(recorded_config) == _sha256(snapshot_path), f"{variant} source config differs from immutable snapshot")
    rebuilt["config_path"] = resolved["config_path"]
    _require(resolved == rebuilt, f"{variant} resolved args do not reconstruct exactly from the snapshot")

    return {
        "run_dir": run_dir,
        "config_path": snapshot_path,
        "config_sha256": _sha256(snapshot_path),
        "resolved_args_path": resolved_path,
        "resolved_args_sha256": _sha256(resolved_path),
        "cfg": cfg,
        "args_by_agent": args_by_agent,
        "class_split": class_split,
        "model_split": model_split,
    }


def _validate_packet_integrity(snapshot, variant):
    path = snapshot["run_dir"] / "metrics" / "packet_integrity_dsdm.json"
    payload = _load_json(path, f"{variant} packet-integrity report")
    _require(payload.get("warnings") == [], f"{variant} packet-integrity report has warnings")
    packets = payload.get("packets")
    _require(isinstance(packets, list) and len(packets) == 5, f"{variant} packet-integrity report is incomplete")
    ids = [_integer(row.get("sender_agent"), f"{variant} packet sender") for row in packets]
    _require(sorted(ids) == EXPECTED_RECEIVERS and len(set(ids)) == 5, f"{variant} packet senders are incomplete")
    summary = payload.get("summary", {})
    _require(_integer(summary.get("total_raw_images"), f"{variant} raw packet images") == 1000, f"{variant} raw packet total mismatch")
    _require(_integer(summary.get("total_train_images"), f"{variant} decoded packet images") == 4000, f"{variant} decoded packet total mismatch")
    require_logits = variant == "full"
    for row in packets:
        agent_id = _integer(row["sender_agent"], f"{variant} packet sender")
        context = f"{variant} sender {agent_id}"
        _require(_boolean(row.get("has_sender_logits"), f"{context} logits") is require_logits, f"{context} logit presence mismatch")
        _require(_integer(row.get("raw_images"), f"{context} raw images") == 200, f"{context} raw image count mismatch")
        _require(_integer(row.get("decoded_or_train_images"), f"{context} decoded images") == 800, f"{context} decoded image count mismatch")
        _require(_integer(row.get("sender_logit_bytes"), f"{context} logit bytes") == (32000 if require_logits else 0), f"{context} logit bytes mismatch")
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _validate_expert_provenance(expert_run, snapshots):
    expert_run = _resolve(expert_run)
    _require(expert_run.name == EXPERT_RUN_NAME, "expert run is not the expected seed0 cosine run")
    preflight_path = expert_run / "metrics" / "cosine_expert_preflight.json"
    preflight = _load_json(preflight_path, "cosine expert preflight")
    _require(preflight.get("status") == "passed", "cosine expert preflight did not pass")
    rows = preflight.get("agents")
    _require(isinstance(rows, list) and len(rows) == 5, "cosine expert preflight is incomplete")
    by_id = {_integer(row.get("agent_id"), "expert preflight agent id"): row for row in rows}
    _require(sorted(by_id) == EXPECTED_RECEIVERS and len(by_id) == 5, "cosine expert preflight agent ids are incomplete")

    shas = {}
    local_accuracies = {}
    for agent_id in EXPECTED_RECEIVERS:
        row = by_id[agent_id]
        source_path = expert_run / "agents" / f"agent_{agent_id}" / "checkpoints" / "expert_model.pt"
        source_sha = _sha256(source_path)
        _require(row.get("checkpoint_sha256") == source_sha, f"expert agent {agent_id} preflight SHA mismatch")
        _require(row.get("model") == EXPECTED_MODELS[agent_id], f"expert agent {agent_id} model mismatch")
        _require(row.get("output_shape") == [2, 100], f"expert agent {agent_id} preflight output shape mismatch")
        local_accuracy = _finite_float(row.get("local_test_accuracy"), f"expert agent {agent_id} local accuracy")
        _require(0.0 <= local_accuracy <= 100.0, f"expert agent {agent_id} local accuracy is outside [0,100]")
        shas[agent_id] = source_sha
        local_accuracies[agent_id] = local_accuracy
        for variant, snapshot in snapshots.items():
            reused = snapshot["run_dir"] / "agents" / f"agent_{agent_id}" / "checkpoints" / "expert_model.pt"
            _require(_sha256(reused) == source_sha, f"{variant} agent {agent_id} reused expert SHA mismatch")

    return {
        "run_dir": str(expert_run),
        "preflight_path": str(preflight_path.resolve()),
        "preflight_sha256": _sha256(preflight_path),
        "checkpoint_sha256": shas,
        "local_test_accuracy": local_accuracies,
    }


def _validate_communication_preflight(full_snapshot, expert_shas):
    path = full_snapshot["run_dir"] / "metrics" / "communication_preflight.json"
    payload = _load_json(path, "communication preflight")
    _require(payload.get("status") == "passed", "communication preflight did not pass")
    _require(payload.get("protocol") == full_snapshot["cfg"]["communication"]["protocol"], "communication preflight protocol mismatch")

    expert_rows = payload.get("expert_provenance")
    _require(isinstance(expert_rows, list) and len(expert_rows) == 5, "communication preflight expert provenance is incomplete")
    expert_by_id = {_integer(row.get("agent_id"), "communication expert agent"): row for row in expert_rows}
    _require(sorted(expert_by_id) == EXPECTED_RECEIVERS, "communication preflight expert ids are incomplete")
    receiver_rows = payload.get("receivers")
    _require(isinstance(receiver_rows, list) and len(receiver_rows) == 5, "communication preflight receivers are incomplete")
    receiver_by_id = {_integer(row.get("receiver_agent"), "communication receiver id"): row for row in receiver_rows}
    _require(sorted(receiver_by_id) == EXPECTED_RECEIVERS, "communication preflight receiver ids are incomplete")

    for agent_id in EXPECTED_RECEIVERS:
        expected_sha = expert_shas[agent_id]
        _require(expert_by_id[agent_id].get("checkpoint_sha256") == expected_sha, f"communication expert {agent_id} SHA mismatch")
        row = receiver_by_id[agent_id]
        _require(row.get("receiver_model") == EXPECTED_MODELS[agent_id], f"communication receiver {agent_id} model mismatch")
        _require(row.get("init_checkpoint_sha256") == expected_sha, f"communication receiver {agent_id} init SHA mismatch")
        _require(row.get("fr_teacher_checkpoint_sha256") == expected_sha, f"communication receiver {agent_id} FR SHA mismatch")
        _require(_integer(row.get("prototype_classes"), f"communication receiver {agent_id} prototypes") == 80, f"communication receiver {agent_id} prototype mismatch")
        _require(_integer(row.get("local_real_images"), f"communication receiver {agent_id} local images") == 10000, f"communication receiver {agent_id} local image mismatch")
        _require(_integer(row.get("external_decoded_images"), f"communication receiver {agent_id} decoded images") == 3200, f"communication receiver {agent_id} decoded image mismatch")
    return {"path": str(path.resolve()), "sha256": _sha256(path), "receivers": receiver_by_id}


def _read_csv(path, variant):
    path = Path(path)
    _require(path.is_file(), f"{variant} social CSV is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames == V2_SOCIAL_RESULT_FIELDS, f"{variant} social CSV schema mismatch")
        rows = list(reader)
    _require(len(rows) == 5, f"{variant} social CSV has {len(rows)} rows; expected exactly five")
    receiver_ids = [_integer(row.get("receiver_agent"), f"{variant} receiver id") for row in rows]
    _require(sorted(receiver_ids) == EXPECTED_RECEIVERS, f"{variant} receiver ids are incomplete: {receiver_ids}")
    _require(len(set(receiver_ids)) == 5, f"{variant} receiver ids are duplicated: {receiver_ids}")
    return {int(row["receiver_agent"]): row for row in rows}


def _validate_csv(snapshot, variant, expert_report):
    path = snapshot["run_dir"] / "metrics" / "social_results.csv"
    rows = _read_csv(path, variant)
    receiver_cfg = snapshot["cfg"]["social_learning"]["receiver"]
    logits_cfg = snapshot["cfg"]["logits"]
    require_logits = variant == "full"
    expected_steps = int(receiver_cfg["epochs"]) * math.ceil(10000 / int(receiver_cfg["local_batch_size"]))
    parsed = {}

    for receiver_id in EXPECTED_RECEIVERS:
        row = rows[receiver_id]
        context = f"{variant} receiver {receiver_id}"
        _require(row["run_name"] == EXPECTED_RUN_NAMES[variant], f"{context} run_name mismatch")
        _require(row["protocol"] == "dkp_sl_v1" and row["dkp_variant"] == variant, f"{context} protocol/variant mismatch")
        _require(row["receiver_model"] == EXPECTED_MODELS[receiver_id], f"{context} model mismatch")
        class_ids = [_integer(value, f"{context} class id") for value in row["expert_classes"].split(",")]
        _require(class_ids == snapshot["class_split"][receiver_id], f"{context} class order mismatch")
        _require(row["method"] == ("DKP_SL" if require_logits else "DKP_CE_ONLY"), f"{context} method mismatch")
        _require(row["packet_method"].lower() == "dsdm", f"{context} packet method mismatch")
        _require(row["init_mode"] == "expert" and row["self_data_mode"] == "real", f"{context} initialization/data mode mismatch")
        _require(_boolean(row["use_fr"], f"{context} use_fr") is require_logits, f"{context} FR flag mismatch")
        _require(_boolean(row["use_logits"], f"{context} use_logits") is require_logits, f"{context} logits flag mismatch")
        _require(not _boolean(row["use_generalist_logits"], f"{context} generalist logits"), f"{context} used generalist logits")
        _require(_integer(row["ipc"], f"{context} IPC") == 10, f"{context} IPC mismatch")
        _require(_integer(row["self_real_images"], f"{context} local images") == 10000, f"{context} local images mismatch")
        _require(_integer(row["external_comm_images"], f"{context} external raw images") == 800, f"{context} external raw image mismatch")
        _require(_integer(row["external_comm_logit_bytes"], f"{context} logit bytes") == (128000 if require_logits else 0), f"{context} logit byte mismatch")
        _require(_integer(row["external_comm_generalist_logit_bytes"], f"{context} generalist bytes") == 0, f"{context} generalist bytes are nonzero")
        _require(_integer(row["prototype_initialized_classes"], f"{context} prototypes") == 80, f"{context} prototype count mismatch")
        _require(_integer(row["optimizer_steps"], f"{context} optimizer steps") == expected_steps == 9420, f"{context} optimizer steps mismatch")
        _require(_integer(row["receiver_seed"], f"{context} seed") == 100000 + receiver_id, f"{context} seed mismatch")
        _require(row["receiver_init_checkpoint_sha256"] == expert_report["checkpoint_sha256"][receiver_id], f"{context} init SHA mismatch")
        _require(row["fr_teacher_checkpoint_sha256"] == expert_report["checkpoint_sha256"][receiver_id], f"{context} FR teacher SHA mismatch")
        _require(_finite_float(row["training_seconds"], f"{context} training time") > 0.0, f"{context} training time is not positive")
        _require(str(row["time"]).strip(), f"{context} completion time is missing")

        _same(row["lambda_fr"], receiver_cfg["lambda_fr"], f"{context} lambda_FR")
        _same(row["lambda_sc"], receiver_cfg["lambda_sc"], f"{context} lambda_SC")
        _same(row["lambda_kd"], logits_cfg["lambda_kd"], f"{context} lambda_KD")
        _same(row["supcon_temperature"], receiver_cfg["supcon_temperature"], f"{context} SupCon temperature")
        _same(row["kd_temperature"], logits_cfg["temperature"], f"{context} KD temperature")

        metrics = {
            name: _finite_float(row[field], f"{context} {field}")
            for name, field in (
                ("global_before", "acc_global_before"),
                ("expert_before", "acc_expert_before"),
                ("global", "acc_global_after"),
                ("expert", "acc_expert_after"),
                ("new", "acc_new_after"),
                ("forgetting", "forgetting"),
            )
        }
        for name, value in metrics.items():
            if name != "forgetting":
                _require(0.0 <= value <= 100.0, f"{context} {name} is outside [0,100]")
        _same(
            metrics["expert_before"],
            expert_report["local_test_accuracy"][receiver_id],
            f"{context} expert-before provenance",
        )
        _same(metrics["global_before"], 0.2 * metrics["expert_before"], f"{context} before metric identity")
        _same(metrics["global"], 0.2 * metrics["expert"] + 0.8 * metrics["new"], f"{context} global metric identity")
        _same(metrics["forgetting"], metrics["expert_before"] - metrics["expert"], f"{context} forgetting identity")

        losses = {field: _finite_float(row[field], f"{context} {field}") for field in LOSS_FIELDS}
        _require(all(value >= 0.0 for value in losses.values()), f"{context} contains a negative loss")
        _same(
            losses["loss_cls"],
            0.2 * losses["loss_ce_local"] + 0.8 * losses["loss_ce_external"],
            f"{context} CE identity",
            tolerance=LOSS_IDENTITY_ABS_TOLERANCE,
        )
        expected_total = (
            losses["loss_cls"]
            + float(receiver_cfg["lambda_fr"]) * losses["loss_fr"]
            + float(logits_cfg["lambda_kd"]) * losses["loss_kd"]
            + float(receiver_cfg["lambda_sc"]) * losses["loss_sc"]
        )
        _same(
            losses["loss"],
            expected_total,
            f"{context} total-loss identity",
            tolerance=LOSS_IDENTITY_ABS_TOLERANCE,
        )
        _same(losses["loss_sender_kd"], losses["loss_kd"], f"{context} sender-KD identity")
        _same(losses["loss_generalist_kd"], 0.0, f"{context} generalist-KD loss")
        if require_logits:
            for field in ("loss_fr", "loss_kd", "loss_sc"):
                _require(losses[field] > 0.0, f"{context} {field} is not positive")
        else:
            for field in ("loss_fr", "loss_kd", "loss_sc"):
                _same(losses[field], 0.0, f"{context} {field}")

        parsed[receiver_id] = {"metrics": metrics, "losses": losses, "row": row}

    return {"path": path.resolve(), "sha256": _sha256(path), "receivers": parsed}


def _default_model_builder(args):
    dsdm_root = ROOT / "DSDM"
    if str(dsdm_root) not in sys.path:
        sys.path.append(str(dsdm_root))
    from models import alexnet_cifar as alexnet_models
    from models import convnet as convnet_models
    from models import resnet as resnet_models

    classifier_type = str(args.classifier_type)
    scale = float(args.cosine_scale_init)
    if args.net_type == "convnet":
        return convnet_models.ConvNet(
            100,
            net_norm=args.norm_type,
            net_depth=args.depth,
            net_width=int(128 * args.width),
            channel=args.nch,
            im_size=(args.size, args.size),
            classifier_type=classifier_type,
            cosine_scale_init=scale,
        )
    if args.net_type == "alexnet":
        return alexnet_models.alexnet_cifar(
            100,
            nch=args.nch,
            classifier_type=classifier_type,
            cosine_scale_init=scale,
        )
    if args.net_type == "resnet_cifar_standard":
        return resnet_models.ResNet(
            args.dataset,
            args.depth,
            100,
            norm_type=args.norm_type,
            size=args.size,
            nch=args.nch,
            cifar_base_width=64,
            classifier_type=classifier_type,
            cosine_scale_init=scale,
        )
    raise FinalReceiverAuditError(f"unsupported audited model family: {args.net_type}")


def _tensor_is_finite(tensor):
    if tensor.is_sparse:
        tensor = tensor.coalesce().values()
    if tensor.is_floating_point() or tensor.is_complex():
        return bool(torch.isfinite(tensor).all())
    return True


def _feature_tensors(result):
    if isinstance(result, tuple):
        result = result[0]
    if torch.is_tensor(result):
        return [result]
    if isinstance(result, (list, tuple)):
        return [value for value in result if torch.is_tensor(value)]
    return []


def _audit_checkpoint(checkpoint_path, args, model_builder):
    checkpoint_path = Path(checkpoint_path)
    _require(checkpoint_path.is_file(), f"receiver checkpoint is missing: {checkpoint_path}")
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise FinalReceiverAuditError(f"cannot load receiver checkpoint: {checkpoint_path}") from exc
    _require(isinstance(state, dict) and state, f"receiver checkpoint is not a non-empty state dict: {checkpoint_path}")
    _require(all(isinstance(name, str) for name in state), f"receiver checkpoint has a non-string key: {checkpoint_path}")
    _require(all(torch.is_tensor(value) for value in state.values()), f"receiver checkpoint contains non-tensor state: {checkpoint_path}")
    nonfinite = [name for name, value in state.items() if not _tensor_is_finite(value)]
    _require(not nonfinite, f"receiver checkpoint contains non-finite tensors: {checkpoint_path}: {nonfinite[:5]}")

    model = model_builder(args)
    _require(isinstance(model, torch.nn.Module), "model builder did not return torch.nn.Module")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise FinalReceiverAuditError(f"strict checkpoint load failed: {checkpoint_path}: {exc}") from exc
    del state
    model.eval()
    model_nonfinite = [
        name for name, value in list(model.named_parameters()) + list(model.named_buffers())
        if not _tensor_is_finite(value.detach())
    ]
    _require(not model_nonfinite, f"loaded model contains non-finite tensors: {checkpoint_path}: {model_nonfinite[:5]}")

    input_size = int(args.size)
    sample = torch.linspace(-1.0, 1.0, steps=2 * int(args.nch) * input_size * input_size)
    sample = sample.reshape(2, int(args.nch), input_size, input_size)
    with torch.no_grad():
        output = model(sample)
        _require(torch.is_tensor(output), f"model output is not a tensor: {checkpoint_path}")
        _require(tuple(output.shape) == (2, 100), f"model output shape is {tuple(output.shape)}, expected (2, 100): {checkpoint_path}")
        _require(_tensor_is_finite(output), f"model output is non-finite: {checkpoint_path}")
        feature_result = model.get_feature(sample, int(args.idx_from), int(args.idx_from))
    features = _feature_tensors(feature_result)
    _require(features, f"get_feature returned no tensors: {checkpoint_path}")
    for index, feature in enumerate(features):
        _require(feature.ndim >= 1 and feature.shape[0] == 2, f"feature {index} has invalid batch shape: {checkpoint_path}")
        _require(_tensor_is_finite(feature), f"feature {index} is non-finite: {checkpoint_path}")

    report = {
        "path": str(checkpoint_path.resolve()),
        "sha256": _sha256(checkpoint_path),
        "bytes": checkpoint_path.stat().st_size,
        "state_tensor_count": len(model.state_dict()),
        "state_numel": sum(value.numel() for value in model.state_dict().values()),
        "output_shape": list(output.shape),
        "feature_shapes": [list(feature.shape) for feature in features],
    }
    del model, output, features, feature_result, sample
    gc.collect()
    return report


def _audit_run_checkpoints(snapshot, variant, model_builder):
    run_dir = snapshot["run_dir"]
    protocol = snapshot["cfg"]["social_learning"]["receiver"]["protocol"]
    expected_paths = {
        agent_id: run_dir
        / "social_learning"
        / f"receiver_agent_{agent_id}"
        / "checkpoints"
        / f"{protocol}_{variant}"
        / "after_social.pt"
        for agent_id in EXPECTED_RECEIVERS
    }
    discovered = sorted(run_dir.glob("social_learning/receiver_agent_*/checkpoints/**/after_social.pt"))
    _require(len(discovered) == 5, f"{variant} run has {len(discovered)} final receiver checkpoints; expected five")
    _require({path.resolve() for path in discovered} == {path.resolve() for path in expected_paths.values()}, f"{variant} final checkpoint layout mismatch")
    reports = []
    for agent_id in EXPECTED_RECEIVERS:
        report = _audit_checkpoint(expected_paths[agent_id], snapshot["args_by_agent"][agent_id], model_builder)
        report.update({"receiver_agent": agent_id, "receiver_model": EXPECTED_MODELS[agent_id]})
        reports.append(report)
    return reports


def _summary_source(summary, path_key, sha_key, expected_path, context, require_same_path=True):
    sources = summary.get("sources", {})
    _require(path_key in sources and sha_key in sources, f"summary is missing {context} source fields")
    declared_path = _resolve(sources[path_key])
    expected_path = Path(expected_path).resolve()
    if require_same_path:
        _require(declared_path == expected_path, f"summary {context} path mismatch")
    _require(_sha256(declared_path) == sources[sha_key], f"summary {context} declared SHA mismatch")
    _require(_sha256(expected_path) == sources[sha_key], f"summary {context} SHA differs from audited artifact")


def _validate_summary(summary_path, snapshots, csv_reports, expert_report):
    summary_path = _resolve(summary_path)
    summary = _load_json(summary_path, "first-round summary")
    _require(summary.get("status") == "complete_diagnostic", "first-round summary status is not complete_diagnostic")
    _require(summary.get("formal_result") is False, "first-round summary incorrectly claims a formal result")
    _require(summary.get("paper_eligible") is False, "first-round summary incorrectly claims paper eligibility")
    _require(summary.get("receiver_ids") == EXPECTED_RECEIVERS, "first-round summary receiver ids are incomplete")
    _require(_integer(summary.get("seed"), "summary seed") == 0, "summary seed mismatch")
    _require(_integer(summary.get("agents"), "summary agent count") == 5, "summary agent count mismatch")
    _require(_integer(summary.get("classes_per_agent"), "summary classes per agent") == 20, "summary classes-per-agent mismatch")
    _require(_integer(summary.get("ipc"), "summary IPC") == 10, "summary IPC mismatch")

    _summary_source(summary, "ce_only_social_results", "ce_only_social_results_sha256", csv_reports["ce_only"]["path"], "CE CSV")
    _summary_source(summary, "full_social_results", "full_social_results_sha256", csv_reports["full"]["path"], "full CSV")
    _summary_source(summary, "ce_only_config", "ce_only_config_sha256", snapshots["ce_only"]["config_path"], "CE config", require_same_path=False)
    _summary_source(summary, "full_config", "full_config_sha256", snapshots["full"]["config_path"], "full config", require_same_path=False)
    sources = summary.get("sources", {})
    _require(_resolve(sources.get("expert_run_dir", "")) == Path(expert_report["run_dir"]).resolve(), "summary expert run path mismatch")
    if "expert_config_sha256" in sources:
        expert_config = _resolve(sources.get("expert_config", ""))
        _require(_sha256(expert_config) == sources["expert_config_sha256"], "summary expert config SHA mismatch")

    entries = summary.get("per_receiver")
    _require(isinstance(entries, list) and len(entries) == 5, "summary per-receiver entries are incomplete")
    by_id = {_integer(row.get("receiver_agent"), "summary receiver id"): row for row in entries}
    _require(sorted(by_id) == EXPECTED_RECEIVERS and len(by_id) == 5, "summary receiver ids are duplicated or incomplete")
    for receiver_id in EXPECTED_RECEIVERS:
        entry = by_id[receiver_id]
        _require(entry.get("receiver_model") == EXPECTED_MODELS[receiver_id], f"summary receiver {receiver_id} model mismatch")
        _require(entry.get("local_expert", {}).get("checkpoint_sha256") == expert_report["checkpoint_sha256"][receiver_id], f"summary receiver {receiver_id} expert SHA mismatch")
        _same(
            entry.get("local_expert", {}).get("expert"),
            expert_report["local_test_accuracy"][receiver_id],
            f"summary receiver {receiver_id} local expert accuracy",
        )
        for variant, summary_key in (("ce_only", "ce_only"), ("full", "full_dkp_sl")):
            source = csv_reports[variant]["receivers"][receiver_id]
            target = entry.get(summary_key, {})
            for metric in ("global", "new", "expert"):
                _same(target.get(metric), source["metrics"][metric], f"summary {variant} receiver {receiver_id} {metric}")
            for loss in ("loss", "loss_cls", "loss_ce_local", "loss_ce_external", "loss_fr", "loss_kd", "loss_sc"):
                _same(target.get("loss_mean", {}).get(loss), source["losses"][loss], f"summary {variant} receiver {receiver_id} {loss}")
            row = source["row"]
            _same(target.get("training_seconds"), row["training_seconds"], f"summary {variant} receiver {receiver_id} training time")
            for key, csv_key in (
                ("optimizer_steps", "optimizer_steps"),
                ("raw_external_images", "external_comm_images"),
                ("logit_bytes", "external_comm_logit_bytes"),
                ("prototype_initialized_classes", "prototype_initialized_classes"),
            ):
                _require(_integer(target.get(key), f"summary {variant} receiver {receiver_id} {key}") == _integer(row[csv_key], f"CSV {variant} receiver {receiver_id} {csv_key}"), f"summary {variant} receiver {receiver_id} {key} mismatch")

    aggregate = summary.get("mean_over_five_receivers", {})
    local_expert_mean = sum(expert_report["local_test_accuracy"].values()) / 5.0
    _same(
        aggregate.get("local_expert", {}).get("expert"),
        local_expert_mean,
        "summary aggregate local expert accuracy",
    )
    if "official_test_accuracy_report_only_mean" in aggregate.get("local_expert", {}):
        _same(
            aggregate["local_expert"]["official_test_accuracy_report_only_mean"],
            local_expert_mean,
            "summary aggregate official local expert accuracy",
        )
    for variant, aggregate_key in (("ce_only", "ce_only"), ("full", "full_dkp_sl")):
        rows = [csv_reports[variant]["receivers"][receiver_id] for receiver_id in EXPECTED_RECEIVERS]
        target = aggregate.get(aggregate_key, {})
        for metric in ("global", "new", "expert"):
            expected = sum(row["metrics"][metric] for row in rows) / 5.0
            _same(target.get("metrics_mean", {}).get(metric), expected, f"summary {variant} aggregate {metric}")
        for loss in ("loss", "loss_cls", "loss_ce_local", "loss_ce_external", "loss_fr", "loss_kd", "loss_sc"):
            expected = sum(row["losses"][loss] for row in rows) / 5.0
            _same(target.get("loss_mean", {}).get(loss), expected, f"summary {variant} aggregate {loss}")
        resources = target.get("resources", {})
        expected_seconds = sum(float(row["row"]["training_seconds"]) for row in rows)
        _same(resources.get("training_seconds_total"), expected_seconds, f"summary {variant} total training time")
        _same(resources.get("training_seconds_mean"), expected_seconds / 5.0, f"summary {variant} mean training time")
        for key, csv_key in (
            ("optimizer_steps", "optimizer_steps"),
            ("raw_external_images", "external_comm_images"),
            ("logit_bytes", "external_comm_logit_bytes"),
        ):
            expected_total = sum(int(row["row"][csv_key]) for row in rows)
            _same(resources.get(f"{key}_total"), expected_total, f"summary {variant} total {key}")
            _same(resources.get(f"{key}_mean"), expected_total / 5.0, f"summary {variant} mean {key}")
    return {"path": str(summary_path), "sha256": _sha256(summary_path), "status": summary["status"]}


def build_audit(
    ce_run=DEFAULT_CE_RUN,
    full_run=DEFAULT_FULL_RUN,
    expert_run=DEFAULT_EXPERT_RUN,
    summary_path=None,
    model_builder=None,
):
    snapshots = {
        "ce_only": _validate_snapshot(ce_run, "ce_only"),
        "full": _validate_snapshot(full_run, "full"),
    }
    _require(snapshots["ce_only"]["class_split"] == snapshots["full"]["class_split"], "CE/full class partitions differ")
    _require(snapshots["ce_only"]["model_split"] == snapshots["full"]["model_split"], "CE/full model assignments differ")
    packet_reports = {
        variant: _validate_packet_integrity(snapshot, variant)
        for variant, snapshot in snapshots.items()
    }
    expert_report = _validate_expert_provenance(expert_run, snapshots)
    communication_report = _validate_communication_preflight(
        snapshots["full"], expert_report["checkpoint_sha256"]
    )
    csv_reports = {
        variant: _validate_csv(snapshot, variant, expert_report)
        for variant, snapshot in snapshots.items()
    }
    for receiver_id in EXPECTED_RECEIVERS:
        ce_row = csv_reports["ce_only"]["receivers"][receiver_id]["row"]
        full_row = csv_reports["full"]["receivers"][receiver_id]["row"]
        for sha_field in ("receiver_init_checkpoint_sha256", "fr_teacher_checkpoint_sha256"):
            _require(ce_row[sha_field] == full_row[sha_field], f"receiver {receiver_id} CE/full {sha_field} differs")

    resolved_summary = summary_path or (snapshots["full"]["run_dir"] / "metrics" / "first_round_seed0_summary.json")
    summary_report = _validate_summary(resolved_summary, snapshots, csv_reports, expert_report)
    builder = model_builder or _default_model_builder
    checkpoint_reports = {
        variant: _audit_run_checkpoints(snapshot, variant, builder)
        for variant, snapshot in snapshots.items()
    }

    return {
        "status": "passed",
        "formal_result": False,
        "paper_eligible": False,
        "scope": "ICLR 2027 seed0 5-agent IPC10 final receiver checkpoint audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "receiver_ids": EXPECTED_RECEIVERS,
        "preflights": {
            "cosine_experts": {
                "path": expert_report["preflight_path"],
                "sha256": expert_report["preflight_sha256"],
                "status": "passed",
            },
            "communication": {
                "path": communication_report["path"],
                "sha256": communication_report["sha256"],
                "status": "passed",
            },
        },
        "summary": summary_report,
        "runs": {
            variant: {
                "run_name": EXPECTED_RUN_NAMES[variant],
                "run_dir": str(snapshot["run_dir"]),
                "config_snapshot": str(snapshot["config_path"]),
                "config_snapshot_sha256": snapshot["config_sha256"],
                "resolved_args": str(snapshot["resolved_args_path"]),
                "resolved_args_sha256": snapshot["resolved_args_sha256"],
                "packet_integrity": packet_reports[variant],
                "social_results_csv": str(csv_reports[variant]["path"]),
                "social_results_sha256": csv_reports[variant]["sha256"],
                "checkpoints": checkpoint_reports[variant],
            }
            for variant, snapshot in snapshots.items()
        },
    }


def _report_path(full_run, requested=None):
    metrics_dir = _resolve(full_run) / "metrics"
    path = metrics_dir / DEFAULT_REPORT_NAME if requested is None else _resolve(requested)
    _require(path.suffix.lower() == ".json", "audit report must be JSON")
    _require(path.parent == metrics_dir.resolve(), "audit report must stay in the full v2 run metrics directory")
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ce-run-dir", default=str(DEFAULT_CE_RUN))
    parser.add_argument("--full-run-dir", default=str(DEFAULT_FULL_RUN))
    parser.add_argument("--expert-run-dir", default=str(DEFAULT_EXPERT_RUN))
    parser.add_argument("--summary-json")
    parser.add_argument("--write-report", action="store_true", help="Explicitly write the audit JSON after all gates pass")
    parser.add_argument("--output-json", help=f"Defaults to full-run metrics/{DEFAULT_REPORT_NAME}")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing audit JSON")
    return parser.parse_args(argv)


def main(argv=None):
    cli = parse_args(argv)
    try:
        report = build_audit(
            ce_run=cli.ce_run_dir,
            full_run=cli.full_run_dir,
            expert_run=cli.expert_run_dir,
            summary_path=cli.summary_json,
        )
        if cli.output_json and not cli.write_report:
            raise FinalReceiverAuditError("--output-json requires the explicit --write-report flag")
        if cli.overwrite and not cli.write_report:
            raise FinalReceiverAuditError("--overwrite requires the explicit --write-report flag")
        if cli.write_report:
            output = _report_path(cli.full_run_dir, cli.output_json)
            _require(cli.overwrite or not output.exists(), f"refusing to overwrite existing audit report: {output}")
            atomic_write_json(report, output)
            print(f"[final-receiver-audit] wrote {output}")
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    except Exception as exc:
        print(f"[final-receiver-audit] FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
