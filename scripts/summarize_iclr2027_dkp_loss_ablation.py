#!/usr/bin/env python3
"""Strictly summarize the complete seed0 2^3 DKP-SL loss ablation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_adapter import load_config  # noqa: E402
from output_manager import atomic_write_json  # noqa: E402
from scripts.prepare_iclr2027_dkp_loss_ablation import (  # noqa: E402
    FIXED_HYPERPARAMETERS,
    MISSING_COMBINATIONS,
    config_filename,
    run_name,
    validate_variant,
    variant_id,
)
from scripts.summarize_iclr2027_dkp_first_round import (  # noqa: E402
    EXPECTED_RECEIVERS,
    LOSS_FIELDS,
    REQUIRED_RESULT_FIELDS,
    load_complete_experts,
)
from scripts.validate_iclr2027_cosine_experts import (  # noqa: E402
    DEFAULT_CONFIG as DEFAULT_EXPERT_CONFIG,
    resolve_run_dir,
    sha256_file,
    validate_protocol_config,
)


DEFAULT_CE_CONFIG = ROOT / "configs/iclr2027/cifar100_5agent20cls_dkp_ce_only_ipc10_seed0_v2.yaml"
DEFAULT_FULL_CONFIG = ROOT / "configs/iclr2027/cifar100_5agent20cls_dkp_sl_full_ipc10_seed0_v2.yaml"
DEFAULT_CONFIG_DIR = ROOT / "configs/iclr2027"
DEFAULT_OUTPUT = ROOT / "outputs/diagnostics/iclr2027_cifar100_5agent20cls_ipc10_seed0_loss_ablation_v1.json"
ALL_SWITCHES = tuple(
    (fr, kd, sc)
    for fr in (False, True)
    for kd in (False, True)
    for sc in (False, True)
)
ENDPOINT_CONFIGS = {
    "fr0_kd0_sc0": DEFAULT_CE_CONFIG,
    "fr1_kd1_sc1": DEFAULT_FULL_CONFIG,
}
NEW_PROVENANCE_FIELDS = {
    "classifier_type",
    "prototype_init_mode",
    "prototype_alpha",
    "prototype_beta",
    "prototype_weight_norm_min",
    "prototype_weight_norm_max",
    "checkpoint_retention",
    "after_social_checkpoint_sha256",
    "receiver_provenance_path",
    "receiver_provenance_sha256",
}


class LossAblationSummaryError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise LossAblationSummaryError(message)


def _resolve_path(path):
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved.resolve()


def _finite_float(value, context):
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise LossAblationSummaryError(f"{context} is not numeric: {value!r}") from exc
    _require(math.isfinite(resolved), f"{context} is not finite: {resolved}")
    return resolved


def _integer(value, context):
    resolved = _finite_float(value, context)
    _require(resolved.is_integer(), f"{context} is not an integer: {resolved}")
    return int(resolved)


def _boolean(value, context):
    normalized = str(value).strip().lower()
    _require(normalized in {"true", "false"}, f"{context} is not a boolean: {value!r}")
    return normalized == "true"


def _same_float(actual, expected, context, tolerance=1e-12):
    _require(
        math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance),
        f"{context} mismatch: {actual} != {expected}",
    )


def default_config_paths(config_dir=DEFAULT_CONFIG_DIR):
    config_dir = Path(config_dir)
    paths = dict(ENDPOINT_CONFIGS)
    for fr, kd, sc in MISSING_COMBINATIONS:
        paths[variant_id(fr, kd, sc)] = config_dir / config_filename(fr, kd, sc)
    return {key: paths[key] for key in sorted(paths)}


def _expected_run_name(switches):
    if switches == (False, False, False):
        return "cifar100_5agent20cls_dkp_ce_only_ipc10_seed0_v2"
    if switches == (True, True, True):
        return "cifar100_5agent20cls_dkp_sl_full_ipc10_seed0_v2"
    return run_name(*switches)


def _expected_dkp_variant(switches):
    if switches == (False, False, False):
        return "ce_only"
    if switches == (True, True, True):
        return "full"
    return f"ablation_{variant_id(*switches)}"


def _validate_config(path, switches, expert_args, reference_cfg):
    path = _resolve_path(path)
    _require(path.is_file(), f"config is missing: {path}")
    cfg = load_config(path)
    args_by_agent = validate_protocol_config(cfg, path)
    condition = variant_id(*switches)
    project = cfg.get("project", {})
    receiver = cfg.get("social_learning", {}).get("receiver", {})
    communication = cfg.get("communication", {})
    logits = cfg.get("logits", {})
    fr, kd, sc = switches

    _require(project.get("run_name") == _expected_run_name(switches), f"{condition}: unexpected run_name")
    _require(project.get("paper_eligible") is False, f"{condition}: must remain paper-ineligible")
    _require(receiver.get("dkp_variant") == _expected_dkp_variant(switches), f"{condition}: dkp_variant mismatch")
    _require(receiver.get("protocol") == "dkp_sl_v1", f"{condition}: receiver protocol mismatch")
    _require(communication.get("strict_packet_validation") is True, f"{condition}: strict packets disabled")
    _require(communication.get("use_sender_logits") is kd, f"{condition}: sender-logit switch mismatch")
    _require(logits.get("enabled") is kd, f"{condition}: logits.enabled mismatch")
    _same_float(receiver.get("lambda_fr"), 0.2 if fr else 0.0, f"{condition}: lambda_FR")
    _same_float(logits.get("lambda_kd"), 0.6 if kd else 0.0, f"{condition}: lambda_KD")
    _same_float(receiver.get("lambda_sc"), 0.1 if sc else 0.0, f"{condition}: lambda_SC")
    _same_float(logits.get("temperature"), 2.0, f"{condition}: KD temperature")
    _same_float(receiver.get("supcon_temperature"), 0.07, f"{condition}: SupCon temperature")
    _require(int(receiver.get("epochs", -1)) == 60, f"{condition}: epochs drifted")
    _require(int(receiver.get("local_batch_size", -1)) == 64, f"{condition}: local batch size drifted")
    _require(int(receiver.get("external_batch_size", -1)) == 64, f"{condition}: external batch size drifted")
    _require(receiver.get("scheduler_milestones") == [39, 51], f"{condition}: milestones drifted")
    _require(int(cfg.get("runtime", {}).get("seed", -1)) == 0, f"{condition}: seed drifted")
    _require(int(cfg.get("distillation", {}).get("ipc", -1)) == 10, f"{condition}: IPC drifted")

    for agent_id in EXPECTED_RECEIVERS:
        _require(
            list(args_by_agent[agent_id].active_class_ids) == list(expert_args[agent_id].active_class_ids),
            f"{condition}: agent {agent_id} class split differs from expert run",
        )
        _require(
            args_by_agent[agent_id].model_name == expert_args[agent_id].model_name,
            f"{condition}: agent {agent_id} model differs from expert run",
        )
    _require(cfg.get("dataset") == reference_cfg.get("dataset"), f"{condition}: dataset config drifted")
    _require(cfg.get("pool_reuse") == reference_cfg.get("pool_reuse"), f"{condition}: pool provenance drifted")
    if switches in MISSING_COMBINATIONS:
        validate_variant(cfg, switches, reference_cfg)
    return path, cfg, args_by_agent


def validate_configs(expert_config, config_paths):
    expert_path = _resolve_path(expert_config)
    _require(expert_path.is_file(), f"expert config is missing: {expert_path}")
    expert_cfg = load_config(expert_path)
    expert_args = validate_protocol_config(expert_cfg, expert_path)
    _require(
        expert_cfg.get("project", {}).get("stage") == "iclr2027_cosine_expert_training",
        "expert config is not the seed0 cosine-expert protocol",
    )
    expected_keys = {variant_id(*switches) for switches in ALL_SWITCHES}
    _require(set(config_paths) == expected_keys, f"loss matrix configs are incomplete: {sorted(config_paths)}")
    reference_cfg = load_config(DEFAULT_FULL_CONFIG)
    validated = {}
    for switches in ALL_SWITCHES:
        condition = variant_id(*switches)
        validated[condition] = _validate_config(
            config_paths[condition],
            switches,
            expert_args,
            reference_cfg,
        )
    return expert_path, expert_cfg, expert_args, validated


def _read_exact_results(path):
    _require(path.is_file(), f"social result CSV is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        _require(len(fieldnames) == len(set(fieldnames)), f"social result CSV has duplicate columns: {path}")
        missing = sorted(REQUIRED_RESULT_FIELDS - set(fieldnames))
        _require(not missing, f"social result CSV is missing columns {missing}: {path}")
        rows = list(reader)
    _require(len(rows) == 5, f"{path} has {len(rows)} receiver rows; expected exactly five")
    receiver_ids = [_integer(row.get("receiver_agent"), f"{path.name} receiver id") for row in rows]
    _require(sorted(receiver_ids) == EXPECTED_RECEIVERS, f"{path} receiver ids are incomplete: {receiver_ids}")
    _require(len(set(receiver_ids)) == 5, f"{path} contains duplicate receivers: {receiver_ids}")
    return sorted(rows, key=lambda row: int(row["receiver_agent"])), set(fieldnames)


def validate_social_run(run_dir, cfg, args_by_agent, experts, switches):
    fr, kd, sc = switches
    condition = variant_id(*switches)
    run_dir = Path(run_dir).resolve()
    result_path = run_dir / "metrics" / "social_results.csv"
    rows, fieldnames = _read_exact_results(result_path)
    if switches in MISSING_COMBINATIONS:
        missing_provenance = sorted(NEW_PROVENANCE_FIELDS - fieldnames)
        _require(
            not missing_provenance,
            f"{condition}: social result CSV is missing provenance columns {missing_provenance}",
        )
    receiver_cfg = cfg["social_learning"]["receiver"]
    logits_cfg = cfg["logits"]
    expected_steps = 60 * math.ceil(10_000 / 64)
    expected_variant = _expected_dkp_variant(switches)
    expected_method = {
        (False, False, False): "DKP_CE_ONLY",
        (True, True, True): "DKP_SL",
    }.get(switches, "DKP_SL_ABLATION")
    parsed = {}

    for row in rows:
        receiver_id = _integer(row["receiver_agent"], f"{condition} receiver id")
        context = f"{condition} receiver {receiver_id}"
        expected_args = args_by_agent[receiver_id]
        _require(row["run_name"] == cfg["project"]["run_name"], f"{context}: run_name mismatch")
        _require(row["receiver_model"] == expected_args.model_name, f"{context}: model mismatch")
        classes = [_integer(value, f"{context}: class id") for value in row["expert_classes"].split(",") if value]
        _require(classes == list(expected_args.active_class_ids), f"{context}: expert classes mismatch")
        _require(row["protocol"] == "dkp_sl_v1", f"{context}: protocol mismatch")
        _require(row["dkp_variant"] == expected_variant, f"{context}: variant mismatch")
        _require(row["method"] == expected_method, f"{context}: method mismatch")
        _require(row["packet_method"].lower() == "dsdm", f"{context}: packet method mismatch")
        _require(row["init_mode"] == "expert", f"{context}: init mode mismatch")
        _require(row["self_data_mode"] == "real", f"{context}: local stream is not real data")
        _require(_boolean(row["use_fr"], f"{context}: use_fr") is fr, f"{context}: FR switch mismatch")
        _require(_boolean(row["use_logits"], f"{context}: use_logits") is kd, f"{context}: KD/logit switch mismatch")
        _require(not _boolean(row["use_generalist_logits"], f"{context}: generalist logits"), f"{context}: used generalist logits")
        _same_float(_finite_float(row["lambda_fr"], f"{context}: lambda_FR"), 0.2 if fr else 0.0, f"{context}: lambda_FR")
        _same_float(_finite_float(row["lambda_kd"], f"{context}: lambda_KD"), 0.6 if kd else 0.0, f"{context}: lambda_KD")
        _same_float(_finite_float(row["lambda_sc"], f"{context}: lambda_SC"), 0.1 if sc else 0.0, f"{context}: lambda_SC")
        _same_float(_finite_float(row["kd_temperature"], f"{context}: KD temperature"), 2.0, f"{context}: KD temperature")
        _same_float(_finite_float(row["supcon_temperature"], f"{context}: SupCon temperature"), 0.07, f"{context}: SupCon temperature")
        _require(_integer(row["ipc"], f"{context}: IPC") == 10, f"{context}: IPC is not 10")
        _require(_integer(row["self_real_images"], f"{context}: local images") == 10_000, f"{context}: local image count mismatch")
        external_images = _integer(row["external_comm_images"], f"{context}: external images")
        _require(external_images == 800, f"{context}: raw external images are {external_images}, expected 800")
        logit_bytes = _integer(row["external_comm_logit_bytes"], f"{context}: logit bytes")
        expected_logit_bytes = 128_000 if kd else 0
        _require(logit_bytes == expected_logit_bytes, f"{context}: logit bytes are {logit_bytes}, expected {expected_logit_bytes}")
        _require(
            _integer(row["external_comm_generalist_logit_bytes"], f"{context}: generalist bytes") == 0,
            f"{context}: communicated generalist logits",
        )
        prototypes = _integer(row["prototype_initialized_classes"], f"{context}: prototypes")
        _require(prototypes == 80, f"{context}: initialized {prototypes} prototypes, expected 80")
        steps = _integer(row["optimizer_steps"], f"{context}: optimizer steps")
        _require(steps == expected_steps == 9_420, f"{context}: optimizer steps are {steps}, expected 9420")
        training_seconds = _finite_float(row["training_seconds"], f"{context}: training seconds")
        _require(training_seconds > 0.0, f"{context}: training time is not positive")
        _require(_integer(row["receiver_seed"], f"{context}: receiver seed") == 100_000 + receiver_id, f"{context}: seed mismatch")
        _require(str(row["time"]).strip(), f"{context}: completion timestamp is empty")

        metrics = {}
        for output_name, field in (("global", "acc_global_after"), ("new", "acc_new_after"), ("expert", "acc_expert_after")):
            value = _finite_float(row[field], f"{context}: {output_name} accuracy")
            _require(0.0 <= value <= 100.0, f"{context}: {output_name} accuracy is outside [0,100]")
            metrics[output_name] = value
        before_expert = _finite_float(row["acc_expert_before"], f"{context}: expert-before accuracy")
        _require(
            abs(before_expert - experts[receiver_id]["official_test_accuracy_report_only"]) <= 0.1,
            f"{context}: expert-before accuracy differs from its manifest by more than 0.1 points",
        )
        _finite_float(row["acc_global_before"], f"{context}: global-before accuracy")
        losses = {field: _finite_float(row[field], f"{context}: {field}") for field in LOSS_FIELDS}
        for enabled, field in ((fr, "loss_fr"), (kd, "loss_kd"), (sc, "loss_sc")):
            if not enabled:
                _same_float(losses[field], 0.0, f"{context}: inactive {field}")
        expected_total = losses["loss_cls"] + (0.2 if fr else 0.0) * losses["loss_fr"]
        expected_total += (0.6 if kd else 0.0) * losses["loss_kd"]
        expected_total += (0.1 if sc else 0.0) * losses["loss_sc"]
        _same_float(losses["loss"], expected_total, f"{context}: total-loss identity", tolerance=1e-6)

        expected_sha = experts[receiver_id]["checkpoint_sha256"]
        _require(row["receiver_init_checkpoint_sha256"] == expected_sha, f"{context}: init checkpoint SHA mismatch")
        _require(row["fr_teacher_checkpoint_sha256"] == expected_sha, f"{context}: FR teacher SHA mismatch")
        if switches in MISSING_COMBINATIONS:
            _require(row["classifier_type"] == "cosine", f"{context}: classifier is not cosine")
            _require(
                row["prototype_init_mode"] == "cosine_unit_weight_rows",
                f"{context}: prototype mode mismatch",
            )
            _require(row["prototype_alpha"] == "", f"{context}: cosine prototype alpha must be empty")
            _require(row["prototype_beta"] == "", f"{context}: cosine prototype beta must be empty")
            norm_min = _finite_float(row["prototype_weight_norm_min"], f"{context}: prototype norm min")
            norm_max = _finite_float(row["prototype_weight_norm_max"], f"{context}: prototype norm max")
            _same_float(norm_min, 1.0, f"{context}: prototype norm min", tolerance=1e-6)
            _same_float(norm_max, 1.0, f"{context}: prototype norm max", tolerance=1e-6)
            _require(row["checkpoint_retention"] == "final_only", f"{context}: checkpoint retention mismatch")

            checkpoint_dir = (
                run_dir
                / "social_learning"
                / f"receiver_agent_{receiver_id}"
                / "checkpoints"
                / f"dkp_sl_v1_{expected_variant}"
            )
            provenance_path = _resolve_path(row["receiver_provenance_path"])
            expected_provenance_path = (checkpoint_dir / "receiver_provenance.json").resolve()
            _require(provenance_path == expected_provenance_path, f"{context}: provenance path mismatch")
            _require(provenance_path.is_file(), f"{context}: provenance file is missing")
            _require(
                sha256_file(provenance_path) == row["receiver_provenance_sha256"],
                f"{context}: provenance SHA mismatch",
            )
            try:
                provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise LossAblationSummaryError(f"{context}: invalid receiver provenance") from exc
            _require(provenance.get("dkp_variant") == expected_variant, f"{context}: provenance variant mismatch")
            _require(
                provenance.get("loss_switches") == {"fr": fr, "kd": kd, "supcon": sc},
                f"{context}: provenance loss switches mismatch",
            )
            _require(provenance.get("checkpoint_retention") == "final_only", f"{context}: provenance retention mismatch")
            _require(provenance.get("receiver_init_checkpoint_sha256") == expected_sha, f"{context}: provenance init SHA mismatch")
            _require(provenance.get("fr_teacher_checkpoint_sha256") == expected_sha, f"{context}: provenance FR SHA mismatch")
            prototype = provenance.get("prototype_initialization", {})
            _require(prototype.get("classifier_type") == "cosine", f"{context}: provenance classifier mismatch")
            _require(prototype.get("mode") == "cosine_unit_weight_rows", f"{context}: provenance prototype mode mismatch")
            expected_external = sorted(
                set(range(100)) - set(int(value) for value in expected_args.active_class_ids)
            )
            _require(
                sorted(int(value) for value in prototype.get("initialized_classes", [])) == expected_external,
                f"{context}: provenance initialized class set mismatch",
            )
            artifacts = provenance.get("checkpoint_artifacts", {})
            _require(set(artifacts) == {"after_social"}, f"{context}: final_only retained extra checkpoints")
            after_path = _resolve_path(artifacts["after_social"].get("path", ""))
            _require(after_path == (checkpoint_dir / "after_social.pt").resolve(), f"{context}: final checkpoint path mismatch")
            _require(after_path.is_file(), f"{context}: final checkpoint is missing")
            after_sha = sha256_file(after_path)
            _require(after_sha == artifacts["after_social"].get("sha256"), f"{context}: provenance final SHA mismatch")
            _require(after_sha == row["after_social_checkpoint_sha256"], f"{context}: result final SHA mismatch")
            retained_files = {path.name for path in checkpoint_dir.iterdir() if path.is_file()}
            _require(
                retained_files == {"after_social.pt", "receiver_provenance.json"},
                f"{context}: final_only checkpoint directory contains {sorted(retained_files)}",
            )

        parsed[receiver_id] = {
            "metrics": metrics,
            "losses": losses,
            "training_seconds": training_seconds,
            "optimizer_steps": steps,
            "raw_external_images": external_images,
            "logit_bytes": logit_bytes,
            "prototype_initialized_classes": prototypes,
        }
    return {
        "condition": condition,
        "switches": {"fr": fr, "kd": kd, "supcon": sc},
        "run_name": cfg["project"]["run_name"],
        "run_dir": str(run_dir),
        "config_weights": {
            "lambda_fr": receiver_cfg["lambda_fr"],
            "lambda_kd": logits_cfg["lambda_kd"],
            "lambda_sc": receiver_cfg["lambda_sc"],
        },
        "social_results_csv": str(result_path.resolve()),
        "social_results_sha256": sha256_file(result_path),
        "receivers": parsed,
    }


def _mean(values):
    _require(len(values) == 5, "loss-ablation mean requires exactly five receivers")
    return statistics.fmean(values)


def _aggregate(validated):
    rows = [validated["receivers"][receiver_id] for receiver_id in EXPECTED_RECEIVERS]
    return {
        "metrics_mean": {
            metric: _mean([row["metrics"][metric] for row in rows])
            for metric in ("global", "new", "expert")
        },
        "loss_mean": {
            field: _mean([row["losses"][field] for row in rows])
            for field in LOSS_FIELDS
        },
        "resources": {
            "training_seconds_mean": _mean([row["training_seconds"] for row in rows]),
            "training_seconds_total": sum(row["training_seconds"] for row in rows),
            "optimizer_steps_mean": _mean([row["optimizer_steps"] for row in rows]),
            "optimizer_steps_total": sum(row["optimizer_steps"] for row in rows),
            "raw_external_images_mean": _mean([row["raw_external_images"] for row in rows]),
            "raw_external_images_total": sum(row["raw_external_images"] for row in rows),
            "logit_bytes_mean": _mean([row["logit_bytes"] for row in rows]),
            "logit_bytes_total": sum(row["logit_bytes"] for row in rows),
        },
    }


def build_summary(
    expert_config=DEFAULT_EXPERT_CONFIG,
    config_paths=None,
    expert_run_dir=None,
    run_dirs=None,
    min_local_test_accuracy=20.0,
):
    config_paths = default_config_paths() if config_paths is None else dict(config_paths)
    run_dirs = {} if run_dirs is None else dict(run_dirs)
    expert_path, expert_cfg, expert_args, configs = validate_configs(expert_config, config_paths)
    experts = load_complete_experts(
        resolve_run_dir(expert_cfg, override=expert_run_dir),
        expert_args,
        min_local_test_accuracy=min_local_test_accuracy,
    )
    validated = {}
    sources = {
        "expert_config": str(expert_path),
        "expert_config_sha256": sha256_file(expert_path),
    }
    for switches in ALL_SWITCHES:
        condition = variant_id(*switches)
        config_path, cfg, args_by_agent = configs[condition]
        run_dir = resolve_run_dir(cfg, override=run_dirs.get(condition))
        result = validate_social_run(run_dir, cfg, args_by_agent, experts, switches)
        validated[condition] = result
        sources[condition] = {
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "run_dir": str(run_dir),
            "social_results_csv": result["social_results_csv"],
            "social_results_sha256": result["social_results_sha256"],
        }

    per_receiver = []
    for receiver_id in EXPECTED_RECEIVERS:
        conditions = {}
        for switches in ALL_SWITCHES:
            condition = variant_id(*switches)
            row = validated[condition]["receivers"][receiver_id]
            conditions[condition] = {
                **row["metrics"],
                "loss_mean": row["losses"],
                "training_seconds": row["training_seconds"],
                "optimizer_steps": row["optimizer_steps"],
                "raw_external_images": row["raw_external_images"],
                "logit_bytes": row["logit_bytes"],
                "prototype_initialized_classes": row["prototype_initialized_classes"],
            }
        per_receiver.append(
            {
                "receiver_agent": receiver_id,
                "receiver_model": experts[receiver_id]["receiver_model"],
                "conditions": conditions,
            }
        )

    return {
        "status": "complete_diagnostic",
        "formal_result": False,
        "paper_eligible": False,
        "result_scope": "fixed seed0 DKP-SL loss ablation; not hyperparameter search",
        "dataset": "cifar100",
        "seed": 0,
        "agents": 5,
        "classes_per_agent": 20,
        "ipc": 10,
        "fixed_hyperparameters": FIXED_HYPERPARAMETERS,
        "condition_order": [variant_id(*switches) for switches in ALL_SWITCHES],
        "receiver_ids": EXPECTED_RECEIVERS,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "per_receiver": per_receiver,
        "mean_over_five_receivers": {
            condition: {
                "switches": validated[condition]["switches"],
                **_aggregate(validated[condition]),
            }
            for condition in [variant_id(*switches) for switches in ALL_SWITCHES]
        },
    }


def _validate_output_path(path):
    path = _resolve_path(path)
    _require(path.suffix.lower() == ".json", "diagnostic output must be a JSON file")
    forbidden = [
        ROOT / "RESULTS.md",
        ROOT / "paper_tables",
        ROOT / "experiments/registry.yaml",
        ROOT / "outputs/experiment_registry",
    ]
    _require(path != forbidden[0].resolve(), "summary must not write RESULTS.md")
    _require(forbidden[1].resolve() not in path.parents, "summary must not write paper_tables")
    _require(path != forbidden[2].resolve(), "summary must not write the registry")
    _require(forbidden[3].resolve() not in path.parents, "summary must not write generated registry outputs")
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-config", default=str(DEFAULT_EXPERT_CONFIG))
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--expert-run-dir")
    parser.add_argument("--min-local-test-accuracy", type=float, default=20.0)
    parser.add_argument("--output-json", "--output", dest="output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args(argv)


def main(argv=None):
    cli = parse_args(argv)
    try:
        output_path = _validate_output_path(cli.output)
        _require(not output_path.exists(), f"diagnostic output already exists: {output_path}")
        summary = build_summary(
            expert_config=cli.expert_config,
            config_paths=default_config_paths(cli.config_dir),
            expert_run_dir=cli.expert_run_dir,
            min_local_test_accuracy=cli.min_local_test_accuracy,
        )
        atomic_write_json(summary, output_path)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ICLR2027 DKP loss-ablation summary FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "passed", "output": str(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
