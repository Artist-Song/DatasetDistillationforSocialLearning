#!/usr/bin/env python3
"""Build a diagnostic-only summary for the complete ICLR 2027 DKP first round."""

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
from scripts.validate_iclr2027_cosine_experts import (  # noqa: E402
    DEFAULT_CONFIG as DEFAULT_EXPERT_CONFIG,
    resolve_run_dir,
    sha256_file,
    validate_protocol_config,
)


DEFAULT_CE_CONFIG = ROOT / "configs/iclr2027/cifar100_5agent20cls_dkp_ce_only_ipc10_seed0_v2.yaml"
DEFAULT_FULL_CONFIG = ROOT / "configs/iclr2027/cifar100_5agent20cls_dkp_sl_full_ipc10_seed0_v2.yaml"
DEFAULT_OUTPUT = ROOT / "outputs/diagnostics/iclr2027_cifar100_5agent20cls_ipc10_seed0_first_round.json"
EXPECTED_RECEIVERS = list(range(5))
LOSS_FIELDS = [
    "loss",
    "loss_cls",
    "loss_ce_local",
    "loss_ce_external",
    "loss_fr",
    "loss_kd",
    "loss_sc",
]
REQUIRED_RESULT_FIELDS = {
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
    "use_fr",
    "lambda_fr",
    "lambda_sc",
    "supcon_temperature",
    "use_logits",
    "communication_mode",
    "use_generalist_logits",
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
    "optimizer_steps",
    "training_seconds",
    "prototype_initialized_classes",
    "receiver_seed",
    "receiver_init_checkpoint_sha256",
    "fr_teacher_checkpoint_sha256",
    "time",
    *LOSS_FIELDS,
}


class SummaryError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise SummaryError(message)


def _resolve_path(path):
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved.resolve()


def _finite_float(value, context):
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise SummaryError(f"{context} is not numeric: {value!r}") from exc
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


def _same_float(actual, expected, context):
    _require(
        math.isclose(actual, float(expected), rel_tol=0.0, abs_tol=1e-12),
        f"{context} mismatch: {actual} != {float(expected)}",
    )


def _config_and_args(config_path, expected_variant=None):
    config_path = _resolve_path(config_path)
    _require(config_path.is_file(), f"config is missing: {config_path}")
    cfg = load_config(config_path)
    args_by_agent = validate_protocol_config(cfg, config_path)
    if expected_variant is not None:
        receiver = cfg.get("social_learning", {}).get("receiver", {})
        communication = cfg.get("communication", {})
        _require(receiver.get("protocol") == "dkp_sl_v1", f"{config_path.name} receiver protocol is not dkp_sl_v1")
        _require(receiver.get("dkp_variant") == expected_variant, f"{config_path.name} has the wrong DKP variant")
        _require(communication.get("strict_packet_validation") is True, f"{config_path.name} disabled strict packets")
        expected_logits = expected_variant == "full"
        _require(
            communication.get("use_sender_logits") is expected_logits,
            f"{config_path.name} sender-logit setting does not match {expected_variant}",
        )
    return config_path, cfg, args_by_agent


def _validate_configs(expert_config, ce_config, full_config):
    expert_path, expert_cfg, expert_args = _config_and_args(expert_config)
    # The expert-only config intentionally has communication disabled and labels its
    # receiver block ce_only; only its class/model/expert definition is consumed here.
    _require(
        expert_cfg.get("project", {}).get("stage") == "iclr2027_cosine_expert_training",
        "expert config is not the ICLR 2027 cosine-expert run",
    )
    ce_path, ce_cfg, ce_args = _config_and_args(ce_config, "ce_only")
    full_path, full_cfg, full_args = _config_and_args(full_config, "full")

    for agent_id in EXPECTED_RECEIVERS:
        expected_classes = list(expert_args[agent_id].active_class_ids)
        expected_model = expert_args[agent_id].model_name
        for name, candidate in (("ce_only", ce_args), ("full", full_args)):
            _require(
                list(candidate[agent_id].active_class_ids) == expected_classes,
                f"{name} agent {agent_id} class split differs from expert run",
            )
            _require(
                candidate[agent_id].model_name == expected_model,
                f"{name} agent {agent_id} model differs from expert run",
            )
    return {
        "expert": (expert_path, expert_cfg, expert_args),
        "ce_only": (ce_path, ce_cfg, ce_args),
        "full": (full_path, full_cfg, full_args),
    }


def _manifest_agent_id(path):
    name = path.parent.parent.name
    _require(name.startswith("agent_"), f"unexpected expert manifest path: {path}")
    try:
        return int(name.removeprefix("agent_"))
    except ValueError as exc:
        raise SummaryError(f"unexpected expert agent directory: {name}") from exc


def load_complete_experts(run_dir, args_by_agent, min_local_test_accuracy=20.0):
    run_dir = Path(run_dir).resolve()
    manifest_paths = sorted(run_dir.glob("agents/agent_*/checkpoints/expert_manifest.json"))
    manifest_ids = [_manifest_agent_id(path) for path in manifest_paths]
    _require(len(manifest_paths) == 5, f"expert run has {len(manifest_paths)} manifests; expected exactly five")
    _require(sorted(manifest_ids) == EXPECTED_RECEIVERS, f"expert manifest ids are incomplete: {manifest_ids}")
    _require(len(set(manifest_ids)) == 5, f"expert manifest ids are duplicated: {manifest_ids}")

    threshold = _finite_float(min_local_test_accuracy, "minimum local expert accuracy")
    experts = {}
    for path in manifest_paths:
        agent_id = _manifest_agent_id(path)
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SummaryError(f"cannot read expert manifest: {path}") from exc
        _require(isinstance(manifest, dict), f"expert manifest is not an object: {path}")
        _require(_integer(manifest.get("agent_id"), f"agent {agent_id} manifest id") == agent_id, f"agent {agent_id} manifest id mismatch")
        _require(
            manifest.get("role") == "fully_converged_agent_expert_and_logit_teacher",
            f"agent {agent_id} manifest is not the final expert",
        )
        _require(manifest.get("test_used_for_selection") is False, f"agent {agent_id} used test data for selection")
        _require(manifest.get("masked_local_ce") is True, f"agent {agent_id} local CE mask is disabled")
        _require(manifest.get("labels") == "global", f"agent {agent_id} labels are not global")
        _require(_integer(manifest.get("global_output_dim"), f"agent {agent_id} output dim") == 100, f"agent {agent_id} output dim is not 100")
        active_class_ids = [_integer(value, f"agent {agent_id} class id") for value in manifest.get("active_class_ids", [])]
        _require(
            active_class_ids == list(args_by_agent[agent_id].active_class_ids),
            f"agent {agent_id} manifest class set differs from config",
        )
        classifier = manifest.get("classifier", {})
        _require(classifier.get("type") == "cosine", f"agent {agent_id} expert is not cosine")
        local_accuracy = _finite_float(
            manifest.get("official_test_accuracy_report_only"),
            f"agent {agent_id} local expert accuracy",
        )
        _require(0.0 <= local_accuracy <= 100.0, f"agent {agent_id} local expert accuracy is outside [0,100]")
        _require(local_accuracy > threshold, f"agent {agent_id} local expert accuracy does not clear the diagnostic floor")

        checkpoint_path = run_dir / "agents" / f"agent_{agent_id}" / "checkpoints" / "expert_model.pt"
        _require(checkpoint_path.is_file(), f"agent {agent_id} expert checkpoint is missing")
        checkpoint_sha256 = sha256_file(checkpoint_path)
        _require(
            checkpoint_sha256 == str(manifest.get("expert_sha256", "")).lower(),
            f"agent {agent_id} expert checkpoint SHA-256 differs from manifest",
        )
        experts[agent_id] = {
            "receiver_agent": agent_id,
            "receiver_model": args_by_agent[agent_id].model_name,
            "active_class_ids": active_class_ids,
            "official_test_accuracy_report_only": local_accuracy,
            "selected_epoch": _integer(manifest.get("selected_epoch"), f"agent {agent_id} selected epoch"),
            "checkpoint_sha256": checkpoint_sha256,
            "manifest": str(path.resolve()),
            "manifest_sha256": sha256_file(path),
        }
    return experts


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
    return sorted(rows, key=lambda row: int(row["receiver_agent"]))


def validate_social_run(run_dir, cfg, args_by_agent, experts, variant):
    run_dir = Path(run_dir).resolve()
    result_path = run_dir / "metrics" / "social_results.csv"
    rows = _read_exact_results(result_path)
    expected_run_name = str(cfg["project"]["run_name"])
    receiver_cfg = cfg["social_learning"]["receiver"]
    communication_cfg = cfg["communication"]
    logits_cfg = cfg["logits"]
    expected_method = "DKP_CE_ONLY" if variant == "ce_only" else "DKP_SL"
    expected_logits = variant == "full"
    expected_steps = int(receiver_cfg["epochs"]) * math.ceil(
        10_000 / int(receiver_cfg["local_batch_size"])
    )
    expected_logit_bytes = 128_000 if expected_logits else 0
    parsed = {}

    for row in rows:
        receiver_id = _integer(row["receiver_agent"], "receiver id")
        context = f"{variant} receiver {receiver_id}"
        expected_args = args_by_agent[receiver_id]
        _require(row["run_name"] == expected_run_name, f"{context} run_name mismatch")
        _require(row["receiver_model"] == expected_args.model_name, f"{context} model mismatch")
        row_classes = [_integer(value, f"{context} class id") for value in row["expert_classes"].split(",") if value != ""]
        _require(row_classes == list(expected_args.active_class_ids), f"{context} expert classes mismatch")
        _require(row["protocol"] == "dkp_sl_v1", f"{context} protocol mismatch")
        _require(row["dkp_variant"] == variant, f"{context} variant mismatch")
        _require(row["method"] == expected_method, f"{context} method mismatch")
        _require(row["packet_method"].lower() == "dsdm", f"{context} packet method mismatch")
        _require(row["init_mode"] == "expert", f"{context} did not initialize from its expert")
        _require(row["self_data_mode"] == "real", f"{context} did not use local real data")
        _require(row["communication_mode"] == communication_cfg["mode"], f"{context} communication mode mismatch")
        _require(_boolean(row["use_logits"], f"{context} use_logits") is expected_logits, f"{context} logits flag mismatch")
        _require(_boolean(row["use_fr"], f"{context} use_fr") is expected_logits, f"{context} FR flag mismatch")
        _require(not _boolean(row["use_generalist_logits"], f"{context} generalist logits"), f"{context} used generalist logits")
        _require(_integer(row["ipc"], f"{context} IPC") == 10, f"{context} IPC is not 10")
        _require(_integer(row["self_real_images"], f"{context} local images") == 10_000, f"{context} did not use all local expert data")
        external_images = _integer(row["external_comm_images"], f"{context} external images")
        _require(external_images == 800, f"{context} raw external image count is {external_images}, expected 800")
        logit_bytes = _integer(row["external_comm_logit_bytes"], f"{context} logit bytes")
        _require(
            logit_bytes == expected_logit_bytes,
            f"{context} logit bytes are {logit_bytes}, expected {expected_logit_bytes}",
        )
        _require(
            _integer(row["external_comm_generalist_logit_bytes"], f"{context} generalist bytes") == 0,
            f"{context} communicated generalist logits",
        )
        prototype_classes = _integer(row["prototype_initialized_classes"], f"{context} prototype classes")
        _require(prototype_classes == 80, f"{context} initialized {prototype_classes} external classes, expected 80")
        optimizer_steps = _integer(row["optimizer_steps"], f"{context} optimizer steps")
        _require(
            optimizer_steps == expected_steps,
            f"{context} optimizer steps are {optimizer_steps}, expected {expected_steps}",
        )
        training_seconds = _finite_float(row["training_seconds"], f"{context} training time")
        _require(training_seconds > 0.0, f"{context} training time is not positive")
        _require(_integer(row["receiver_seed"], f"{context} seed") == 100_000 + receiver_id, f"{context} seed mismatch")
        _require(str(row["time"]).strip(), f"{context} has no completion timestamp")

        _same_float(_finite_float(row["lambda_fr"], f"{context} lambda_fr"), receiver_cfg["lambda_fr"], f"{context} lambda_fr")
        _same_float(_finite_float(row["lambda_sc"], f"{context} lambda_sc"), receiver_cfg["lambda_sc"], f"{context} lambda_sc")
        _same_float(_finite_float(row["lambda_kd"], f"{context} lambda_kd"), logits_cfg["lambda_kd"], f"{context} lambda_kd")
        _same_float(_finite_float(row["kd_temperature"], f"{context} KD temperature"), logits_cfg["temperature"], f"{context} KD temperature")
        _same_float(
            _finite_float(row["supcon_temperature"], f"{context} SupCon temperature"),
            receiver_cfg["supcon_temperature"],
            f"{context} SupCon temperature",
        )

        metrics = {}
        for output_name, field in (
            ("global", "acc_global_after"),
            ("new", "acc_new_after"),
            ("expert", "acc_expert_after"),
        ):
            value = _finite_float(row[field], f"{context} {output_name} accuracy")
            _require(0.0 <= value <= 100.0, f"{context} {output_name} accuracy is outside [0,100]")
            metrics[output_name] = value
        before_expert = _finite_float(row["acc_expert_before"], f"{context} expert-before accuracy")
        _require(
            abs(before_expert - experts[receiver_id]["official_test_accuracy_report_only"]) <= 0.1,
            f"{context} expert-before accuracy differs from its expert manifest by more than 0.1 points",
        )
        _finite_float(row["acc_global_before"], f"{context} global-before accuracy")
        losses = {field: _finite_float(row[field], f"{context} {field}") for field in LOSS_FIELDS}
        if variant == "ce_only":
            for field in ("loss_fr", "loss_kd", "loss_sc"):
                _same_float(losses[field], 0.0, f"{context} {field}")

        expected_sha = experts[receiver_id]["checkpoint_sha256"]
        _require(row["receiver_init_checkpoint_sha256"] == expected_sha, f"{context} init checkpoint SHA mismatch")
        _require(row["fr_teacher_checkpoint_sha256"] == expected_sha, f"{context} FR teacher SHA mismatch")
        parsed[receiver_id] = {
            "metrics": metrics,
            "losses": losses,
            "training_seconds": training_seconds,
            "optimizer_steps": optimizer_steps,
            "raw_external_images": external_images,
            "logit_bytes": logit_bytes,
            "prototype_initialized_classes": prototype_classes,
        }

    return {
        "variant": variant,
        "run_name": expected_run_name,
        "run_dir": str(run_dir),
        "social_results_csv": str(result_path.resolve()),
        "social_results_sha256": sha256_file(result_path),
        "receivers": parsed,
    }


def _mean(values):
    _require(len(values) == 5, "diagnostic mean requires exactly five receivers")
    return statistics.fmean(values)


def _aggregate_social(validated):
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


def _format_receiver_social(row):
    return {
        "global": row["metrics"]["global"],
        "new": row["metrics"]["new"],
        "expert": row["metrics"]["expert"],
        "loss_mean": row["losses"],
        "training_seconds": row["training_seconds"],
        "optimizer_steps": row["optimizer_steps"],
        "raw_external_images": row["raw_external_images"],
        "logit_bytes": row["logit_bytes"],
        "prototype_initialized_classes": row["prototype_initialized_classes"],
    }


def build_summary(
    expert_config=DEFAULT_EXPERT_CONFIG,
    ce_config=DEFAULT_CE_CONFIG,
    full_config=DEFAULT_FULL_CONFIG,
    expert_run_dir=None,
    ce_run_dir=None,
    full_run_dir=None,
    min_local_test_accuracy=20.0,
):
    configs = _validate_configs(expert_config, ce_config, full_config)
    expert_path, expert_cfg, expert_args = configs["expert"]
    ce_path, ce_cfg, ce_args = configs["ce_only"]
    full_path, full_cfg, full_args = configs["full"]
    resolved_expert_run = resolve_run_dir(expert_cfg, override=expert_run_dir)
    resolved_ce_run = resolve_run_dir(ce_cfg, override=ce_run_dir)
    resolved_full_run = resolve_run_dir(full_cfg, override=full_run_dir)
    experts = load_complete_experts(
        resolved_expert_run,
        expert_args,
        min_local_test_accuracy=min_local_test_accuracy,
    )
    ce = validate_social_run(resolved_ce_run, ce_cfg, ce_args, experts, "ce_only")
    full = validate_social_run(resolved_full_run, full_cfg, full_args, experts, "full")

    per_receiver = []
    for receiver_id in EXPECTED_RECEIVERS:
        per_receiver.append(
            {
                "receiver_agent": receiver_id,
                "receiver_model": experts[receiver_id]["receiver_model"],
                "local_expert": {
                    "expert": experts[receiver_id]["official_test_accuracy_report_only"],
                    "official_test_accuracy_report_only": experts[receiver_id][
                        "official_test_accuracy_report_only"
                    ],
                    "selected_epoch": experts[receiver_id]["selected_epoch"],
                    "checkpoint_sha256": experts[receiver_id]["checkpoint_sha256"],
                },
                "ce_only": _format_receiver_social(ce["receivers"][receiver_id]),
                "full_dkp_sl": _format_receiver_social(full["receivers"][receiver_id]),
            }
        )

    return {
        "status": "complete_diagnostic",
        "formal_result": False,
        "paper_eligible": False,
        "result_scope": "ICLR 2027 first-round protocol validation only",
        "dataset": "cifar100",
        "seed": 0,
        "agents": 5,
        "classes_per_agent": 20,
        "ipc": 10,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "expert_config": str(expert_path),
            "expert_config_sha256": sha256_file(expert_path),
            "expert_run_dir": str(resolved_expert_run),
            "ce_only_config": str(ce_path),
            "ce_only_config_sha256": sha256_file(ce_path),
            "ce_only_social_results": ce["social_results_csv"],
            "ce_only_social_results_sha256": ce["social_results_sha256"],
            "full_config": str(full_path),
            "full_config_sha256": sha256_file(full_path),
            "full_social_results": full["social_results_csv"],
            "full_social_results_sha256": full["social_results_sha256"],
        },
        "receiver_ids": EXPECTED_RECEIVERS,
        "per_receiver": per_receiver,
        "mean_over_five_receivers": {
            "local_expert": {
                "expert": _mean(
                    [experts[receiver_id]["official_test_accuracy_report_only"] for receiver_id in EXPECTED_RECEIVERS]
                ),
                "official_test_accuracy_report_only_mean": _mean(
                    [experts[receiver_id]["official_test_accuracy_report_only"] for receiver_id in EXPECTED_RECEIVERS]
                ),
            },
            "ce_only": _aggregate_social(ce),
            "full_dkp_sl": _aggregate_social(full),
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
    parser.add_argument("--ce-config", default=str(DEFAULT_CE_CONFIG))
    parser.add_argument("--full-config", default=str(DEFAULT_FULL_CONFIG))
    parser.add_argument("--expert-run-dir")
    parser.add_argument("--ce-run-dir")
    parser.add_argument("--full-run-dir")
    parser.add_argument("--min-local-test-accuracy", type=float, default=20.0)
    parser.add_argument("--output-json", "--output", dest="output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing diagnostic JSON")
    return parser.parse_args(argv)


def main(argv=None):
    cli = parse_args(argv)
    try:
        output_path = _validate_output_path(cli.output)
        _require(cli.overwrite or not output_path.exists(), f"diagnostic output already exists: {output_path}")
        summary = build_summary(
            expert_config=cli.expert_config,
            ce_config=cli.ce_config,
            full_config=cli.full_config,
            expert_run_dir=cli.expert_run_dir,
            ce_run_dir=cli.ce_run_dir,
            full_run_dir=cli.full_run_dir,
            min_local_test_accuracy=cli.min_local_test_accuracy,
        )
        atomic_write_json(summary, output_path)
        print(json.dumps({"status": "passed", "output": str(output_path)}, indent=2))
        return 0
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ICLR2027 first-round summary FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
