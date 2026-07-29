#!/usr/bin/env python3
"""Build a strict diagnostic-only summary for the matched-linear DKP run."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_data import build_agent_args  # noqa: E402
from config_adapter import load_config  # noqa: E402
from output_manager import atomic_write_json  # noqa: E402
from scripts.audit_iclr2027_linear_final_receivers import (  # noqa: E402
    _default_model_builder,
    validate_linear_receiver_checkpoint,
)
from scripts.summarize_iclr2027_dkp_first_round import (  # noqa: E402
    EXPECTED_RECEIVERS,
    SummaryError,
    _aggregate_social,
    _finite_float,
    _format_receiver_social,
    _integer,
    _mean,
    _read_exact_results,
    _require,
    _resolve_path,
    _validate_output_path,
    validate_social_run,
)
from scripts.validate_iclr2027_cosine_experts import resolve_run_dir, sha256_file  # noqa: E402
from scripts.validate_iclr2027_linear_communication import (  # noqa: E402
    CE_CONFIG,
    EXPERT_CONFIG,
    FULL_CONFIG,
    validate_config_contract,
)


DEFAULT_EXPERT_CONFIG = ROOT / EXPERT_CONFIG
DEFAULT_CE_CONFIG = ROOT / CE_CONFIG
DEFAULT_FULL_CONFIG = ROOT / FULL_CONFIG
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/cifar100_5agent20cls_dkp_linear_full_ipc10_seed0_v1"
    / "metrics/linear_head_seed0_summary.json"
)
LINEAR_RESULT_FIELDS = {
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


def _config_and_args(path):
    path = _resolve_path(path)
    _require(path.is_file(), f"linear diagnostic config is missing: {path}")
    config = load_config(path)
    args = {agent_id: build_agent_args(config, path, agent_id) for agent_id in EXPECTED_RECEIVERS}
    return path, config, args


def _validate_configs(expert_config, ce_config, full_config):
    expert = _config_and_args(expert_config)
    ce_only = _config_and_args(ce_config)
    full = _config_and_args(full_config)
    validate_config_contract(expert[1], ce_only[1], full[1])
    _require(
        expert[1]["project"].get("stage") == "iclr2027_linear_expert_training",
        "expert config is not the matched-linear training run",
    )
    for agent_id in EXPECTED_RECEIVERS:
        expected_classes = list(expert[2][agent_id].active_class_ids)
        expected_model = expert[2][agent_id].model_name
        for role, candidate in (("ce_only", ce_only), ("full", full)):
            _require(candidate[2][agent_id].classifier_type == "linear", f"{role} head is not linear")
            _require(
                list(candidate[2][agent_id].active_class_ids) == expected_classes,
                f"{role} agent {agent_id} class split differs",
            )
            _require(candidate[2][agent_id].model_name == expected_model, f"{role} model assignment differs")
    return {"expert": expert, "ce_only": ce_only, "full": full}


def _manifest_agent_id(path):
    name = path.parent.parent.name
    _require(name.startswith("agent_"), f"unexpected expert manifest path: {path}")
    try:
        return int(name.removeprefix("agent_"))
    except ValueError as exc:
        raise SummaryError(f"unexpected expert directory: {name}") from exc


def load_complete_linear_experts(run_dir, args_by_agent, min_local_test_accuracy=20.0):
    """Load exactly five complete linear expert manifests and verify checkpoint SHAs."""
    run_dir = Path(run_dir).resolve()
    paths = sorted(run_dir.glob("agents/agent_*/checkpoints/expert_manifest.json"))
    ids = [_manifest_agent_id(path) for path in paths]
    _require(len(paths) == 5, f"linear expert run has {len(paths)} manifests; expected exactly five")
    _require(sorted(ids) == EXPECTED_RECEIVERS and len(set(ids)) == 5, "linear expert ids are incomplete")
    threshold = _finite_float(min_local_test_accuracy, "minimum linear expert accuracy")
    experts = {}
    for path in paths:
        agent_id = _manifest_agent_id(path)
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SummaryError(f"cannot read linear expert manifest: {path}") from exc
        _require(isinstance(manifest, dict), f"linear expert manifest is not an object: {path}")
        _require(_integer(manifest.get("agent_id"), "manifest agent id") == agent_id, "manifest id mismatch")
        _require(
            manifest.get("role") == "fully_converged_agent_expert_and_logit_teacher",
            f"agent {agent_id} manifest role mismatch",
        )
        _require(manifest.get("test_used_for_selection") is False, f"agent {agent_id} used test selection")
        _require(
            manifest.get("retrained_on_full_local_train") is True,
            f"agent {agent_id} was not retrained on all local training data",
        )
        _require(manifest.get("masked_local_ce") is True, f"agent {agent_id} masked CE is disabled")
        _require(manifest.get("labels") == "global", f"agent {agent_id} labels are not global")
        _require(_integer(manifest.get("global_output_dim"), "global output dim") == 100, "output dim drifted")
        _require(
            [int(value) for value in manifest.get("active_class_ids", [])]
            == list(args_by_agent[agent_id].active_class_ids),
            f"agent {agent_id} active classes differ",
        )
        classifier = manifest.get("classifier", {})
        _require(classifier.get("type") == "linear", f"agent {agent_id} expert is not linear")
        _require(classifier.get("bias") is True, f"agent {agent_id} linear bias is missing")
        _require(classifier.get("feature_normalization") is False, "linear feature normalization drifted")
        _require(classifier.get("weight_normalization") is False, "linear weight normalization drifted")
        local_accuracy = _finite_float(
            manifest.get("official_test_accuracy_report_only"),
            f"agent {agent_id} local expert accuracy",
        )
        _require(local_accuracy > threshold, f"agent {agent_id} local accuracy is below the diagnostic floor")
        checkpoint = run_dir / f"agents/agent_{agent_id}/checkpoints/expert_model.pt"
        _require(checkpoint.is_file(), f"agent {agent_id} linear checkpoint is missing")
        recorded_checkpoint = _resolve_path(manifest.get("expert_path", ""))
        _require(recorded_checkpoint == checkpoint.resolve(), f"agent {agent_id} expert path differs")
        checkpoint_sha = sha256_file(checkpoint)
        _require(checkpoint_sha == str(manifest.get("expert_sha256", "")).lower(), "expert SHA mismatch")
        experts[agent_id] = {
            "receiver_agent": agent_id,
            "receiver_model": args_by_agent[agent_id].model_name,
            "active_class_ids": list(args_by_agent[agent_id].active_class_ids),
            "official_test_accuracy_report_only": local_accuracy,
            "selected_epoch": _integer(manifest.get("selected_epoch"), "selected epoch"),
            "checkpoint_sha256": checkpoint_sha,
            "manifest": str(path.resolve()),
            "manifest_sha256": sha256_file(path),
        }
    return experts


def _load_json(path, context):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SummaryError(f"cannot read {context}: {path}") from exc
    _require(isinstance(payload, dict), f"{context} is not an object")
    return payload


def _validate_linear_receiver_artifacts(
    run_dir,
    variant,
    row,
    receiver_id,
    expert_sha,
    receiver_args,
    model_builder,
):
    context = f"linear {variant} receiver {receiver_id}"
    _require(row.get("classifier_type") == "linear", f"{context} classifier is not linear")
    _require(
        row.get("prototype_init_mode") == "linear_local_row_norm_bias_mean",
        f"{context} prototype mode drifted",
    )
    alpha = _finite_float(row.get("prototype_alpha"), f"{context} alpha")
    beta = _finite_float(row.get("prototype_beta"), f"{context} beta")
    norm_min = _finite_float(row.get("prototype_weight_norm_min"), f"{context} min norm")
    norm_max = _finite_float(row.get("prototype_weight_norm_max"), f"{context} max norm")
    _require(alpha > 0.0, f"{context} alpha is not positive")
    _require(math.isclose(norm_min, alpha, abs_tol=1e-6, rel_tol=0.0), f"{context} min norm != alpha")
    _require(math.isclose(norm_max, alpha, abs_tol=1e-6, rel_tol=0.0), f"{context} max norm != alpha")
    _require(row.get("checkpoint_retention") == "final_only", f"{context} retention is not final_only")

    expected_dir = (
        Path(run_dir).resolve()
        / f"social_learning/receiver_agent_{receiver_id}/checkpoints/dkp_sl_v1_{variant}"
    )
    provenance_path = Path(row.get("receiver_provenance_path", "")).resolve()
    expected_provenance = expected_dir / "receiver_provenance.json"
    _require(provenance_path == expected_provenance, f"{context} provenance path drifted")
    _require(provenance_path.is_file(), f"{context} provenance is missing")
    _require(sha256_file(provenance_path) == row.get("receiver_provenance_sha256"), "provenance SHA mismatch")
    provenance = _load_json(provenance_path, f"{context} provenance")
    _require(provenance.get("classifier_type") == "linear", f"{context} provenance classifier drifted")
    _require(provenance.get("checkpoint_retention") == "final_only", "provenance retention drifted")
    _require(provenance.get("receiver_init_checkpoint_sha256") == expert_sha, "init expert SHA drifted")
    _require(provenance.get("fr_teacher_checkpoint_sha256") == expert_sha, "FR teacher SHA drifted")
    prototype = provenance.get("prototype_initialization", {})
    _require(prototype.get("mode") == "linear_local_row_norm_bias_mean", "provenance prototype mode drifted")
    _require(math.isclose(_finite_float(prototype.get("alpha"), "provenance alpha"), alpha, abs_tol=1e-7), "provenance alpha mismatch")
    _require(math.isclose(_finite_float(prototype.get("beta"), "provenance beta"), beta, abs_tol=1e-7), "provenance beta mismatch")
    _require(len(prototype.get("initialized_classes", [])) == 80, "provenance does not list 80 external classes")

    artifacts = provenance.get("checkpoint_artifacts", {})
    _require(set(artifacts) == {"after_social"}, f"{context} retained redundant checkpoints")
    final_path = expected_dir / "after_social.pt"
    _require(final_path.is_file(), f"{context} final checkpoint is missing")
    final_sha = sha256_file(final_path)
    _require(final_sha == row.get("after_social_checkpoint_sha256"), "final checkpoint result SHA mismatch")
    recorded_final_path = Path(artifacts["after_social"].get("path", "")).resolve()
    _require(recorded_final_path == final_path.resolve(), "final checkpoint provenance path mismatch")
    _require(artifacts["after_social"].get("sha256") == final_sha, "final checkpoint provenance SHA mismatch")
    _require(not (expected_dir / "before_social.pt").exists(), f"{context} retained before_social")
    _require(not (expected_dir / "after_prototype_init.pt").exists(), f"{context} retained prototype checkpoint")
    try:
        checkpoint_audit = validate_linear_receiver_checkpoint(
            final_path,
            receiver_args,
            expected_sha256=final_sha,
            model_builder=model_builder,
            random_seed=30_000 + receiver_id,
        )
    except RuntimeError as exc:
        raise SummaryError(f"{context} checkpoint audit failed: {exc}") from exc
    _require(checkpoint_audit["sha256"] == final_sha, f"{context} strict-load SHA mismatch")
    _require(checkpoint_audit["output_shape"] == [2, 100], f"{context} strict-load output shape mismatch")
    return {
        "prototype_alpha": alpha,
        "prototype_beta": beta,
        "prototype_weight_norm_min": norm_min,
        "prototype_weight_norm_max": norm_max,
        "after_social_checkpoint_sha256": final_sha,
        "receiver_provenance": str(provenance_path),
        "receiver_provenance_sha256": row["receiver_provenance_sha256"],
        "checkpoint_audit": checkpoint_audit,
    }


def validate_linear_social_run(
    run_dir,
    config,
    args_by_agent,
    experts,
    variant,
    model_builder=_default_model_builder,
):
    validated = validate_social_run(run_dir, config, args_by_agent, experts, variant)
    result_path = Path(run_dir).resolve() / "metrics/social_results.csv"
    rows = _read_exact_results(result_path)
    missing = sorted(LINEAR_RESULT_FIELDS - set(rows[0]))
    _require(not missing, f"linear social results are missing columns: {missing}")
    for row in rows:
        receiver_id = int(row["receiver_agent"])
        stats = _validate_linear_receiver_artifacts(
            run_dir,
            variant,
            row,
            receiver_id,
            experts[receiver_id]["checkpoint_sha256"],
            args_by_agent[receiver_id],
            model_builder,
        )
        validated["receivers"][receiver_id]["linear_prototype"] = stats
    return validated


def _format_linear_social(row):
    payload = _format_receiver_social(row)
    payload["linear_prototype"] = row["linear_prototype"]
    return payload


def build_summary(
    expert_config=DEFAULT_EXPERT_CONFIG,
    ce_config=DEFAULT_CE_CONFIG,
    full_config=DEFAULT_FULL_CONFIG,
    expert_run_dir=None,
    ce_run_dir=None,
    full_run_dir=None,
    min_local_test_accuracy=20.0,
    model_builder=None,
):
    configs = _validate_configs(expert_config, ce_config, full_config)
    expert_path, expert_cfg, expert_args = configs["expert"]
    ce_path, ce_cfg, ce_args = configs["ce_only"]
    full_path, full_cfg, full_args = configs["full"]
    expert_run = resolve_run_dir(expert_cfg, override=expert_run_dir)
    ce_run = resolve_run_dir(ce_cfg, override=ce_run_dir)
    full_run = resolve_run_dir(full_cfg, override=full_run_dir)
    resolved_model_builder = model_builder or _default_model_builder
    experts = load_complete_linear_experts(expert_run, expert_args, min_local_test_accuracy)
    ce = validate_linear_social_run(
        ce_run,
        ce_cfg,
        ce_args,
        experts,
        "ce_only",
        model_builder=resolved_model_builder,
    )
    full = validate_linear_social_run(
        full_run,
        full_cfg,
        full_args,
        experts,
        "full",
        model_builder=resolved_model_builder,
    )

    per_receiver = []
    for receiver_id in EXPECTED_RECEIVERS:
        per_receiver.append(
            {
                "receiver_agent": receiver_id,
                "receiver_model": experts[receiver_id]["receiver_model"],
                "local_linear_expert": {
                    "expert": experts[receiver_id]["official_test_accuracy_report_only"],
                    "selected_epoch": experts[receiver_id]["selected_epoch"],
                    "checkpoint_sha256": experts[receiver_id]["checkpoint_sha256"],
                },
                "ce_only": _format_linear_social(ce["receivers"][receiver_id]),
                "full_dkp_sl": _format_linear_social(full["receivers"][receiver_id]),
            }
        )
    return {
        "status": "complete_diagnostic",
        "formal_result": False,
        "paper_eligible": False,
        "result_scope": "matched-linear classifier seed0 internal ablation only",
        "dataset": "cifar100",
        "seed": 0,
        "agents": 5,
        "classes_per_agent": 20,
        "ipc": 10,
        "classifier": "linear",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "receiver_ids": EXPECTED_RECEIVERS,
        "sources": {
            "expert_config": str(expert_path),
            "expert_config_sha256": sha256_file(expert_path),
            "expert_run_dir": str(expert_run),
            "ce_only_config": str(ce_path),
            "ce_only_config_sha256": sha256_file(ce_path),
            "ce_only_social_results": ce["social_results_csv"],
            "ce_only_social_results_sha256": ce["social_results_sha256"],
            "full_config": str(full_path),
            "full_config_sha256": sha256_file(full_path),
            "full_social_results": full["social_results_csv"],
            "full_social_results_sha256": full["social_results_sha256"],
        },
        "per_receiver": per_receiver,
        "mean_over_five_receivers": {
            "local_linear_expert": {
                "expert": _mean(
                    [experts[receiver_id]["official_test_accuracy_report_only"] for receiver_id in EXPECTED_RECEIVERS]
                )
            },
            "ce_only": _aggregate_social(ce),
            "full_dkp_sl": _aggregate_social(full),
        },
    }


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
    return parser.parse_args(argv)


def main(argv=None):
    cli = parse_args(argv)
    try:
        output = _validate_output_path(cli.output)
        _require(not output.exists(), f"diagnostic output already exists: {output}")
        summary = build_summary(
            expert_config=cli.expert_config,
            ce_config=cli.ce_config,
            full_config=cli.full_config,
            expert_run_dir=cli.expert_run_dir,
            ce_run_dir=cli.ce_run_dir,
            full_run_dir=cli.full_run_dir,
            min_local_test_accuracy=cli.min_local_test_accuracy,
        )
        atomic_write_json(summary, output)
        print(json.dumps({"status": "passed", "output": str(output)}, indent=2))
        return 0
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ICLR2027 linear-head summary FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
