#!/usr/bin/env python3
"""Fail-closed packet and prototype gate for the matched-linear DKP diagnostic."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_data import build_agent_args, get_agent_class_split, get_agent_model_split  # noqa: E402
from config_adapter import load_config  # noqa: E402
from output_manager import atomic_write_json  # noqa: E402
from packet_consumer import consume_external_manifest_packets, load_receiver_local_real_data  # noqa: E402
from packet_integrity import file_sha256  # noqa: E402
from scripts.prepare_iclr2027_linear_head_ablation import (  # noqa: E402
    CATALOG,
    CE_ONLY_RUN,
    EXPERT_RUN,
    FULL_RUN,
)
from scripts.validate_iclr2027_dkp_communication import (  # noqa: E402
    _load_strict_packets,
    _run_dir,
    _validate_reused_experts,
    _validate_variant_packets,
)
from social_trainer import SocialTrainer  # noqa: E402


EXPERT_CONFIG = "configs/iclr2027/cifar100_5agent20cls_dkp_linear_experts_seed0_v1.yaml"
CE_CONFIG = "configs/iclr2027/cifar100_5agent20cls_dkp_linear_ce_only_ipc10_seed0_v1.yaml"
FULL_CONFIG = "configs/iclr2027/cifar100_5agent20cls_dkp_linear_full_ipc10_seed0_v1.yaml"


class LinearCommunicationPreflightError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise LinearCommunicationPreflightError(message)


def _resolve(path):
    resolved = Path(path)
    return resolved if resolved.is_absolute() else ROOT / resolved


def validate_config_contract(expert, ce_only, full):
    """Require exact matched-linear configs and immutable packet-pool reuse."""
    expected_runs = {"expert": EXPERT_RUN, "ce_only": CE_ONLY_RUN, "full": FULL_RUN}
    configs = {"expert": expert, "ce_only": ce_only, "full": full}
    class_split = get_agent_class_split(expert)
    model_split = get_agent_model_split(expert)
    _require(sorted(class_split) == list(range(5)), "linear expert class split is incomplete")
    _require(sorted(model_split) == list(range(5)), "linear expert model split is incomplete")
    for role, config in configs.items():
        _require(config["project"].get("run_name") == expected_runs[role], f"unexpected {role} run_name")
        _require(config["project"].get("paper_eligible") is False, f"{role} is not diagnostic-only")
        _require(get_agent_class_split(config) == class_split, f"{role} class split drifted")
        _require(get_agent_model_split(config) == model_split, f"{role} model split drifted")
        _require(int(config["distillation"]["ipc"]) == 10, f"{role} IPC drifted")
        _require(config["communication"].get("pool_catalog") == CATALOG, f"{role} pool catalog drifted")
        for model_name in model_split.values():
            classifier = config["model_pool"]["models"][model_name].get("classifier")
            _require(classifier == {"type": "linear"}, f"{role}/{model_name} is not matched linear")
            _require(
                config["model_pool"]["models"][model_name]
                .get("expert_training", {})
                .get("masked_local_ce")
                is True,
                f"{role}/{model_name} masked CE is disabled",
            )

    expected_variants = {
        "ce_only": (ce_only, False, {"fr": False, "kd": False, "supcon": False}),
        "full": (full, True, {"fr": True, "kd": True, "supcon": True}),
    }
    for role, (config, logits, switches) in expected_variants.items():
        receiver = config["social_learning"]["receiver"]
        _require(receiver.get("dkp_variant") == role, f"{role} variant mismatch")
        _require(receiver.get("loss_switches") == switches, f"{role} loss switches drifted")
        _require(receiver.get("checkpoint_retention") == "final_only", f"{role} retention is not final_only")
        _require(config["communication"].get("use_sender_logits") is logits, f"{role} logit switch drifted")
        _require(config.get("expert_reuse", {}).get("source_run") == EXPERT_RUN, f"{role} expert source drifted")
    return class_split, model_split


def _validate_linear_receiver_prototypes(config, config_path, rows, source_experts, device):
    source_sha = {record["agent_id"]: record["checkpoint_sha256"] for record in source_experts}
    reports = []
    for receiver_id in range(5):
        args = build_agent_args(config, config_path, receiver_id)
        args.device = str(device)
        args.init_mode = "expert"
        args.use_logits = True
        args.use_generalist_logits = False
        args.packet_method = "dsdm"
        trainer = SocialTrainer(args, receiver_id, rows)
        model_old, model_new = trainer._build_models()
        from models.cosine_classifier import get_output_classifier, get_output_classifier_type

        old_classifier = get_output_classifier(model_old)
        new_classifier = get_output_classifier(model_new)
        _require(get_output_classifier_type(model_new) == "linear", f"receiver {receiver_id} head is not linear")
        local_ids = sorted(int(value) for value in trainer.expert_classes)
        local_index = torch.tensor(local_ids, device=new_classifier.weight.device, dtype=torch.long)
        local_weight_before = new_classifier.weight.detach().index_select(0, local_index).clone()
        local_bias_before = new_classifier.bias.detach().index_select(0, local_index).clone()
        expected_alpha = float(old_classifier.weight.detach().index_select(0, local_index).norm(dim=1).mean())
        expected_beta = float(old_classifier.bias.detach().index_select(0, local_index).mean())
        _require(math.isfinite(expected_alpha) and expected_alpha > 0.0, "linear alpha is invalid")
        _require(math.isfinite(expected_beta), "linear beta is invalid")

        expert_path = _run_dir(config) / "agents" / f"agent_{receiver_id}" / "checkpoints/expert_model.pt"
        _require(file_sha256(expert_path) == source_sha[receiver_id], f"receiver {receiver_id} expert SHA mismatch")
        local = load_receiver_local_real_data(args)
        external = consume_external_manifest_packets(args, rows, receiver_id, require_logits=True)
        trainer._validate_dkp_streams(local, external, require_logits=True)
        external_images = trainer._normalize_images(external["images"])
        external_ids = trainer._initialize_external_prototypes(
            model_old,
            model_new,
            external_images,
            external["labels"],
        )
        stats = trainer._prototype_init_stats
        _require(stats["classifier_type"] == "linear", "prototype classifier mode is not linear")
        _require(stats["mode"] == "linear_local_row_norm_bias_mean", "linear prototype mode drifted")
        _require(math.isclose(stats["alpha"], expected_alpha, rel_tol=0.0, abs_tol=1e-7), "alpha mismatch")
        _require(math.isclose(stats["beta"], expected_beta, rel_tol=0.0, abs_tol=1e-7), "beta mismatch")
        _require(len(external_ids) == 80, f"receiver {receiver_id} did not initialize 80 classes")
        _require(torch.equal(local_weight_before, new_classifier.weight.detach().index_select(0, local_index)), "local W changed")
        _require(torch.equal(local_bias_before, new_classifier.bias.detach().index_select(0, local_index)), "local b changed")
        external_index = torch.tensor(external_ids, device=new_classifier.weight.device, dtype=torch.long)
        external_weight = new_classifier.weight.detach().index_select(0, external_index)
        external_bias = new_classifier.bias.detach().index_select(0, external_index)
        norms = external_weight.norm(p=2, dim=1)
        _require(bool(torch.isfinite(external_weight).all()), "external linear weights are non-finite")
        _require(bool(torch.isfinite(external_bias).all()), "external linear biases are non-finite")
        _require(
            torch.allclose(norms, torch.full_like(norms, expected_alpha), atol=1e-6, rtol=0),
            "external linear row norms do not equal alpha",
        )
        _require(
            torch.allclose(external_bias, torch.full_like(external_bias, expected_beta), atol=1e-7, rtol=0),
            "external linear biases do not equal beta",
        )
        reports.append(
            {
                "receiver_agent": receiver_id,
                "receiver_model": get_agent_model_split(config)[receiver_id],
                "init_checkpoint_sha256": source_sha[receiver_id],
                "fr_teacher_checkpoint_sha256": source_sha[receiver_id],
                "prototype_classes": len(external_ids),
                "prototype_mode": stats["mode"],
                "prototype_alpha": expected_alpha,
                "prototype_beta": expected_beta,
                "prototype_norm_min": float(norms.min()),
                "prototype_norm_max": float(norms.max()),
                "external_decoded_images": int(external["images"].shape[0]),
                "local_real_images": int(local["num_images"]),
            }
        )
        del model_old, model_new, trainer, local, external, external_images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return reports


def run_preflight(expert_path=EXPERT_CONFIG, ce_path=CE_CONFIG, full_path=FULL_CONFIG, device="cuda"):
    expert_path, ce_path, full_path = map(_resolve, (expert_path, ce_path, full_path))
    expert = load_config(expert_path)
    ce_only = load_config(ce_path)
    full = load_config(full_path)
    class_split, model_split = validate_config_contract(expert, ce_only, full)
    ce_experts = _validate_reused_experts(expert, ce_only, "linear_ce_only")
    full_experts = _validate_reused_experts(expert, full, "linear_full")
    _require(ce_experts == full_experts, "linear CE/full expert SHA provenance differs")
    _, _, ce_packets = _load_strict_packets(ce_only, ce_path, require_logits=False)
    _, full_rows, full_packets = _load_strict_packets(full, full_path, require_logits=True)
    packet_report = _validate_variant_packets(full_experts, ce_packets, full_packets)
    _require(all(record["raw_images"] == 200 for record in packet_report), "sender raw image count is not 200")
    _require(all(record["logit_bytes"] == 32_000 for record in packet_report), "sender logit bytes are not 32000")
    receiver_report = _validate_linear_receiver_prototypes(
        full,
        full_path,
        full_rows,
        full_experts,
        device,
    )
    return {
        "status": "passed",
        "protocol": full["communication"]["protocol"],
        "diagnostic": "matched_linear_classifier",
        "formal_result": False,
        "paper_eligible": False,
        "class_partition": {str(key): value for key, value in class_split.items()},
        "model_assignment": {str(key): value for key, value in model_split.items()},
        "expert_provenance": full_experts,
        "packets": packet_report,
        "receivers": receiver_report,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-config", default=EXPERT_CONFIG)
    parser.add_argument("--ce-config", default=CE_CONFIG)
    parser.add_argument("--full-config", default=FULL_CONFIG)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--report-json", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    cli = parse_args(argv)
    try:
        report_path = _resolve(cli.report_json)
        if report_path.exists():
            raise FileExistsError(f"refusing to overwrite linear communication report: {report_path}")
        report = run_preflight(cli.expert_config, cli.ce_config, cli.full_config, device=cli.device)
        atomic_write_json(report, report_path)
    except Exception as exc:
        print(f"[linear-dkp-communication-preflight] FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
