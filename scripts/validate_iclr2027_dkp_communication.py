#!/usr/bin/env python3
"""Fail-closed communication preflight for the ICLR 2027 seed0 DKP round."""

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
from config_adapter import build_dsdm_args_from_config, load_config  # noqa: E402
from output_manager import atomic_write_json  # noqa: E402
from packet_consumer import consume_external_manifest_packets, load_receiver_local_real_data  # noqa: E402
from packet_integrity import file_sha256  # noqa: E402
from social_output_manager import read_packet_manifest  # noqa: E402
from social_trainer import SocialTrainer  # noqa: E402


EXPERT_CONFIG = "configs/iclr2027/cifar100_5agent20cls_dkp_cosine_experts_seed0_v1.yaml"
CE_CONFIG = "configs/iclr2027/cifar100_5agent20cls_dkp_ce_only_ipc10_seed0_v2.yaml"
FULL_CONFIG = "configs/iclr2027/cifar100_5agent20cls_dkp_sl_full_ipc10_seed0_v2.yaml"


class CommunicationPreflightError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise CommunicationPreflightError(message)


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _run_dir(config):
    project = config["project"]
    return _resolve(project.get("output_root", "outputs")) / project["run_name"]


def validate_config_contract(expert, ce_only, full):
    """Require the three variants to differ only where the protocol permits."""
    expected_runs = {
        "expert": "cifar100_5agent20cls_dkp_cosine_experts_seed0_v1",
        "ce_only": "cifar100_5agent20cls_dkp_ce_only_ipc10_seed0_v2",
        "full": "cifar100_5agent20cls_dkp_sl_full_ipc10_seed0_v2",
    }
    configs = {"expert": expert, "ce_only": ce_only, "full": full}
    reference_classes = get_agent_class_split(expert)
    reference_models = get_agent_model_split(expert)
    for role, config in configs.items():
        _require(config["project"]["run_name"] == expected_runs[role], f"unexpected {role} run_name")
        _require(get_agent_class_split(config) == reference_classes, f"{role} class split drifted")
        _require(get_agent_model_split(config) == reference_models, f"{role} model split drifted")
        _require(config["project"].get("paper_eligible") is False, f"{role} must remain diagnostic")
        _require(int(config["distillation"]["ipc"]) == 10, f"{role} IPC is not 10")
        for model_name in reference_models.values():
            classifier = config["model_pool"]["models"][model_name].get("classifier", {})
            _require(classifier.get("type") == "cosine", f"{role}/{model_name} is not cosine")
            _require(math.isclose(float(classifier.get("scale_init", -1)), 10.0), f"{role}/{model_name} scale changed")
    _require(ce_only["social_learning"]["receiver"].get("dkp_variant") == "ce_only", "CE variant mismatch")
    _require(full["social_learning"]["receiver"].get("dkp_variant") == "full", "full variant mismatch")
    _require(ce_only["communication"].get("use_sender_logits") is False, "CE-only cannot transmit logits")
    _require(full["communication"].get("use_sender_logits") is True, "full DKP-SL requires logits")
    return reference_classes, reference_models


def _validate_reused_experts(source_config, target_config, role):
    source_dir = _run_dir(source_config)
    target_dir = _run_dir(target_config)
    reuse_path = target_dir / "provenance" / "expert_reuse_manifest.json"
    _require(reuse_path.is_file(), f"{role} expert reuse manifest is missing")
    reuse = json.loads(reuse_path.read_text(encoding="utf-8"))
    _require(reuse.get("source_run") == source_config["project"]["run_name"], f"{role} source run mismatch")
    _require(reuse.get("target_run") == target_config["project"]["run_name"], f"{role} target run mismatch")
    records = {int(record["agent_id"]): record for record in reuse.get("agents", [])}
    _require(sorted(records) == list(range(5)), f"{role} reuse manifest is incomplete")
    result = []
    for agent_id in range(5):
        source = source_dir / "agents" / f"agent_{agent_id}" / "checkpoints" / "expert_model.pt"
        target = target_dir / "agents" / f"agent_{agent_id}" / "checkpoints" / "expert_model.pt"
        _require(source.is_file() and target.is_file(), f"{role} agent {agent_id} expert is missing")
        source_sha = file_sha256(source)
        target_sha = file_sha256(target)
        _require(source_sha == target_sha, f"{role} agent {agent_id} expert SHA differs from source")
        recorded = records[agent_id].get("artifacts", {}).get("expert_model.pt", {}).get("sha256")
        _require(recorded == source_sha, f"{role} agent {agent_id} reuse provenance SHA mismatch")
        result.append({"agent_id": agent_id, "checkpoint_sha256": source_sha})
    return result


def _load_strict_packets(config, config_path, require_logits):
    args = build_dsdm_args_from_config(config, config_path=config_path)
    rows = read_packet_manifest(args, "dsdm")
    packets = {}
    for row in rows:
        agent_id = int(row["sender_agent"])
        packet_path = _resolve(row["packet_path"])
        packet = torch.load(packet_path, map_location="cpu", weights_only=False)
        _require(bool(packet.get("has_sender_logits", False)) == require_logits, f"agent {agent_id} logit presence mismatch")
        packets[agent_id] = (row, packet, packet_path)
    _require(sorted(packets) == list(range(5)), "packet set is incomplete")
    return args, rows, packets


def _validate_variant_packets(source_experts, ce_packets, full_packets):
    report = []
    source_sha = {record["agent_id"]: record["checkpoint_sha256"] for record in source_experts}
    for agent_id in range(5):
        ce_row, ce_packet, ce_path = ce_packets[agent_id]
        full_row, full_packet, full_path = full_packets[agent_id]
        _require(torch.equal(ce_packet["images"], full_packet["images"]), f"agent {agent_id} CE/full raw images differ")
        _require(torch.equal(ce_packet["labels"], full_packet["labels"]), f"agent {agent_id} CE/full labels differ")
        _require(ce_packet.get("class_ids") == full_packet.get("class_ids"), f"agent {agent_id} class order differs")
        logits = full_packet.get("sender_logits")
        _require(torch.is_tensor(logits) and tuple(logits.shape) == (800, 20), f"agent {agent_id} logit shape is invalid")
        _require(logits.dtype == torch.float16, f"agent {agent_id} transmitted logits are not float16")
        _require(bool(torch.isfinite(logits).all()), f"agent {agent_id} logits are non-finite")
        teacher_sha = full_packet.get("sender_logit_teacher", {}).get("checkpoint_sha256")
        _require(teacher_sha == source_sha[agent_id], f"agent {agent_id} sender teacher SHA mismatch")
        _require(full_row.get("expert_checkpoint_sha256") == source_sha[agent_id], f"agent {agent_id} manifest teacher SHA mismatch")
        _require(ce_row.get("expert_checkpoint_sha256", "") == "", f"agent {agent_id} CE manifest leaked teacher provenance")
        report.append(
            {
                "sender_agent": agent_id,
                "ce_packet_sha256": file_sha256(ce_path),
                "full_packet_sha256": file_sha256(full_path),
                "expert_checkpoint_sha256": source_sha[agent_id],
                "raw_images": int(full_packet["images"].shape[0]),
                "decoded_logits": int(logits.shape[0]),
                "logit_columns": int(logits.shape[1]),
                "logit_dtype": str(logits.dtype).replace("torch.", ""),
                "logit_bytes": int(logits.numel() * logits.element_size()),
                "decoded_alignment_sha256": full_packet["decoded_integrity"]["decoded_alignment_sha256"],
            }
        )
    return report


def _validate_receiver_prototypes(config, config_path, rows, source_experts, device):
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
        expert_path = _run_dir(config) / "agents" / f"agent_{receiver_id}" / "checkpoints" / "expert_model.pt"
        _require(file_sha256(expert_path) == source_sha[receiver_id], f"receiver {receiver_id} init expert SHA mismatch")
        for name, old_value in model_old.state_dict().items():
            _require(torch.equal(old_value, model_new.state_dict()[name]), f"receiver {receiver_id} init/FR state differs at {name}")

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
        from models.cosine_classifier import get_cosine_classifier

        classifier = get_cosine_classifier(model_new)
        index = torch.tensor(external_ids, device=classifier.weight.device, dtype=torch.long)
        norms = classifier.weight.detach().index_select(0, index).norm(dim=1).cpu()
        _require(len(external_ids) == 80, f"receiver {receiver_id} did not initialize 80 external classes")
        _require(bool(torch.isfinite(norms).all()), f"receiver {receiver_id} prototype norms are non-finite")
        reports.append(
            {
                "receiver_agent": receiver_id,
                "receiver_model": get_agent_model_split(config)[receiver_id],
                "init_checkpoint_sha256": source_sha[receiver_id],
                "fr_teacher_checkpoint_sha256": source_sha[receiver_id],
                "prototype_classes": len(external_ids),
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


def run_preflight(expert_path, ce_path, full_path, device="cuda"):
    expert_path, ce_path, full_path = map(_resolve, (expert_path, ce_path, full_path))
    expert = load_config(expert_path)
    ce_only = load_config(ce_path)
    full = load_config(full_path)
    class_split, model_split = validate_config_contract(expert, ce_only, full)
    ce_experts = _validate_reused_experts(expert, ce_only, "ce_only")
    full_experts = _validate_reused_experts(expert, full, "full")
    _require(ce_experts == full_experts, "CE/full expert provenance differs")
    _, _, ce_packets = _load_strict_packets(ce_only, ce_path, require_logits=False)
    _, full_rows, full_packets = _load_strict_packets(full, full_path, require_logits=True)
    packet_report = _validate_variant_packets(full_experts, ce_packets, full_packets)
    receiver_report = _validate_receiver_prototypes(full, full_path, full_rows, full_experts, device)
    return {
        "status": "passed",
        "protocol": full["communication"]["protocol"],
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
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--report-json", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    cli = parse_args(argv)
    try:
        report = run_preflight(cli.expert_config, cli.ce_config, cli.full_config, device=cli.device)
        atomic_write_json(report, _resolve(cli.report_json))
    except Exception as exc:
        print(f"[dkp-communication-preflight] FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
