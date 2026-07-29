#!/usr/bin/env python3
"""Fail-closed packet, expert and prototype preflight for DKP scaling runs."""

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

from agent_data import get_agent_class_split, get_agent_model_split  # noqa: E402
from config_adapter import build_dsdm_args_from_config, load_config  # noqa: E402
from output_manager import atomic_write_json  # noqa: E402
from packet_consumer import (  # noqa: E402
    consume_external_manifest_packets,
    consume_packet_for_training,
    consume_receiver_manifest_packet,
    load_receiver_local_real_data,
)
from packet_integrity import file_sha256, resolve_strict_dkp_contract  # noqa: E402
from run_social_pipeline import build_receiver_args  # noqa: E402
from social_output_manager import read_packet_manifest  # noqa: E402
from social_trainer import SocialTrainer, resolve_class_balanced_ce_weights  # noqa: E402


class ScalingPreflightError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise ScalingPreflightError(message)


def _resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _run_dir(config):
    return _resolve(config["project"].get("output_root", "outputs")) / config["project"]["run_name"]


def _validate_config_pair(expert, full):
    expert_contract = resolve_strict_dkp_contract(expert)
    full_contract = resolve_strict_dkp_contract(full)
    _require(expert_contract == full_contract, "expert/full strict DKP contracts differ")
    _require(get_agent_class_split(expert) == get_agent_class_split(full), "expert/full class splits differ")
    _require(get_agent_model_split(expert) == get_agent_model_split(full), "expert/full model splits differ")
    _require(
        full.get("expert_reuse", {}).get("source_run") == expert["project"]["run_name"],
        "full config points to the wrong expert run",
    )
    _require(full["communication"].get("use_sender_logits") is True, "full scaling requires sender logits")
    receiver = full["social_learning"]["receiver"]
    _require(receiver.get("dkp_variant") == "full", "scaling receiver must use the full loss endpoint")
    _require(
        receiver.get("loss_switches") == {"fr": True, "kd": True, "supcon": True},
        "scaling loss switches drifted",
    )
    fixed_values = {
        "local_ce_real_fraction": 0.02,
        "optimizer_steps": 3780,
        "lambda_fr": 0.2,
        "lambda_sc": 0.1,
        "prototype_decoded_per_class": 40,
        "packet_raw_per_class": 10,
    }
    for field, expected in fixed_values.items():
        actual = receiver.get(field)
        _require(
            math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12),
            f"receiver.{field} drifted: {actual} != {expected}",
        )
    _require(full["logits"].get("lambda_kd") == 0.6, "lambda_KD drifted")
    _require(full["logits"].get("temperature") == 2.0, "KD temperature drifted")
    _require(receiver.get("scheduler_unit") == "optimizer_step", "scheduler must advance per step")
    _require(receiver.get("scheduler_step_milestones") == [2457, 3213], "step milestones drifted")
    return full_contract


def _validate_expert_reuse(expert, full, contract):
    source_dir = _run_dir(expert)
    target_dir = _run_dir(full)
    manifest_path = target_dir / "provenance/expert_reuse_manifest.json"
    _require(manifest_path.is_file(), f"expert reuse manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("protocol") == contract.protocol, "expert reuse protocol drifted")
    _require(int(manifest.get("agent_count", -1)) == contract.agent_count, "expert reuse agent count drifted")
    records = {int(record["agent_id"]): record for record in manifest.get("agents", [])}
    _require(sorted(records) == list(range(contract.agent_count)), "expert reuse manifest is incomplete")
    result = []
    for agent_id in range(contract.agent_count):
        source = source_dir / f"agents/agent_{agent_id}/checkpoints/expert_model.pt"
        target = target_dir / f"agents/agent_{agent_id}/checkpoints/expert_model.pt"
        _require(source.is_file() and target.is_file(), f"agent {agent_id} expert checkpoint is missing")
        source_sha = file_sha256(source)
        _require(file_sha256(target) == source_sha, f"agent {agent_id} reused expert SHA differs")
        recorded = records[agent_id].get("artifacts", {}).get("expert_model.pt", {}).get("sha256")
        _require(recorded == source_sha, f"agent {agent_id} expert reuse SHA provenance differs")
        result.append(
            {
                "agent_id": agent_id,
                "model": contract.model_split[agent_id],
                "checkpoint_sha256": source_sha,
            }
        )
    return result


def _validate_packets(full, full_path, contract, expert_reports):
    args = build_dsdm_args_from_config(full, config_path=full_path)
    rows = read_packet_manifest(args, "dsdm")
    expert_sha = {record["agent_id"]: record["checkpoint_sha256"] for record in expert_reports}
    reports = []
    for row in rows:
        agent_id = int(row["sender_agent"])
        packet_path = _resolve(row["packet_path"])
        packet = torch.load(packet_path, map_location="cpu", weights_only=False)
        consumed = consume_packet_for_training(args, packet_path, require_sender_logits=True)
        logits = consumed["sender_logits"]
        _require(int(consumed["raw_images"]) == contract.raw_per_sender, f"agent {agent_id} raw count drifted")
        _require(int(consumed["num_images"]) == contract.decoded_per_sender, f"agent {agent_id} decoded count drifted")
        _require(
            tuple(logits.shape) == (contract.decoded_per_sender, contract.classes_per_agent),
            f"agent {agent_id} sender-logit shape drifted",
        )
        _require(logits.dtype == torch.float16, f"agent {agent_id} sender logits are not float16")
        _require(bool(torch.isfinite(logits).all()), f"agent {agent_id} sender logits are non-finite")
        _require(
            consumed["class_ids"] == contract.class_split[agent_id],
            f"agent {agent_id} packet class order drifted",
        )
        teacher_sha = packet.get("sender_logit_teacher", {}).get("checkpoint_sha256")
        _require(teacher_sha == expert_sha[agent_id], f"agent {agent_id} sender teacher SHA differs")
        _require(row.get("expert_checkpoint_sha256") == teacher_sha, f"agent {agent_id} manifest teacher SHA differs")
        reports.append(
            {
                "sender_agent": agent_id,
                "model": contract.model_split[agent_id],
                "packet_sha256": file_sha256(packet_path),
                "raw_images": int(consumed["raw_images"]),
                "decoded_images": int(consumed["num_images"]),
                "sender_logit_shape": list(logits.shape),
                "sender_logit_bytes": int(consumed["sender_logit_bytes"]),
                "expert_checkpoint_sha256": teacher_sha,
            }
        )
    _require(len(reports) == contract.agent_count, "packet report does not cover every sender")
    return args, rows, reports


def _validate_receivers(full, full_path, rows, contract, expert_reports, device):
    expected_sha = {record["agent_id"]: record["checkpoint_sha256"] for record in expert_reports}
    reports = []
    for receiver_id in range(contract.agent_count):
        args = build_receiver_args(full, str(full_path), receiver_id, packet_method="dsdm", init_mode="expert")
        args.device = str(device)
        trainer = SocialTrainer(args, receiver_id, rows)
        checkpoint_sha = trainer._receiver_expert_checkpoint_sha(require_manifest_match=True)
        _require(checkpoint_sha == expected_sha[receiver_id], f"receiver {receiver_id} expert SHA differs")
        model_old, model_new = trainer._build_models()
        for name, value in model_old.state_dict().items():
            _require(torch.equal(value, model_new.state_dict()[name]), f"receiver {receiver_id} init/FR state differs at {name}")

        local = load_receiver_local_real_data(args)
        external = consume_external_manifest_packets(args, rows, receiver_id, require_logits=True)
        self_packet = consume_receiver_manifest_packet(args, rows, receiver_id)
        trainer._validate_dkp_streams(local, external, require_logits=True, self_packet=self_packet)
        external_images = trainer._normalize_images(external["images"])
        external_ids = trainer._initialize_external_prototypes(
            model_old,
            model_new,
            external_images,
            external["labels"],
        )
        local_weight, external_weight = resolve_class_balanced_ce_weights(
            contract.class_split[receiver_id],
            contract.num_classes,
        )
        _require(len(external_ids) == contract.num_classes - contract.classes_per_agent, f"receiver {receiver_id} prototype coverage drifted")
        reports.append(
            {
                "receiver_agent": receiver_id,
                "model": contract.model_split[receiver_id],
                "expert_checkpoint_sha256": checkpoint_sha,
                "prototype_classes": len(external_ids),
                "external_raw_images": (contract.agent_count - 1) * contract.raw_per_sender,
                "external_decoded_images": int(external["images"].shape[0]),
                "external_logit_bytes": sum(packet["sender_logit_bytes"] for packet in external["packets"]),
                "ce_local_weight": local_weight,
                "ce_external_weight": external_weight,
            }
        )
        del trainer, model_old, model_new, local, external, self_packet, external_images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return reports


def run_preflight(expert_config, full_config, device="cuda"):
    expert_path = _resolve(expert_config)
    full_path = _resolve(full_config)
    expert = load_config(expert_path)
    full = load_config(full_path)
    contract = _validate_config_pair(expert, full)
    expert_reports = _validate_expert_reuse(expert, full, contract)
    _, rows, packet_reports = _validate_packets(full, full_path, contract, expert_reports)
    receiver_reports = _validate_receivers(
        full,
        full_path,
        rows,
        contract,
        expert_reports,
        device,
    )
    return {
        "status": "passed",
        "protocol": contract.protocol,
        "agent_count": contract.agent_count,
        "classes_per_agent": contract.classes_per_agent,
        "ipc": contract.ipc,
        "factor": contract.factor,
        "expert_config": str(expert_path),
        "full_config": str(full_path),
        "experts": expert_reports,
        "packets": packet_reports,
        "receivers": receiver_reports,
        "unique_sender_payload": {
            "raw_images": contract.agent_count * contract.raw_per_sender,
            "sender_logit_bytes": sum(report["sender_logit_bytes"] for report in packet_reports),
        },
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-config", required=True)
    parser.add_argument("--full-config", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--report-json", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    cli = parse_args(argv)
    try:
        report = run_preflight(cli.expert_config, cli.full_config, device=cli.device)
        atomic_write_json(report, _resolve(cli.report_json))
    except Exception as error:
        print(f"[dkp-scaling-preflight] FAILED: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
