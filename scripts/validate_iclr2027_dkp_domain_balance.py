#!/usr/bin/env python3
"""Fail-closed artifact gate for the seed0 DKP S/U/H domain diagnostic."""

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
from packet_consumer import (  # noqa: E402
    consume_external_manifest_packets,
    consume_receiver_manifest_packet,
    load_receiver_local_real_data,
)
from packet_integrity import file_sha256  # noqa: E402
from scripts.prepare_iclr2027_dkp_domain_balance import (  # noqa: E402
    BASE_CONFIG,
    CONDITIONS,
    EXPECTED_LOSSES,
    EXPECTED_STEP_MILESTONES,
    EXPECTED_STEPS,
    run_name,
    validate_config,
)
from social_output_manager import read_packet_manifest  # noqa: E402
from social_trainer import SocialTrainer  # noqa: E402


CONFIG_DIR = ROOT / "configs/iclr2027"
EXPERT_CONFIG = CONFIG_DIR / "cifar100_5agent20cls_dkp_cosine_experts_seed0_v1.yaml"
EXPERT_RUN = "cifar100_5agent20cls_dkp_cosine_experts_seed0_v1"
POOL_PROTOCOL = "backbone_specific_fullclass_pool_slice_v2_dkp"


class DomainBalancePreflightError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise DomainBalancePreflightError(message)


def _resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _config_path(config_dir, condition):
    return Path(config_dir) / f"{run_name(condition)}.yaml"


def _run_dir(config):
    return _resolve(config["project"].get("output_root", "outputs")) / config["project"]["run_name"]


def _source_experts():
    expert = load_config(EXPERT_CONFIG)
    _require(expert["project"]["run_name"] == EXPERT_RUN, "expert source run drifted")
    source_dir = _run_dir(expert)
    classes = get_agent_class_split(expert)
    models = get_agent_model_split(expert)
    records = {}
    for agent_id in range(5):
        checkpoint = source_dir / "agents" / f"agent_{agent_id}" / "checkpoints/expert_model.pt"
        _require(checkpoint.is_file(), f"source expert is missing: {checkpoint}")
        records[agent_id] = {
            "model": models[agent_id],
            "class_ids": classes[agent_id],
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": file_sha256(checkpoint),
        }
    return records


def _validate_reused_experts(config, sources):
    target_dir = _run_dir(config)
    path = target_dir / "provenance/expert_reuse_manifest.json"
    _require(path.is_file(), f"expert reuse manifest is missing: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DomainBalancePreflightError(f"invalid expert reuse manifest: {path}") from exc
    _require(manifest.get("source_run") == EXPERT_RUN, f"expert reuse source mismatch: {path}")
    _require(manifest.get("target_run") == config["project"]["run_name"], f"expert reuse target mismatch: {path}")
    records = {int(record["agent_id"]): record for record in manifest.get("agents", [])}
    _require(sorted(records) == list(range(5)), f"expert reuse manifest is incomplete: {path}")
    for agent_id, source in sources.items():
        target = target_dir / "agents" / f"agent_{agent_id}" / "checkpoints/expert_model.pt"
        _require(target.is_file(), f"reused expert is missing: {target}")
        target_sha = file_sha256(target)
        _require(target_sha == source["checkpoint_sha256"], f"agent {agent_id} expert SHA differs from source")
        recorded = records[agent_id].get("artifacts", {}).get("expert_model.pt", {}).get("sha256")
        _require(recorded == target_sha, f"agent {agent_id} expert reuse provenance SHA mismatch")


def _validate_packets(config, config_path, sources, reference=None):
    args = build_agent_args(config, config_path, 0)
    rows = read_packet_manifest(args, "dsdm")
    rows_by_agent = {int(row["sender_agent"]): row for row in rows}
    _require(len(rows) == 5 and sorted(rows_by_agent) == list(range(5)), f"{config_path.name}: sender manifest is incomplete")
    tensor_records = {}
    reports = []
    for sender_id in range(5):
        row = rows_by_agent[sender_id]
        packet_path = _resolve(row["packet_path"])
        _require(packet_path.is_file(), f"sender {sender_id} packet is missing: {packet_path}")
        _require(file_sha256(packet_path) == row["packet_sha256"], f"sender {sender_id} packet SHA mismatch")
        packet = torch.load(packet_path, map_location="cpu", weights_only=False)
        expected_classes = sources[sender_id]["class_ids"]
        _require(packet.get("source") == "dsdm", f"sender {sender_id} packet source is not DSDM")
        _require(packet.get("class_ids") == expected_classes, f"sender {sender_id} class order mismatch")
        _require(tuple(packet["images"].shape) == (200, 3, 32, 32), f"sender {sender_id} raw image shape mismatch")
        _require(tuple(packet["labels"].shape) == (200,), f"sender {sender_id} label shape mismatch")
        _require(bool(torch.isfinite(packet["images"]).all()), f"sender {sender_id} images are non-finite")
        _require(sorted(int(value) for value in torch.unique(packet["labels"]).tolist()) == sorted(expected_classes), f"sender {sender_id} global labels mismatch")
        _require(packet.get("has_sender_logits", False) is False, f"sender {sender_id} packet unexpectedly contains logits")
        _require("sender_logits" not in packet, f"sender {sender_id} packet leaked sender logits")
        _require(row.get("sender_logits_sha256", "") == "", f"sender {sender_id} manifest leaked logit SHA")
        _require(row.get("decoded_alignment_sha256", "") == "", f"sender {sender_id} manifest leaked logit alignment SHA")
        _require(row.get("expert_checkpoint_sha256", "") == "", f"sender {sender_id} image-only manifest leaked teacher SHA")
        meta = packet.get("meta", {})
        _require(meta.get("pool_protocol") == POOL_PROTOCOL, f"sender {sender_id} pool protocol mismatch")
        _require(meta.get("sender_model") == sources[sender_id]["model"], f"sender {sender_id} pool/model mismatch")
        _require(meta.get("sender_class_ids") == expected_classes, f"sender {sender_id} pool class order mismatch")
        _require(meta.get("guide_weights_communicated") is False, f"sender {sender_id} propagated guide weights")
        decoded = packet.get("decoded_integrity", {})
        _require(decoded.get("decoded_images_sha256") == row.get("decoded_images_sha256"), f"sender {sender_id} decoded image SHA mismatch")
        _require(decoded.get("decoded_labels_sha256") == row.get("decoded_labels_sha256"), f"sender {sender_id} decoded label SHA mismatch")
        _require(all(isinstance(decoded.get(key), str) and len(decoded[key]) == 64 for key in ("decoded_images_sha256", "decoded_labels_sha256")), f"sender {sender_id} decoded integrity is incomplete")
        tensor_records[sender_id] = {
            "images": packet["images"].cpu(),
            "labels": packet["labels"].cpu(),
            "decoded_images_sha256": decoded["decoded_images_sha256"],
            "decoded_labels_sha256": decoded["decoded_labels_sha256"],
            "pool_source_sha256": row["pool_source_sha256"],
            "pool_source_best_snapshot_sha256": row["pool_source_best_snapshot_sha256"],
        }
        if reference is not None:
            for key, actual in tensor_records[sender_id].items():
                expected = reference[sender_id][key]
                if torch.is_tensor(actual):
                    _require(torch.equal(actual, expected), f"sender {sender_id} {key} differs across S/U/H")
                else:
                    _require(actual == expected, f"sender {sender_id} {key} differs across S/U/H")
        reports.append(
            {
                "sender_agent": sender_id,
                "sender_model": sources[sender_id]["model"],
                "packet_path": str(packet_path),
                "packet_sha256": row["packet_sha256"],
                "raw_images": 200,
                "decoded_images": 800,
                "logit_bytes": 0,
                "decoded_images_sha256": decoded["decoded_images_sha256"],
                "decoded_labels_sha256": decoded["decoded_labels_sha256"],
            }
        )
    return rows, tensor_records, reports


def _validate_receivers(config, config_path, rows, sources, device):
    reports = []
    for receiver_id in range(5):
        args = build_agent_args(config, config_path, receiver_id)
        args.device = str(device)
        args.init_mode = "expert"
        args.use_logits = False
        args.use_generalist_logits = False
        args.packet_method = "dsdm"
        trainer = SocialTrainer(args, receiver_id, rows)
        receiver_sha = trainer._receiver_expert_checkpoint_sha(require_manifest_match=False)
        _require(receiver_sha == sources[receiver_id]["checkpoint_sha256"], f"receiver {receiver_id} init/FR expert SHA mismatch")
        local = load_receiver_local_real_data(args)
        external = consume_external_manifest_packets(args, rows, receiver_id, require_logits=False)
        self_packet = consume_receiver_manifest_packet(args, rows, receiver_id)
        trainer._validate_dkp_streams(local, external, require_logits=False, self_packet=self_packet)
        _require(int(local["num_images"]) == 10_000, f"receiver {receiver_id} local-real count mismatch")
        _require(int(self_packet["raw_images"]) == 200, f"receiver {receiver_id} raw self-packet count mismatch")
        _require(int(self_packet["num_images"]) == 800, f"receiver {receiver_id} decoded self-packet count mismatch")
        _require(int(external["images"].shape[0]) == 3_200, f"receiver {receiver_id} external decoded count mismatch")

        model_old, model_new = trainer._build_models()
        external_images = trainer._normalize_images(external["images"])
        external_ids = trainer._initialize_external_prototypes(
            model_old,
            model_new,
            external_images,
            external["labels"],
        )
        prototype = trainer._prototype_init_stats
        _require(len(external_ids) == 80, f"receiver {receiver_id} prototype class count mismatch")
        _require(prototype.get("classifier_type") == "cosine", f"receiver {receiver_id} classifier is not cosine")
        _require(prototype.get("local_rows_preserved") is True, f"receiver {receiver_id} local rows changed")
        for key in ("external_weight_norm_min", "external_weight_norm_max"):
            value = float(prototype[key])
            _require(math.isfinite(value) and abs(value - 1.0) <= 1e-6, f"receiver {receiver_id} prototype norm mismatch")
        reports.append(
            {
                "receiver_agent": receiver_id,
                "receiver_model": sources[receiver_id]["model"],
                "local_ce_source": config["social_learning"]["receiver"]["local_ce_source"],
                "local_real_images": 10_000,
                "self_packet_raw_images": 200,
                "self_packet_decoded_images": 800,
                "external_decoded_images": 3_200,
                "prototype_initialized_classes": 80,
                "prototype_weight_norm_min": float(prototype["external_weight_norm_min"]),
                "prototype_weight_norm_max": float(prototype["external_weight_norm_max"]),
                "init_checkpoint_sha256": receiver_sha,
                "fr_teacher_checkpoint_sha256": receiver_sha,
            }
        )
        del model_old, model_new, trainer, local, external, self_packet, external_images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return reports


def run_preflight(config_dir=CONFIG_DIR, device="cuda"):
    config_dir = _resolve(config_dir)
    base = load_config(BASE_CONFIG)
    sources = _source_experts()
    reference = None
    conditions = {}
    for condition in CONDITIONS:
        config_path = _config_path(config_dir, condition)
        _require(config_path.is_file(), f"domain config is missing: {config_path}")
        config = load_config(config_path)
        validate_config(config, condition, base)
        _validate_reused_experts(config, sources)
        rows, tensors, packets = _validate_packets(config, config_path, sources, reference)
        if reference is None:
            reference = tensors
        receivers = _validate_receivers(config, config_path, rows, sources, device)
        conditions[condition] = {
            "run_name": config["project"]["run_name"],
            "local_ce_source": CONDITIONS[condition],
            "config": str(config_path.resolve()),
            "config_sha256": file_sha256(config_path),
            "loss_switches": EXPECTED_LOSSES,
            "optimizer_steps": EXPECTED_STEPS,
            "scheduler_step_milestones": EXPECTED_STEP_MILESTONES,
            "packets": packets,
            "receivers": receivers,
        }
    return {
        "status": "passed",
        "formal_result": False,
        "paper_eligible": False,
        "protocol": "dkp_sl_v1",
        "expert_source_run": EXPERT_RUN,
        "expert_provenance": sources,
        "conditions": conditions,
        "cross_condition_packet_tensors_identical": True,
        "sender_logits_disabled": True,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default=str(CONFIG_DIR))
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--report-json", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    cli = parse_args(argv)
    try:
        output = _resolve(cli.report_json)
        _require(not output.exists(), f"refusing to overwrite preflight report: {output}")
        report = run_preflight(cli.config_dir, cli.device)
        atomic_write_json(report, output)
    except Exception as exc:
        print(f"ICLR2027 domain-balance preflight FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "passed", "report": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
