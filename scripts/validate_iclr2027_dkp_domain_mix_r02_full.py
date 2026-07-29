#!/usr/bin/env python3
"""Fail-closed packet, logit, expert, and prototype gate for r02 full DKP-SL."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_data import build_agent_args  # noqa: E402
from config_adapter import load_config  # noqa: E402
from output_manager import atomic_write_json  # noqa: E402
from packet_consumer import (  # noqa: E402
    consume_external_manifest_packets,
    consume_receiver_manifest_packet,
    load_receiver_local_real_data,
)
from packet_integrity import file_sha256  # noqa: E402
from scripts.prepare_iclr2027_dkp_domain_mix_r02 import TARGET_CONFIG as R02_CONFIG  # noqa: E402
from scripts.prepare_iclr2027_dkp_domain_mix_r02_full import (  # noqa: E402
    FRACTION,
    TARGET_CONFIG,
    validate_config,
)
from scripts.validate_iclr2027_dkp_communication import (  # noqa: E402
    _load_strict_packets,
    _validate_receiver_prototypes,
    _validate_reused_experts,
    _validate_variant_packets,
)
from scripts.validate_iclr2027_dkp_domain_balance import EXPERT_CONFIG  # noqa: E402
from social_trainer import SocialTrainer  # noqa: E402


class R02FullPreflightError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise R02FullPreflightError(message)


def _resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def run_preflight(device="cuda"):
    expert = load_config(EXPERT_CONFIG)
    image_only = load_config(R02_CONFIG)
    full = load_config(TARGET_CONFIG)
    validate_config(full)
    _require(expert["agents"] == full["agents"], "expert/full agent protocol differs")
    _require(image_only["agents"] == full["agents"], "r02 image/full agent protocol differs")
    source_experts = _validate_reused_experts(expert, full, "r02_full")
    _, _, image_packets = _load_strict_packets(image_only, R02_CONFIG, require_logits=False)
    _, rows, full_packets = _load_strict_packets(full, TARGET_CONFIG, require_logits=True)
    packet_reports = _validate_variant_packets(source_experts, image_packets, full_packets)
    prototype_reports = _validate_receiver_prototypes(
        full,
        TARGET_CONFIG,
        rows,
        source_experts,
        device,
    )
    source_shas = {record["agent_id"]: record["checkpoint_sha256"] for record in source_experts}
    receiver_reports = []
    for receiver_id in range(5):
        args = build_agent_args(full, TARGET_CONFIG, receiver_id)
        args.device = str(device)
        args.init_mode = "expert"
        args.packet_method = "dsdm"
        trainer = SocialTrainer(args, receiver_id, rows)
        receiver_sha = trainer._receiver_expert_checkpoint_sha(require_manifest_match=True)
        _require(receiver_sha == source_shas[receiver_id], f"receiver {receiver_id} init/FR SHA mismatch")
        local = load_receiver_local_real_data(args)
        external = consume_external_manifest_packets(args, rows, receiver_id, require_logits=True)
        self_packet = consume_receiver_manifest_packet(args, rows, receiver_id)
        trainer._validate_dkp_streams(local, external, require_logits=True, self_packet=self_packet)
        _require(int(self_packet["raw_images"]) == 200, f"receiver {receiver_id} self raw count mismatch")
        _require(int(self_packet["num_images"]) == 800, f"receiver {receiver_id} self decoded count mismatch")
        _require(int(external["images"].shape[0]) == 3200, f"receiver {receiver_id} external count mismatch")
        _require(tuple(external["sender_logits"].shape) == (3200, 20), f"receiver {receiver_id} logit shape mismatch")
        _require(math.isclose(args.receiver_local_ce_real_fraction, FRACTION), f"receiver {receiver_id} fraction mismatch")
        receiver_reports.append(
            {
                "receiver_agent": receiver_id,
                "init_checkpoint_sha256": receiver_sha,
                "fr_teacher_checkpoint_sha256": receiver_sha,
                "self_packet_raw_images": 200,
                "self_packet_decoded_images": 800,
                "external_decoded_images": 3200,
                "external_sender_logits_shape": [3200, 20],
                "external_logit_bytes": sum(packet["sender_logit_bytes"] for packet in external["packets"]),
                "local_ce_real_fraction": FRACTION,
            }
        )
    return {
        "status": "passed",
        "formal_result": False,
        "paper_eligible": False,
        "condition": "r02_full",
        "config": str(Path(TARGET_CONFIG).resolve()),
        "config_sha256": file_sha256(TARGET_CONFIG),
        "paired_image_only_config": str(Path(R02_CONFIG).resolve()),
        "paired_raw_images_and_labels_identical": True,
        "expert_provenance": source_experts,
        "packets": packet_reports,
        "prototypes": prototype_reports,
        "receivers": receiver_reports,
        "sender_logits_required": True,
        "adaptive_after_completed_seed0_curve": True,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--report-json", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    cli = parse_args(argv)
    try:
        output = _resolve(cli.report_json)
        _require(not output.exists(), f"refusing to overwrite preflight report: {output}")
        report = run_preflight(cli.device)
        atomic_write_json(report, output)
    except Exception as exc:
        print(f"ICLR2027 r02 full preflight FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "passed", "report": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
