#!/usr/bin/env python3
"""Summarize completed R10 PCBN weight candidates without promoting them to formal results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from output_manager import atomic_write_json  # noqa: E402


OUTPUT = ROOT / "experiments/diagnostics/cifar100_r10_pcbn_weight_sweep_seed0.json"
RUNS = {
    "pure": "cifar100_fullclass_dsdm_resnet10_standard_modelbest_e0200_ipc10_seed0_recovery",
    "w960": "cifar100_fullclass_dsdm_resnet10_standard_modelbest_e0200_ipc10_seed0_recovery_pcbn_w960",
    "w1300": "cifar100_fullclass_dsdm_resnet10_standard_modelbest_e0200_ipc10_seed0_recovery_pcbn_w1300",
    "w2100": "cifar100_fullclass_dsdm_resnet10_standard_modelbest_e0200_ipc10_seed0_recovery_pcbn_w2100",
    "w3400": "cifar100_fullclass_dsdm_resnet10_standard_modelbest_e0200_ipc10_seed0_recovery_pcbn_w3400",
}


def inspect_run(label: str, run_name: str) -> dict:
    root = ROOT / "outputs" / run_name
    manifest_path = root / "agents/agent_0/synthetic/best_manifest.json"
    packet_path = root / "agents/agent_0/packets/dsdm_packet.pt"
    row = {"label": label, "run_name": run_name, "present": root.exists(), "complete": False}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        row.update(
            {
                "best_accuracy": float(manifest["best_acc"]),
                "best_iteration": int(manifest["iteration"]),
                "pcbn_enabled": bool(manifest.get("pcbn_enabled", label != "pure")),
                "pcbn_weight": float(manifest.get("pcbn_weight", 0.0)),
            }
        )
    if packet_path.exists():
        packet = torch.load(packet_path, map_location="cpu", weights_only=False)
        meta = packet.get("meta", {})
        row["complete"] = bool(meta.get("condense_complete", False)) and int(
            meta.get("completed_iterations", 0)
        ) == 10000
        row["raw_images"] = int(packet["images"].shape[0])
        row["finite"] = bool(torch.isfinite(packet["images"]).all().item())
    return row


def main() -> None:
    rows = [inspect_run(label, run_name) for label, run_name in RUNS.items()]
    completed = [row for row in rows if row.get("complete") and row.get("finite")]
    selected = max(completed, key=lambda row: row["best_accuracy"]) if completed else None
    payload = {
        "status": "complete" if all(row["complete"] for row in rows if row["label"] != "w3400") else "interim",
        "role": "single-seed packet-quality tuning; not a formal social-learning result",
        "selection_rule": "highest synthetic evaluator accuracy among completed candidates",
        "selected": selected,
        "runs": rows,
    }
    atomic_write_json(payload, OUTPUT)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"saved: {OUTPUT}")


if __name__ == "__main__":
    main()
