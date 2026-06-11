"""Compare v2 local-only, raw IPC, and strict DSDM IPC metrics."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable

import torch

from src.packet.packet_dataclass import SocialPacket
from src.utils.config import load_yaml
from src.utils.v2_progress import StageTimer, progress
from src.utils.v2_paths import get_v2_comparison_dir, get_v2_metrics_dir, get_v2_packet_dir


VARIANTS = {
    "local_only": {"stage": "expert", "packet_source": None},
    "raw_ipc": {"stage": "socialized", "packet_source": "raw"},
    "strict_dsdm_ipc": {"stage": "socialized", "packet_source": "strict_dsdm"},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare v2 experiment variants.")
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def metrics_path(cfg: dict, variant: str) -> Path:
    if variant == "local_only":
        return get_v2_metrics_dir(cfg) / "local_only" / "metrics_v2.json"
    return get_v2_metrics_dir(cfg) / variant / "metrics_v2.json"


def read_metrics(cfg: dict, variant: str) -> Dict:
    path = metrics_path(cfg, variant)
    if not path.exists():
        return {
            "available": False,
            "metrics_path": str(path),
            "average_expert_accuracy": None,
            "average_new_accuracy": None,
            "average_overall_accuracy": None,
            "num_agents": 0,
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary", {})
    agents = data.get("agents", {})
    return {
        "available": True,
        "metrics_path": str(path),
        "average_expert_accuracy": summary.get("average_expert_accuracy"),
        "average_new_accuracy": summary.get("average_new_accuracy"),
        "average_overall_accuracy": summary.get("average_overall_accuracy"),
        "num_agents": len(agents),
    }


def packet_communication_stats(cfg: dict, packet_source: str) -> Dict[str, int]:
    if packet_source is None:
        return {
            "packet_count": 0,
            "bytes_images": 0,
            "bytes_labels": 0,
            "bytes_total": 0,
        }

    packet_dir = get_v2_packet_dir(cfg, packet_source)
    totals = {
        "packet_count": 0,
        "bytes_images": 0,
        "bytes_labels": 0,
        "bytes_total": 0,
    }
    if not packet_dir.exists():
        return totals

    for packet_path in sorted(packet_dir.glob("agent_*_packet.pt")):
        packet: SocialPacket = torch_load(packet_path, map_location="cpu")
        stats = packet.byte_stats()
        totals["packet_count"] += 1
        totals["bytes_images"] += int(stats["bytes_images"])
        totals["bytes_labels"] += int(stats["bytes_labels"])
        totals["bytes_total"] += int(stats["bytes_total"])
    return totals


def write_csv(path: Path, rows: Iterable[Dict]) -> None:
    fieldnames = [
        "variant",
        "available",
        "num_agents",
        "average_expert_accuracy",
        "average_new_accuracy",
        "average_overall_accuracy",
        "packet_count",
        "bytes_images",
        "bytes_labels",
        "bytes_total",
        "metrics_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    print("=== run_compare_v2 ===")
    print(f"config: {args.config}")
    print(f"experiment: {cfg.get('experiment', {}).get('name')}")

    comparison = {
        "experiment": cfg["experiment"]["name"],
        "variants": {},
    }
    rows = []
    with StageTimer("run_compare_v2 total"):
        for variant, variant_cfg in progress(VARIANTS.items(), desc="compare variants"):
            metric_stats = read_metrics(cfg, variant)
            comm_stats = packet_communication_stats(cfg, variant_cfg["packet_source"])
            row = {
                "variant": variant,
                **metric_stats,
                **comm_stats,
            }
            comparison["variants"][variant] = row
            rows.append(row)
            if metric_stats["available"]:
                print(
                    f"{variant}: expert={metric_stats['average_expert_accuracy']:.4f} "
                    f"new={metric_stats['average_new_accuracy']:.4f} "
                    f"overall={metric_stats['average_overall_accuracy']:.4f} "
                    f"bytes={comm_stats['bytes_total']}"
                )
            else:
                print(f"{variant}: metrics missing at {metric_stats['metrics_path']} bytes={comm_stats['bytes_total']}")

    out_dir = get_v2_comparison_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "comparison_v2.json"
    csv_path = out_dir / "comparison_v2.csv"
    json_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)
    print(f"saved comparison json: {json_path}")
    print(f"saved comparison csv: {csv_path}")


if __name__ == "__main__":
    main()
