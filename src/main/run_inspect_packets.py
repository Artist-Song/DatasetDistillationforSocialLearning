"""
Inspect cached social packets against their sender anchor models.

This diagnostic script checks whether packet images carry usable sender
knowledge before receiver social training consumes them. It reports anchor
accuracy/confidence, soft-target entropy, and simple image statistics, and it
saves packet image grids for visual inspection.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from src.main.run_eval import build_model
from src.main.run_local_pretrain import resolve_device
from src.utils.agent_selection import parse_agent_ids
from src.utils.config import load_yaml
from src.utils.run_name import build_base_run_name
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="experiment config path")
    parser.add_argument(
        "--agent-ids",
        type=str,
        default="all",
        help='sender agents to inspect, e.g. "all", "0", "0,2,4", or "0-3"',
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=64,
        help="maximum packet images to save per agent grid",
    )
    return parser.parse_args()


def load_torch_object(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def tensor_entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * probs.clamp_min(1e-8).log()).sum(dim=1)


def packet_image_stats(images: torch.Tensor):
    return {
        "image_min": float(images.min().item()),
        "image_max": float(images.max().item()),
        "image_mean": float(images.mean().item()),
        "image_std": float(images.std(unbiased=False).item()),
    }


@torch.no_grad()
def inspect_packet(anchor_model, packet, device: torch.device):
    images = packet.images.to(device)
    hard_labels = packet.hard_labels.to(device)
    soft_targets = packet.soft_targets.to(device)

    anchor_model.eval()
    logits = anchor_model(images)
    probs = F.softmax(logits, dim=1)
    pred = probs.argmax(dim=1)
    top_conf = probs.max(dim=1).values
    hard_conf = probs.gather(dim=1, index=hard_labels.view(-1, 1)).squeeze(1)
    soft_entropy = tensor_entropy(soft_targets)
    anchor_entropy = tensor_entropy(probs)

    return {
        "num_images": int(images.size(0)),
        "anchor_packet_accuracy": float((pred == hard_labels).float().mean().item()),
        "anchor_top1_confidence": float(top_conf.mean().item()),
        "anchor_hard_label_confidence": float(hard_conf.mean().item()),
        "anchor_prediction_entropy": float(anchor_entropy.mean().item()),
        "soft_target_entropy": float(soft_entropy.mean().item()),
        "soft_target_max_probability": float(soft_targets.max(dim=1).values.mean().item()),
        **packet_image_stats(packet.images),
    }


def write_markdown(report, path: Path):
    rows = report["agents"]
    fieldnames = [
        "agent_id",
        "packet_source",
        "num_images",
        "anchor_packet_accuracy",
        "anchor_top1_confidence",
        "anchor_hard_label_confidence",
        "soft_target_entropy",
        "soft_target_max_probability",
        "image_mean",
        "image_std",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for row in rows:
            values = []
            for name in fieldnames:
                value = row[name]
                if isinstance(value, float):
                    values.append(f"{value:.4f}")
                else:
                    values.append(str(value))
            f.write("| " + " | ".join(values) + " |\n")


def save_packet_grid(packet, path: Path, max_images: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    images = packet.images[:max_images].detach().cpu().clamp(0.0, 1.0)
    nrow = min(10, max(1, images.size(0)))
    save_image(images, path, nrow=nrow)


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    set_seed(cfg["seed"])
    device = resolve_device(cfg.get("device", "cpu"))

    if cfg["split"]["mode"] != "direct":
        raise NotImplementedError("run_inspect_packets currently supports direct split only.")

    base_run_name = build_base_run_name(cfg)
    packet_source = cfg.get("packet", {}).get("source", "raw")
    output_root = Path(cfg["output"]["root"])
    ckpt_dir = output_root / "checkpoints" / "local_pretrain" / base_run_name
    packet_dir = output_root / "packets" / base_run_name / packet_source
    report_dir = output_root / "reports" / "packet_inspect"
    image_dir = report_dir / "packet_images"
    report_dir.mkdir(parents=True, exist_ok=True)

    selected_agent_ids = parse_agent_ids(args.agent_ids, cfg["split"]["num_agents"])

    print("=== run_inspect_packets ===")
    print(f"config: {args.config}")
    print(f"device: {device}")
    print(f"packet_source: {packet_source}")
    print(f"ckpt_dir: {ckpt_dir}")
    print(f"packet_dir: {packet_dir}")
    print(f"selected_agent_ids: {selected_agent_ids}")

    report = {
        "config": args.config,
        "dataset": cfg["dataset"]["name"],
        "split_mode": cfg["split"]["mode"],
        "model": cfg["model"]["name"],
        "base_run_name": base_run_name,
        "packet_source": packet_source,
        "packet_dir": str(packet_dir),
        "agents": [],
    }

    for agent_id in selected_agent_ids:
        ckpt_path = ckpt_dir / f"agent_{agent_id}_anchor.pt"
        packet_path = packet_dir / f"agent_{agent_id}_packet.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"anchor checkpoint not found: {ckpt_path}")
        if not packet_path.exists():
            raise FileNotFoundError(f"packet not found: {packet_path}")

        model = build_model(cfg, device)
        ckpt = load_torch_object(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        packet = load_torch_object(packet_path, map_location="cpu")
        metrics = inspect_packet(model, packet, device)
        grid_path = image_dir / f"{packet_source}_agent_{agent_id}.png"
        save_packet_grid(packet, grid_path, args.max_images)

        row = {
            "agent_id": agent_id,
            "sender_id": packet.sender_id,
            "class_ids": packet.class_ids.tolist(),
            "packet_source": packet_source,
            "packet_path": str(packet_path),
            "image_grid_path": str(grid_path),
            "packet_meta": packet.meta,
            **metrics,
        }
        report["agents"].append(row)

        print(
            f"agent_{agent_id} classes={row['class_ids']} "
            f"anchor_acc={row['anchor_packet_accuracy']:.4f} "
            f"hard_conf={row['anchor_hard_label_confidence']:.4f} "
            f"soft_entropy={row['soft_target_entropy']:.4f} "
            f"grid={grid_path}"
        )

    if report["agents"]:
        report["summary"] = {
            "average_anchor_packet_accuracy": sum(row["anchor_packet_accuracy"] for row in report["agents"]) / len(report["agents"]),
            "average_anchor_hard_label_confidence": sum(row["anchor_hard_label_confidence"] for row in report["agents"]) / len(report["agents"]),
            "average_soft_target_entropy": sum(row["soft_target_entropy"] for row in report["agents"]) / len(report["agents"]),
            "average_soft_target_max_probability": sum(row["soft_target_max_probability"] for row in report["agents"]) / len(report["agents"]),
        }

    suffix = "all" if args.agent_ids == "all" else args.agent_ids.replace(",", "_").replace("-", "to")
    json_path = report_dir / f"packet_inspection_{base_run_name}_{packet_source}_{suffix}.json"
    md_path = report_dir / f"packet_inspection_{base_run_name}_{packet_source}_{suffix}.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    write_markdown(report, md_path)

    print("=== summary ===")
    for key, value in report.get("summary", {}).items():
        print(f"{key}: {value:.4f}")
    print(f"saved_json: {json_path}")
    print(f"saved_markdown: {md_path}")


if __name__ == "__main__":
    main()
