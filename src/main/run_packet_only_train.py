"""
Fine-tune receiver agents only with cached social packets.

This packet-only path keeps the sender/packet pipeline unchanged but changes
the receiver update after communication:
- initialize each receiver from its local anchor
- load every agent packet, including the receiver's own packet
- train only on distilled packet images

The objective is:
L = CE(packet_logits, hard_labels) + lambda_kd * KD(packet_logits, soft_targets)
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.cifar import make_direct_class_splits
from src.main.run_eval import build_model
from src.main.run_local_pretrain import resolve_device
from src.main.run_social_train import compute_packet_kd_loss, packets_to_dataset
from src.utils.agent_selection import parse_agent_ids
from src.utils.config import load_yaml
from src.utils.run_name import build_base_run_name, build_social_run_name
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="experiment config path")
    parser.add_argument(
        "--agent-ids",
        type=str,
        default="all",
        help='receiver agents to train, e.g. "all", "0", "0,2,4", or "0-3"',
    )
    return parser.parse_args()


def load_torch_object(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_all_packets(packet_dir: Path, num_agents: int):
    packets = []
    for sender_id in range(num_agents):
        packet_path = packet_dir / f"agent_{sender_id}_packet.pt"
        if not packet_path.exists():
            raise FileNotFoundError(f"packet not found: {packet_path}")
        packets.append(load_torch_object(packet_path, map_location="cpu"))
    return packets


def train_receiver_packet_only(
    receiver_id: int,
    class_ids,
    cfg,
    ckpt_dir: Path,
    packet_dir: Path,
    save_dir: Path,
    device: torch.device,
):
    anchor_path = ckpt_dir / f"agent_{receiver_id}_anchor.pt"
    if not anchor_path.exists():
        raise FileNotFoundError(f"anchor checkpoint not found: {anchor_path}")

    model = build_model(cfg, device)
    anchor_ckpt = load_torch_object(anchor_path, map_location=device)
    model.load_state_dict(anchor_ckpt["model_state_dict"])

    packets = load_all_packets(
        packet_dir=packet_dir,
        num_agents=cfg["split"]["num_agents"],
    )
    packet_dataset = packets_to_dataset(packets)
    packet_loader = DataLoader(
        packet_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    packet_only_cfg = cfg.get("packet_only", cfg.get("social", {}))
    epochs = packet_only_cfg.get("epochs", cfg.get("social", {}).get("epochs", cfg["train"]["epochs"]))
    lr = packet_only_cfg.get("lr", cfg.get("social", {}).get("lr", cfg["train"]["lr"]))
    lambda_kd = packet_only_cfg.get("lambda_kd", cfg.get("social", {}).get("lambda_kd", 1.0))
    packet_kd_mode = packet_only_cfg.get(
        "packet_kd_mode",
        cfg.get("social", {}).get("packet_kd_mode", "sender_subset"),
    )
    if packet_kd_mode not in ["none", "full", "sender_subset"]:
        raise ValueError(f"unknown packet_kd_mode: {packet_kd_mode}")
    temperature = cfg.get("packet", {}).get("temperature", 2.0)

    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    print(f"\n=== packet-only train agent_{receiver_id} ===")
    print(f"class_ids: {class_ids}")
    print(f"packet_samples: {len(packet_dataset)}")
    print(f"includes_self_packet: True")
    print(f"epochs: {epochs}")
    print(f"packet_kd_mode: {packet_kd_mode}")
    print(f"lambda_kd: {lambda_kd}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_packet_acc = 0
        total_seen = 0

        progress = tqdm(packet_loader, desc=f"agent_{receiver_id} packet-only epoch {epoch + 1}/{epochs}")
        for packet_images, packet_labels, packet_soft_targets, packet_class_ids in progress:
            packet_images = packet_images.to(device, non_blocking=True)
            packet_labels = packet_labels.to(device, non_blocking=True)
            packet_soft_targets = packet_soft_targets.to(device, non_blocking=True)
            packet_class_ids = packet_class_ids.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            packet_logits = model(packet_images)
            loss_packet_ce = ce_loss(packet_logits, packet_labels)
            loss_packet_kd = compute_packet_kd_loss(
                packet_kd_mode,
                packet_logits,
                packet_soft_targets,
                packet_class_ids,
                temperature,
            )
            loss = loss_packet_ce + lambda_kd * loss_packet_kd

            loss.backward()
            optimizer.step()

            batch_size = packet_labels.size(0)
            total_loss += loss.item() * batch_size
            total_packet_acc += (packet_logits.argmax(dim=1) == packet_labels).sum().item()
            total_seen += batch_size

            progress.set_postfix(
                loss=f"{total_loss / total_seen:.4f}",
                packet_acc=f"{total_packet_acc / total_seen:.4f}",
            )

    save_path = save_dir / f"agent_{receiver_id}_packet_only.pt"
    torch.save(
        {
            "agent_id": receiver_id,
            "class_ids": class_ids,
            "model_state_dict": model.state_dict(),
            "cfg": cfg,
            "train_stage": "packet_only_train",
        },
        save_path,
    )
    print(f"saved: {save_path}")


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    set_seed(cfg["seed"])
    device = resolve_device(cfg.get("device", "cpu"))

    if cfg["split"]["mode"] != "direct":
        raise NotImplementedError("run_packet_only_train currently supports direct split only.")
    if cfg.get("social", {}).get("policy", "all_to_all") != "all_to_all":
        raise NotImplementedError("run_packet_only_train currently supports all_to_all only.")

    class_splits = make_direct_class_splits(
        num_classes=cfg["dataset"]["num_classes"],
        num_agents=cfg["split"]["num_agents"],
        classes_per_agent=cfg["split"]["classes_per_agent"],
    )

    base_run_name = build_base_run_name(cfg)
    social_run_name = build_social_run_name(cfg)
    ckpt_dir = Path(cfg["output"]["root"]) / "checkpoints" / "local_pretrain" / base_run_name
    packet_source = cfg.get("packet", {}).get("source", "raw")
    packet_dir = Path(cfg["output"]["root"]) / "packets" / base_run_name / packet_source
    save_dir = Path(cfg["output"]["root"]) / "checkpoints" / "packet_only_train" / social_run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=== run_packet_only_train ===")
    print(f"dataset: {cfg['dataset']['name']}")
    print(f"device: {device}")
    print(f"ckpt_dir: {ckpt_dir}")
    print(f"packet_dir: {packet_dir}")
    print(f"packet_only_run_name: {social_run_name}")
    print(f"save_dir: {save_dir}")

    selected_agent_ids = parse_agent_ids(args.agent_ids, cfg["split"]["num_agents"])
    print(f"selected_agent_ids: {selected_agent_ids}")

    for receiver_id in selected_agent_ids:
        class_ids = class_splits[receiver_id]
        train_receiver_packet_only(
            receiver_id=receiver_id,
            class_ids=class_ids,
            cfg=cfg,
            ckpt_dir=ckpt_dir,
            packet_dir=packet_dir,
            save_dir=save_dir,
            device=device,
        )


if __name__ == "__main__":
    main()
