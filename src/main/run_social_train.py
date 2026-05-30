"""
Train receiver agents with cached social packets.

This first social-training path uses all-to-all packets and optimizes:
L = L_local + lambda_packet * (CE(packet labels) + lambda_kd * KD(packet soft targets))
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.datasets.cifar import (
    build_cifar_train_dataset,
    make_direct_class_splits,
    subset_by_classes,
)
from src.main.run_eval import build_model
from src.main.run_local_pretrain import resolve_device
from src.utils.config import load_yaml
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="实验配置文件路径")
    return parser.parse_args()


def make_cycle(loader):
    while True:
        for batch in loader:
            yield batch


def kd_loss(logits: torch.Tensor, soft_targets: torch.Tensor, temperature: float) -> torch.Tensor:
    log_probs = F.log_softmax(logits / temperature, dim=1)
    return F.kl_div(log_probs, soft_targets, reduction="batchmean") * (temperature ** 2)


def load_packets_for_receiver(packet_dir: Path, receiver_id: int, num_agents: int):
    packets = []
    for sender_id in range(num_agents):
        if sender_id == receiver_id:
            continue
        packet_path = packet_dir / f"agent_{sender_id}_packet.pt"
        if not packet_path.exists():
            raise FileNotFoundError(f"packet 不存在: {packet_path}")
        packets.append(torch.load(packet_path, map_location="cpu"))
    return packets


def packets_to_dataset(packets):
    images = torch.cat([packet.images for packet in packets], dim=0)
    hard_labels = torch.cat([packet.hard_labels for packet in packets], dim=0)
    soft_targets = torch.cat([packet.soft_targets for packet in packets], dim=0)
    return TensorDataset(images, hard_labels, soft_targets)


def train_receiver(
    receiver_id: int,
    class_ids,
    cfg,
    train_dataset,
    ckpt_dir: Path,
    packet_dir: Path,
    save_dir: Path,
    device: torch.device,
):
    anchor_path = ckpt_dir / f"agent_{receiver_id}_anchor.pt"
    if not anchor_path.exists():
        raise FileNotFoundError(f"anchor checkpoint 不存在: {anchor_path}")

    model = build_model(cfg, device)
    anchor_ckpt = torch.load(anchor_path, map_location=device)
    model.load_state_dict(anchor_ckpt["model_state_dict"])

    local_subset = subset_by_classes(train_dataset, class_ids)
    local_loader = DataLoader(
        local_subset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    packets = load_packets_for_receiver(
        packet_dir=packet_dir,
        receiver_id=receiver_id,
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
    packet_iter = make_cycle(packet_loader)

    social_cfg = cfg.get("social", {})
    epochs = social_cfg.get("epochs", cfg["train"]["epochs"])
    lr = social_cfg.get("lr", cfg["train"]["lr"])
    lambda_packet = social_cfg.get("lambda_packet", 1.0)
    lambda_kd = social_cfg.get("lambda_kd", 1.0)
    temperature = cfg.get("packet", {}).get("temperature", 2.0)

    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    print(f"\n=== social train agent_{receiver_id} ===")
    print(f"class_ids: {class_ids}")
    print(f"local_samples: {len(local_subset)}")
    print(f"packet_samples: {len(packet_dataset)}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_local_acc = 0
        total_seen = 0

        progress = tqdm(local_loader, desc=f"agent_{receiver_id} social epoch {epoch + 1}/{epochs}")
        for local_images, local_labels in progress:
            packet_images, packet_labels, packet_soft_targets = next(packet_iter)

            local_images = local_images.to(device, non_blocking=True)
            local_labels = local_labels.to(device, non_blocking=True)
            packet_images = packet_images.to(device, non_blocking=True)
            packet_labels = packet_labels.to(device, non_blocking=True)
            packet_soft_targets = packet_soft_targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            local_logits = model(local_images)
            packet_logits = model(packet_images)

            loss_local = ce_loss(local_logits, local_labels)
            loss_packet_ce = ce_loss(packet_logits, packet_labels)
            loss_packet_kd = kd_loss(packet_logits, packet_soft_targets, temperature)
            loss = loss_local + lambda_packet * (loss_packet_ce + lambda_kd * loss_packet_kd)

            loss.backward()
            optimizer.step()

            batch_size = local_labels.size(0)
            total_loss += loss.item() * batch_size
            total_local_acc += (local_logits.argmax(dim=1) == local_labels).sum().item()
            total_seen += batch_size

            progress.set_postfix(
                loss=f"{total_loss / total_seen:.4f}",
                local_acc=f"{total_local_acc / total_seen:.4f}",
            )

    save_path = save_dir / f"agent_{receiver_id}_social.pt"
    torch.save(
        {
            "agent_id": receiver_id,
            "class_ids": class_ids,
            "model_state_dict": model.state_dict(),
            "cfg": cfg,
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
        raise NotImplementedError("当前 run_social_train 先支持 direct split。")
    if cfg.get("social", {}).get("policy", "all_to_all") != "all_to_all":
        raise NotImplementedError("当前 run_social_train 先支持 all_to_all。")

    train_dataset = build_cifar_train_dataset(
        name=cfg["dataset"]["name"],
        root=cfg["dataset"]["root"],
        image_size=tuple(cfg["dataset"]["image_size"]),
        download=True,
    )
    class_splits = make_direct_class_splits(
        num_classes=cfg["dataset"]["num_classes"],
        num_agents=cfg["split"]["num_agents"],
        classes_per_agent=cfg["split"]["classes_per_agent"],
    )

    run_name = f"{cfg['dataset']['name']}_{cfg['split']['mode']}_{cfg['model']['name']}"
    ckpt_dir = Path(cfg["output"]["root"]) / "checkpoints" / "local_pretrain" / run_name
    packet_dir = Path(cfg["output"]["root"]) / "packets" / run_name
    save_dir = Path(cfg["output"]["root"]) / "checkpoints" / "social_train" / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=== run_social_train ===")
    print(f"dataset: {cfg['dataset']['name']}")
    print(f"device: {device}")
    print(f"ckpt_dir: {ckpt_dir}")
    print(f"packet_dir: {packet_dir}")
    print(f"save_dir: {save_dir}")

    for receiver_id, class_ids in enumerate(class_splits):
        train_receiver(
            receiver_id=receiver_id,
            class_ids=class_ids,
            cfg=cfg,
            train_dataset=train_dataset,
            ckpt_dir=ckpt_dir,
            packet_dir=packet_dir,
            save_dir=save_dir,
            device=device,
        )


if __name__ == "__main__":
    main()
