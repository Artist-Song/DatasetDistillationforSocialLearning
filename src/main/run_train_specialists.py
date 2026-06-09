"""
Train specialist agents on their known classes.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.cifar import build_cifar_train_dataset
from src.datasets.splits import get_partial_split, subset_by_classes
from src.main.run_eval import build_model
from src.main.run_local_pretrain import resolve_device
from src.utils.agent_selection import parse_agent_ids
from src.utils.config import load_yaml
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--agent-ids", type=str, default="all")
    return parser.parse_args()


def train_one_agent(agent_id, split, cfg, train_dataset, device, save_dir):
    known_classes = split.known
    missing_classes = split.missing
    subset = subset_by_classes(train_dataset, known_classes)
    loader = DataLoader(
        subset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    model = build_model(cfg, device)
    specialist_cfg = cfg.get("specialist", {})
    epochs = specialist_cfg.get("epochs", cfg["train"]["epochs"])
    lr = specialist_cfg.get("lr", cfg["train"]["lr"])
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"\n=== train specialist agent_{agent_id} ===")
    print(f"known_classes: {known_classes}")
    print(f"samples: {len(subset)}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        progress = tqdm(loader, desc=f"agent_{agent_id} specialist epoch {epoch + 1}/{epochs}")
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_seen += batch_size
            progress.set_postfix(loss=f"{total_loss / total_seen:.4f}", acc=f"{total_correct / total_seen:.4f}")

    save_path = save_dir / f"agent_{agent_id}_specialist.pt"
    torch.save(
        {
            "agent_id": agent_id,
            "known_classes": known_classes,
            "missing_classes": missing_classes,
            "stage": "specialist_local",
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

    split_name = cfg["split"]["name"]
    splits = get_partial_split(split_name)
    selected_agent_ids = parse_agent_ids(args.agent_ids, len(splits))
    train_dataset = build_cifar_train_dataset(
        name=cfg["dataset"]["name"],
        root=cfg["dataset"]["root"],
        image_size=tuple(cfg["dataset"]["image_size"]),
        download=True,
    )
    save_dir = Path(cfg["output"]["root"]) / "checkpoints" / "specialists"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=== run_train_specialists ===")
    print(f"split: {split_name}")
    print(f"device: {device}")
    print(f"selected_agent_ids: {selected_agent_ids}")

    for agent_id in selected_agent_ids:
        train_one_agent(agent_id, splits[agent_id], cfg, train_dataset, device, save_dir)


if __name__ == "__main__":
    main()
