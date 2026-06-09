"""
Train a generalist teacher on all classes.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.cifar import build_cifar_train_dataset
from src.main.run_eval import build_model
from src.main.run_local_pretrain import resolve_device
from src.utils.config import load_yaml
from src.utils.experiment import get_experiment_id, get_experiment_metadata, get_experiment_root
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    experiment_id = get_experiment_id(cfg, args.config)
    experiment_root = get_experiment_root(cfg, args.config)
    experiment = get_experiment_metadata(cfg, args.config)
    set_seed(cfg["seed"])
    device = resolve_device(cfg.get("device", "cpu"))

    train_dataset = build_cifar_train_dataset(
        name=cfg["dataset"]["name"],
        root=cfg["dataset"]["root"],
        image_size=tuple(cfg["dataset"]["image_size"]),
        download=True,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    model = build_model(cfg, device)
    teacher_cfg = cfg.get("generalist", {})
    epochs = teacher_cfg.get("epochs", cfg["train"]["epochs"])
    lr = teacher_cfg.get("lr", cfg["train"]["lr"])
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    print("=== run_train_generalist ===")
    print(f"experiment_id: {experiment_id}")
    print(f"experiment_root: {experiment_root}")
    print(f"dataset: {cfg['dataset']['name']}")
    print(f"device: {device}")
    print(f"epochs: {epochs}")
    print(f"samples: {len(train_dataset)}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        progress = tqdm(loader, desc=f"generalist epoch {epoch + 1}/{epochs}")
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

    save_dir = experiment_root / "checkpoints" / "generalist"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "generalist.pt"
    torch.save(
        {
            "agent_id": "generalist",
            "known_classes": list(range(cfg["dataset"]["num_classes"])),
            "stage": "generalist",
            "experiment_id": experiment_id,
            "experiment": experiment,
            "model_state_dict": model.state_dict(),
            "cfg": cfg,
        },
        save_path,
    )
    print(f"saved: {save_path}")


if __name__ == "__main__":
    main()
