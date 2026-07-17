from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved FedRE client models on social CIFAR-100 metrics.")
    parser.add_argument("--metadata", required=True, help="split_metadata.json")
    parser.add_argument("--fedre-system", default="external_baselines/repos/FedRE/HtFLlib/system")
    parser.add_argument("--model-dir", required=True, help="Directory containing Client_i_model.pt files")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output", default="external_baselines/outputs/fedre_social/social_metrics.csv")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fedre_system = Path(args.fedre_system).resolve()
    if str(fedre_system) not in sys.path:
        sys.path.insert(0, str(fedre_system))

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    class_split = {int(k): [int(c) for c in v["classes"]] for k, v in metadata["agents"].items()}
    dataset = datasets.CIFAR100(args.data_dir, train=False, transform=transforms.ToTensor(), download=False)
    targets = np.asarray(dataset.targets)
    all_indices = np.arange(len(targets))

    rows = []
    for client_id in range(int(metadata["num_agents"])):
        model_path = Path(args.model_dir) / f"Client_{client_id}_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing FedRE client model: {model_path}")
        model = torch.load(model_path, map_location=device).to(device).eval()
        expert_classes = class_split[client_id]
        expert_mask = np.isin(targets, expert_classes)
        expert_indices = all_indices[expert_mask]
        new_indices = all_indices[~expert_mask]
        rows.append(
            {
                "receiver": client_id,
                "method": "FEDRE",
                "acc_global": _eval_accuracy(model, dataset, all_indices, device, args.batch_size),
                "acc_expert": _eval_accuracy(model, dataset, expert_indices, device, args.batch_size),
                "acc_new": _eval_accuracy(model, dataset, new_indices, device, args.batch_size),
                "expert_classes": f"{min(expert_classes)}-{max(expert_classes)}",
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[fedre-eval] wrote {output_path}")


def _eval_accuracy(model, dataset, indices, device, batch_size: int) -> float:
    """在指定 CIFAR-100 测试子集上计算 top-1 accuracy。"""
    loader = DataLoader(Subset(dataset, [int(i) for i in indices]), batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device).float()
            labels = labels.to(device).long()
            logits = model(images)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += int(labels.numel())
    return correct / max(total, 1)


if __name__ == "__main__":
    main()
