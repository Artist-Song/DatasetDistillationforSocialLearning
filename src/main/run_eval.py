"""
Evaluate locally pretrained anchor models.

The first evaluation path reports:
- expert_accuracy: accuracy on the agent's own classes
- general_accuracy: accuracy on the full test set
- average_accuracy: mean expert/general accuracy over agents
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.cifar import (
    build_cifar_test_dataset,
    make_direct_class_splits,
    subset_by_classes,
)
from src.main.run_local_pretrain import resolve_device
from src.models.agent_model import AgentModel
from src.utils.config import load_yaml
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="实验配置文件路径")
    parser.add_argument(
        "--checkpoint-stage",
        type=str,
        default="local_pretrain",
        choices=["local_pretrain", "social_train"],
        help="要评估的 checkpoint 阶段",
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device: torch.device):
    model.eval()
    total_correct = 0
    total_seen = 0

    for images, labels in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_seen += labels.size(0)

    return total_correct / total_seen


def build_model(cfg, device: torch.device):
    return AgentModel(
        model_name=cfg["model"]["name"],
        dataset=cfg["dataset"]["name"],
        num_classes=cfg["dataset"]["num_classes"],
        image_size=tuple(cfg["dataset"]["image_size"]),
        norm_type=cfg["model"]["norm_type"],
    ).to(device)


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    set_seed(cfg["seed"])
    device = resolve_device(cfg.get("device", "cpu"))

    if cfg["split"]["mode"] != "direct":
        raise NotImplementedError("当前 run_eval 先支持 direct split。")

    test_dataset = build_cifar_test_dataset(
        name=cfg["dataset"]["name"],
        root=cfg["dataset"]["root"],
        image_size=tuple(cfg["dataset"]["image_size"]),
        download=True,
    )
    full_loader = DataLoader(
        test_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    class_splits = make_direct_class_splits(
        num_classes=cfg["dataset"]["num_classes"],
        num_agents=cfg["split"]["num_agents"],
        classes_per_agent=cfg["split"]["classes_per_agent"],
    )
    run_name = f"{cfg['dataset']['name']}_{cfg['split']['mode']}_{cfg['model']['name']}"
    ckpt_dir = Path(cfg["output"]["root"]) / "checkpoints" / args.checkpoint_stage / run_name
    ckpt_suffix = "anchor" if args.checkpoint_stage == "local_pretrain" else "social"

    print(f"=== run_eval {args.checkpoint_stage} ===")
    print(f"dataset: {cfg['dataset']['name']}")
    print(f"device: {device}")
    print(f"ckpt_dir: {ckpt_dir}")

    results = {
        "checkpoint_stage": args.checkpoint_stage,
        "dataset": cfg["dataset"]["name"],
        "split_mode": cfg["split"]["mode"],
        "model": cfg["model"]["name"],
        "agents": [],
    }
    expert_accs = []
    general_accs = []
    for agent_id, class_ids in enumerate(class_splits):
        ckpt_path = ckpt_dir / f"agent_{agent_id}_{ckpt_suffix}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")

        model = build_model(cfg, device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        expert_dataset = subset_by_classes(test_dataset, class_ids)
        expert_loader = DataLoader(
            expert_dataset,
            batch_size=cfg["train"]["batch_size"],
            shuffle=False,
            num_workers=2,
            pin_memory=device.type == "cuda",
        )

        expert_acc = evaluate(model, expert_loader, device)
        general_acc = evaluate(model, full_loader, device)
        expert_accs.append(expert_acc)
        general_accs.append(general_acc)
        results["agents"].append(
            {
                "agent_id": agent_id,
                "class_ids": class_ids,
                "expert_accuracy": expert_acc,
                "general_accuracy": general_acc,
            }
        )

        print(
            f"agent_{agent_id} classes={class_ids} "
            f"expert_accuracy={expert_acc:.4f} general_accuracy={general_acc:.4f}"
        )

    print("=== summary ===")
    average_expert_accuracy = sum(expert_accs) / len(expert_accs)
    average_general_accuracy = sum(general_accs) / len(general_accs)
    results["summary"] = {
        "average_expert_accuracy": average_expert_accuracy,
        "average_general_accuracy": average_general_accuracy,
    }
    print(f"average_expert_accuracy: {average_expert_accuracy:.4f}")
    print(f"average_general_accuracy: {average_general_accuracy:.4f}")

    report_dir = Path(cfg["output"]["root"]) / "reports" / "eval"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{args.checkpoint_stage}_{run_name}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"saved_report: {report_path}")


if __name__ == "__main__":
    main()
