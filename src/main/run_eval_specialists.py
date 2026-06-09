"""
Evaluate partial-known specialists on known, missing, and full test sets.
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.cifar import build_cifar_test_dataset
from src.datasets.splits import get_partial_split, subset_by_classes
from src.main.run_eval import build_model
from src.main.run_local_pretrain import resolve_device
from src.models.social_head_model import SocialHeadAgent
from src.utils.agent_selection import parse_agent_ids
from src.utils.config import load_yaml
from src.utils.experiment import get_experiment_id, get_experiment_metadata, get_experiment_root
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint-stage", type=str, choices=["local", "social_head"], required=True)
    parser.add_argument("--agent-ids", type=str, default="all")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device: torch.device, head: str = None):
    model.eval()
    total_correct = 0
    total_seen = 0
    for images, labels in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images, head=head) if head is not None else model(images)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_seen += labels.size(0)
    return total_correct / total_seen


def make_loader(dataset, cfg, device):
    return DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )


def load_stage_model(cfg, device, agent_id: int, stage: str, experiment_root: Path, experiment_id: str):
    if stage == "local":
        ckpt_path = experiment_root / "checkpoints" / "specialists" / f"agent_{agent_id}_specialist.pt"
        model = build_model(cfg, device)
        head = None
    else:
        ckpt_path = experiment_root / "checkpoints" / "social_head" / f"agent_{agent_id}_social_head.pt"
        social_cfg = cfg.get("social_head", cfg.get("social", {}))
        model = SocialHeadAgent(cfg, device=device, feature_idx=social_cfg.get("feature_idx"))
        head = "social"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_experiment_id = ckpt.get("experiment_id")
    if ckpt_experiment_id != experiment_id:
        raise RuntimeError(
            f"checkpoint experiment_id mismatch for {ckpt_path}: "
            f"expected {experiment_id}, got {ckpt_experiment_id}"
        )
    model.load_state_dict(ckpt["model_state_dict"])
    return model, head, ckpt_path


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    experiment_id = get_experiment_id(cfg, args.config)
    experiment_root = get_experiment_root(cfg, args.config)
    experiment = get_experiment_metadata(cfg, args.config)
    set_seed(cfg["seed"])
    device = resolve_device(cfg.get("device", "cpu"))

    splits = get_partial_split(cfg["split"]["name"])
    selected_agent_ids = parse_agent_ids(args.agent_ids, len(splits))
    test_dataset = build_cifar_test_dataset(
        name=cfg["dataset"]["name"],
        root=cfg["dataset"]["root"],
        image_size=tuple(cfg["dataset"]["image_size"]),
        download=True,
    )
    full_loader = make_loader(test_dataset, cfg, device)

    print(f"=== run_eval_specialists {args.checkpoint_stage} ===")
    print(f"experiment_id: {experiment_id}")
    print(f"experiment_root: {experiment_root}")
    print(f"device: {device}")
    print(f"selected_agent_ids: {selected_agent_ids}")

    results = {
        "experiment_id": experiment_id,
        "experiment": experiment,
        "checkpoint_stage": args.checkpoint_stage,
        "dataset": cfg["dataset"]["name"],
        "split_name": cfg["split"]["name"],
        "selected_agent_ids": selected_agent_ids,
        "agents": [],
    }
    known_accs = []
    missing_accs = []
    general_accs = []

    for agent_id in selected_agent_ids:
        split = splits[agent_id]
        model, head, ckpt_path = load_stage_model(cfg, device, agent_id, args.checkpoint_stage, experiment_root, experiment_id)
        known_loader = make_loader(subset_by_classes(test_dataset, split.known), cfg, device)
        missing_loader = make_loader(subset_by_classes(test_dataset, split.missing), cfg, device)

        known_acc = evaluate(model, known_loader, device, head=head)
        missing_acc = evaluate(model, missing_loader, device, head=head)
        general_acc = evaluate(model, full_loader, device, head=head)
        known_accs.append(known_acc)
        missing_accs.append(missing_acc)
        general_accs.append(general_acc)

        results["agents"].append(
            {
                "agent_id": agent_id,
                "known_classes": split.known,
                "missing_classes": split.missing,
                "checkpoint": str(ckpt_path),
                "known_accuracy": known_acc,
                "missing_accuracy": missing_acc,
                "general_accuracy": general_acc,
            }
        )
        print(
            f"agent_{agent_id} known_accuracy={known_acc:.4f} "
            f"missing_accuracy={missing_acc:.4f} general_accuracy={general_acc:.4f}"
        )

    results["summary"] = {
        "known_accuracy": sum(known_accs) / len(known_accs),
        "missing_accuracy": sum(missing_accs) / len(missing_accs),
        "general_accuracy": sum(general_accs) / len(general_accs),
    }
    print("=== summary ===")
    for key, value in results["summary"].items():
        print(f"{key}: {value:.4f}")

    report_dir = experiment_root / "reports" / "generalist_packet"
    report_dir.mkdir(parents=True, exist_ok=True)
    agent_suffix = "all" if args.agent_ids == "all" else args.agent_ids.replace(",", "_").replace("-", "to")
    report_path = report_dir / f"eval_{args.checkpoint_stage}_{agent_suffix}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"saved_report: {report_path}")


if __name__ == "__main__":
    main()
