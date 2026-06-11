"""Evaluate v2 expert or socialized agents."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from src.datasets.cifar import build_cifar_test_dataset, make_direct_class_splits, subset_by_classes
from src.models.agent_model import build_agent_model
from src.training.v2_train_utils import SyntheticCIFARDataset, get_new_classes
from src.utils.agent_selection import parse_agent_ids
from src.utils.config import load_yaml
from src.utils.seed import set_seed
from src.utils.v2_progress import StageTimer, progress
from src.utils.v2_paths import get_v2_agent_checkpoint_dir, get_v2_metrics_dir, get_v2_socialized_checkpoint_dir
from src.utils.v2_runtime import resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate v2 expert or socialized checkpoints.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint-stage", type=str, choices=["expert", "socialized"], required=True)
    parser.add_argument("--packet-source", type=str, choices=["raw", "strict_dsdm"], default=None)
    parser.add_argument(
        "--adaptation-mode",
        type=str,
        choices=["last_block_anchor", "full_finetune"],
        default="last_block_anchor",
        help="Which socialized checkpoint layout to evaluate.",
    )
    parser.add_argument("--agent-ids", type=str, default="all")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-download", action="store_true", help="Disable CIFAR download.")
    parser.add_argument("--dataset-root", type=str, default=None, help="Override cfg.dataset.root.")
    parser.add_argument("--smoke-synthetic-samples", type=int, default=None)
    return parser.parse_args()


def torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_dataset(cfg: dict, args):
    dataset_cfg = cfg["dataset"]
    image_size = tuple(dataset_cfg.get("image_size", (32, 32)))
    if args.smoke_synthetic_samples is not None:
        return SyntheticCIFARDataset(args.smoke_synthetic_samples, dataset_cfg["num_classes"], image_size)
    try:
        return build_cifar_test_dataset(
            name=dataset_cfg["name"],
            root=dataset_cfg["root"],
            image_size=image_size,
            download=not args.no_download,
        )
    except RuntimeError:
        if not args.no_download:
            raise
        fallback_samples = max(dataset_cfg["num_classes"] * 50, 500)
        print(
            "WARNING: CIFAR test split not found and --no-download was set; "
            f"using {fallback_samples} synthetic samples for smoke testing only."
        )
        return SyntheticCIFARDataset(fallback_samples, dataset_cfg["num_classes"], image_size)


def checkpoint_path_for(
    cfg: dict,
    stage: str,
    agent_id: int,
    packet_source: str = None,
    adaptation_mode: str = "last_block_anchor",
) -> Path:
    if stage == "expert":
        return get_v2_agent_checkpoint_dir(cfg) / f"agent_{agent_id}_expert.pt"
    if stage == "socialized":
        if packet_source is None:
            raise ValueError("packet_source is required for socialized checkpoints")
        return get_v2_socialized_checkpoint_dir(cfg, packet_source, adaptation_mode) / f"agent_{agent_id}_socialized.pt"
    raise ValueError(f"unknown checkpoint stage: {stage}")


def load_model(cfg: dict, stage: str, agent_id: int, packet_source: str, adaptation_mode: str, device: torch.device):
    ckpt_path = checkpoint_path_for(cfg, stage, agent_id, packet_source, adaptation_mode)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{stage} checkpoint missing: {ckpt_path}")
    ckpt = torch_load(ckpt_path, map_location=device)
    model = build_agent_model(cfg, agent_id, device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt, ckpt_path


@torch.no_grad()
def evaluate_accuracy(model, dataset, batch_size: int, device: torch.device, num_workers: int) -> float:
    if len(dataset) == 0:
        return 0.0
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return correct / total if total else 0.0


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def metrics_output_path(
    cfg: dict,
    stage: str,
    packet_source: str = None,
    adaptation_mode: str = "last_block_anchor",
) -> Path:
    if stage == "expert":
        return get_v2_metrics_dir(cfg) / "local_only" / "metrics_v2.json"
    variant_name = f"{packet_source}_ipc"
    if adaptation_mode != "last_block_anchor":
        variant_name = f"{variant_name}_{adaptation_mode}"
    return get_v2_metrics_dir(cfg) / variant_name / "metrics_v2.json"


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    packet_source = args.packet_source or cfg.get("packet", {}).get("source", "strict_dsdm")
    if args.checkpoint_stage == "socialized" and packet_source not in {"raw", "strict_dsdm"}:
        raise ValueError(f"packet_source must be raw or strict_dsdm, got {packet_source}")

    set_seed(cfg["seed"])
    device = resolve_device(cfg.get("device", "cpu"))
    split_cfg = cfg["split"]
    dataset_cfg = cfg["dataset"]
    if args.dataset_root is not None:
        dataset_cfg["root"] = args.dataset_root
    if split_cfg["mode"] != "direct":
        raise NotImplementedError("run_eval_v2 supports split.mode=direct only")

    selected_agent_ids = parse_agent_ids(args.agent_ids, split_cfg["num_agents"])
    class_splits = make_direct_class_splits(
        num_classes=dataset_cfg["num_classes"],
        num_agents=split_cfg["num_agents"],
        classes_per_agent=split_cfg["classes_per_agent"],
    )
    test_dataset = build_dataset(cfg, args)
    batch_size = args.batch_size or cfg["train"]["batch_size"]

    print("=== run_eval_v2 ===")
    print(f"config: {args.config}")
    print(f"experiment: {cfg['experiment']['name']}")
    print(f"checkpoint_stage: {args.checkpoint_stage}")
    print(f"packet_source: {packet_source}")
    print(f"adaptation_mode: {args.adaptation_mode}")
    print(f"device: {device}")
    print(f"selected_agent_ids: {selected_agent_ids}")

    agents: Dict[str, dict] = {}
    with StageTimer("run_eval_v2 total"):
        for agent_id in progress(selected_agent_ids, desc=f"eval {args.checkpoint_stage} agents"):
            with StageTimer(f"eval agent_{agent_id}"):
                expert_classes = list(class_splits[agent_id])
                new_classes = get_new_classes(dataset_cfg["num_classes"], expert_classes)
                model, ckpt, ckpt_path = load_model(
                    cfg,
                    args.checkpoint_stage,
                    agent_id,
                    packet_source,
                    args.adaptation_mode,
                    device,
                )
                expert_dataset = subset_by_classes(test_dataset, expert_classes)
                new_dataset = subset_by_classes(test_dataset, new_classes)

                expert_acc = evaluate_accuracy(model, expert_dataset, batch_size, device, args.num_workers)
                new_acc = evaluate_accuracy(model, new_dataset, batch_size, device, args.num_workers)
                overall_acc = evaluate_accuracy(model, test_dataset, batch_size, device, args.num_workers)
                agents[str(agent_id)] = {
                    "agent_id": agent_id,
                    "checkpoint_path": str(ckpt_path),
                    "checkpoint_stage": args.checkpoint_stage,
                    "packet_source": packet_source if args.checkpoint_stage == "socialized" else None,
                    "adaptation_mode": args.adaptation_mode if args.checkpoint_stage == "socialized" else None,
                    "expert_classes": expert_classes,
                    "new_classes": new_classes,
                    "expert_accuracy": expert_acc,
                    "new_accuracy": new_acc,
                    "overall_accuracy": overall_acc,
                    "stage_in_checkpoint": ckpt.get("stage"),
                }
                print(
                    f"agent_{agent_id}: expert={expert_acc:.4f} "
                    f"new={new_acc:.4f} overall={overall_acc:.4f}"
                )

    summary = {
        "average_expert_accuracy": mean([item["expert_accuracy"] for item in agents.values()]),
        "average_new_accuracy": mean([item["new_accuracy"] for item in agents.values()]),
        "average_overall_accuracy": mean([item["overall_accuracy"] for item in agents.values()]),
    }
    metrics = {
        "experiment": cfg["experiment"]["name"],
        "checkpoint_stage": args.checkpoint_stage,
        "packet_source": packet_source if args.checkpoint_stage == "socialized" else None,
        "adaptation_mode": args.adaptation_mode if args.checkpoint_stage == "socialized" else None,
        "agent_ids": selected_agent_ids,
        "agents": agents,
        "summary": summary,
    }
    out_path = metrics_output_path(cfg, args.checkpoint_stage, packet_source, args.adaptation_mode)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved metrics: {out_path}")


if __name__ == "__main__":
    main()
