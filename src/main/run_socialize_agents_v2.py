"""Socialize v2 agents with own expert data and other agents' packets."""

import argparse
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from src.datasets.cifar import build_cifar_train_dataset, make_direct_class_splits, subset_by_classes
from src.models.agent_model import build_agent_model
from src.packet.packet_dataclass import SocialPacket
from src.training.v2_train_utils import SyntheticCIFARDataset, get_new_classes
from src.utils.agent_selection import parse_agent_ids
from src.utils.config import load_yaml
from src.utils.seed import set_seed
from src.utils.v2_progress import StageTimer, progress
from src.utils.v2_paths import (
    get_v2_agent_checkpoint_dir,
    get_v2_packet_dir,
    get_v2_socialized_checkpoint_dir,
)
from src.utils.v2_runtime import resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description="Socialize v2 agents using hard-label image packets.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--packet-source", type=str, choices=["raw", "strict_dsdm"], default=None)
    parser.add_argument("--agent-ids", type=str, default="all")
    parser.add_argument("--max-epochs-a", type=int, default=None, help="Optional Phase A epoch cap.")
    parser.add_argument("--max-epochs-b", type=int, default=None, help="Optional Phase B epoch cap.")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional batch cap per epoch.")
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


def get_head_module(model: nn.Module) -> nn.Module:
    backbone = model.get_backbone()
    for attr_name in ["classifier", "fc", "head"]:
        if hasattr(backbone, attr_name):
            return getattr(backbone, attr_name)
    raise RuntimeError("backbone classifier head not found")


def freeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def set_module_trainable(module: nn.Module, trainable: bool) -> None:
    for param in module.parameters():
        param.requires_grad = trainable


def set_phase_a_trainable(model: nn.Module) -> List[str]:
    freeze_all(model)
    head = get_head_module(model)
    set_module_trainable(head, True)
    return [name for name, param in model.named_parameters() if param.requires_grad]


def set_phase_b_trainable(model: nn.Module) -> List[str]:
    freeze_all(model)
    backbone = model.get_backbone()
    head = get_head_module(model)
    set_module_trainable(head, True)

    if hasattr(backbone, "layer4"):
        set_module_trainable(backbone.layer4, True)
    elif hasattr(backbone, "layers") and "conv" in backbone.layers:
        last_idx = len(backbone.layers["conv"]) - 1
        set_module_trainable(backbone.layers["conv"][last_idx], True)
        if "norm" in backbone.layers and len(backbone.layers["norm"]) > last_idx:
            set_module_trainable(backbone.layers["norm"][last_idx], True)
    else:
        print("WARNING: could not identify last block; training head only in Phase B")

    return [name for name, param in model.named_parameters() if param.requires_grad]


def build_dataset(cfg: dict, args):
    dataset_cfg = cfg["dataset"]
    image_size = tuple(dataset_cfg.get("image_size", (32, 32)))
    if args.smoke_synthetic_samples is not None:
        return SyntheticCIFARDataset(args.smoke_synthetic_samples, dataset_cfg["num_classes"], image_size)
    try:
        return build_cifar_train_dataset(
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
            "WARNING: CIFAR train split not found and --no-download was set; "
            f"using {fallback_samples} synthetic samples for smoke testing only."
        )
        return SyntheticCIFARDataset(fallback_samples, dataset_cfg["num_classes"], image_size)


def load_expert_model(cfg: dict, agent_id: int, device: torch.device):
    ckpt_path = get_v2_agent_checkpoint_dir(cfg) / f"agent_{agent_id}_expert.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"expert checkpoint missing: {ckpt_path}")
    ckpt = torch_load(ckpt_path, map_location=device)
    model = build_agent_model(cfg, agent_id, device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt, ckpt_path


def load_other_packets(cfg: dict, receiver_id: int, packet_source: str, num_agents: int) -> Tuple[Optional[TensorDataset], List[dict]]:
    packet_dir = get_v2_packet_dir(cfg, packet_source)
    packets = []
    packet_meta = []
    for sender_id in range(num_agents):
        if sender_id == receiver_id:
            continue
        packet_path = packet_dir / f"agent_{sender_id}_packet.pt"
        if not packet_path.exists():
            continue
        packet: SocialPacket = torch_load(packet_path, map_location="cpu")
        packets.append(packet)
        meta = dict(packet.meta)
        meta.setdefault("sender_id", int(packet.sender_id))
        meta.setdefault("path", str(packet_path))
        packet_meta.append(meta)

    if not packets:
        print(f"WARNING: no other-agent packets found for receiver agent_{receiver_id} under {packet_dir}")
        return None, []

    images = torch.cat([packet.images.float() for packet in packets], dim=0)
    labels = torch.cat([packet.hard_labels.long() for packet in packets], dim=0)
    return TensorDataset(images, labels), packet_meta


def anchor_regularization(model: nn.Module, anchor_state: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    loss = torch.zeros((), device=device)
    for name, param in model.named_parameters():
        if param.requires_grad and name in anchor_state:
            loss = loss + torch.sum((param - anchor_state[name].to(device)) ** 2)
    return loss


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: Optional[int] = None,
    lambda_anchor: float = 0.0,
    anchor_state: Optional[Dict[str, torch.Tensor]] = None,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if lambda_anchor > 0.0 and anchor_state is not None:
            loss = loss + lambda_anchor * anchor_regularization(model, anchor_state, device)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_seen += batch_size

    if total_seen == 0:
        raise RuntimeError("no socialization batches were processed")
    return total_loss / total_seen, total_correct / total_seen


def train_phase(
    phase_name: str,
    model: nn.Module,
    loader: DataLoader,
    trainable_names: Iterable[str],
    lr: float,
    epochs: int,
    device: torch.device,
    max_batches: Optional[int],
    lambda_anchor: float = 0.0,
    anchor_state: Optional[Dict[str, torch.Tensor]] = None,
):
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError(f"{phase_name}: no trainable parameters")
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    print(f"{phase_name} trainable params: {list(trainable_names)}")

    final_loss = None
    final_acc = None
    with StageTimer(phase_name):
        for epoch in progress(range(epochs), desc=f"{phase_name} epochs"):
            final_loss, final_acc = train_epoch(
                model=model,
                loader=loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                max_batches=max_batches,
                lambda_anchor=lambda_anchor,
                anchor_state=anchor_state,
            )
            print(f"{phase_name} epoch {epoch + 1}/{epochs}: loss={final_loss:.4f} acc={final_acc:.4f}")
    return {"epochs": epochs, "final_loss": float(final_loss), "final_acc": float(final_acc)}


def socialize_agent(agent_id: int, cfg: dict, train_dataset, class_splits, packet_source: str, device: torch.device, args):
    split_cfg = cfg["split"]
    social_cfg = cfg["social"]
    expert_classes = list(class_splits[agent_id])
    new_classes = get_new_classes(cfg["dataset"]["num_classes"], expert_classes)
    own_dataset = subset_by_classes(train_dataset, expert_classes)
    packet_dataset, packet_meta = load_other_packets(cfg, agent_id, packet_source, split_cfg["num_agents"])
    datasets = [own_dataset] if packet_dataset is None else [own_dataset, packet_dataset]
    social_dataset = ConcatDataset(datasets)
    loader = DataLoader(
        social_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model, expert_ckpt, expert_ckpt_path = load_expert_model(cfg, agent_id, device)
    anchor_state = {name: param.detach().cpu().clone() for name, param in model.named_parameters()}
    phase_a_epochs = social_cfg["phase_a_epochs"]
    phase_b_epochs = social_cfg["phase_b_epochs"]
    if args.max_epochs_a is not None:
        phase_a_epochs = min(phase_a_epochs, args.max_epochs_a)
    if args.max_epochs_b is not None:
        phase_b_epochs = min(phase_b_epochs, args.max_epochs_b)
    if phase_a_epochs <= 0 or phase_b_epochs <= 0:
        raise ValueError("Phase A and Phase B epochs must both be positive")

    print(f"\n=== socialize agent_{agent_id} ===")
    print(f"expert_checkpoint: {expert_ckpt_path}")
    print(f"expert_classes: {expert_classes}")
    print(f"new_classes: {new_classes}")
    print(f"own_expert_samples: {len(own_dataset)}")
    print(f"other_packet_count: {len(packet_meta)}")
    print(f"social_samples: {len(social_dataset)}")

    phase_a_names = set_phase_a_trainable(model)
    phase_a_summary = train_phase(
        "Phase A",
        model,
        loader,
        phase_a_names,
        social_cfg["lr_head"],
        phase_a_epochs,
        device,
        args.max_batches,
    )

    phase_b_names = set_phase_b_trainable(model)
    phase_b_summary = train_phase(
        "Phase B",
        model,
        loader,
        phase_b_names,
        social_cfg["lr_last_block"],
        phase_b_epochs,
        device,
        args.max_batches,
        lambda_anchor=social_cfg.get("lambda_anchor", 0.0),
        anchor_state=anchor_state,
    )

    save_dir = get_v2_socialized_checkpoint_dir(cfg, packet_source)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"agent_{agent_id}_socialized.pt"
    torch.save(
        {
            "agent_id": agent_id,
            "expert_classes": expert_classes,
            "new_classes": new_classes,
            "model_name": expert_ckpt.get("model_name", model.model_name),
            "model_state_dict": model.state_dict(),
            "cfg": cfg,
            "stage": "socialized",
            "packet_source": packet_source,
            "source_expert_checkpoint": str(expert_ckpt_path),
            "packet_meta": packet_meta,
            "train_summary": {
                "phase_a": phase_a_summary,
                "phase_b": phase_b_summary,
                "lambda_anchor": social_cfg.get("lambda_anchor", 0.0),
            },
        },
        ckpt_path,
    )
    print(f"saved: {ckpt_path}")
    return ckpt_path


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    packet_source = args.packet_source or cfg.get("packet", {}).get("source", "strict_dsdm")
    if packet_source not in {"raw", "strict_dsdm"}:
        raise ValueError(f"packet_source must be raw or strict_dsdm, got {packet_source}")

    set_seed(cfg["seed"])
    device = resolve_device(cfg.get("device", "cpu"))
    split_cfg = cfg["split"]
    dataset_cfg = cfg["dataset"]
    if args.dataset_root is not None:
        dataset_cfg["root"] = args.dataset_root
    if split_cfg["mode"] != "direct":
        raise NotImplementedError("run_socialize_agents_v2 supports split.mode=direct only")

    selected_agent_ids = parse_agent_ids(args.agent_ids, split_cfg["num_agents"])
    class_splits = make_direct_class_splits(
        num_classes=dataset_cfg["num_classes"],
        num_agents=split_cfg["num_agents"],
        classes_per_agent=split_cfg["classes_per_agent"],
    )
    train_dataset = build_dataset(cfg, args)

    print("=== run_socialize_agents_v2 ===")
    print(f"config: {args.config}")
    print(f"experiment: {cfg['experiment']['name']}")
    print(f"packet_source: {packet_source}")
    print(f"device: {device}")
    print(f"dataset_root: {dataset_cfg['root']}")
    print(f"download: {not args.no_download}")
    print(f"selected_agent_ids: {selected_agent_ids}")

    with StageTimer("run_socialize_agents_v2 total"):
        for agent_id in progress(selected_agent_ids, desc="socialize agents"):
            with StageTimer(f"socialize agent_{agent_id}"):
                socialize_agent(agent_id, cfg, train_dataset, class_splits, packet_source, device, args)


if __name__ == "__main__":
    main()
