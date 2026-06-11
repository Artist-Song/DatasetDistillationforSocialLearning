"""Build v2 raw IPC or strict DSDM hard-label image packets."""

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from torchvision.utils import save_image

from src.datasets.cifar import build_cifar_train_dataset, make_direct_class_splits
from src.distill.v2_strict_dsdm import build_raw_images, distill_images_with_strict_dsdm, freeze_guide_pool
from src.models.agent_model import build_agent_model
from src.packet.packet_dataclass import SocialPacket
from src.training.v2_train_utils import SyntheticCIFARDataset
from src.utils.agent_selection import parse_agent_ids
from src.utils.config import load_yaml
from src.utils.seed import set_seed
from src.utils.v2_paths import get_v2_dsdm_guide_dir, get_v2_packet_dir, get_v2_packet_visual_dir
from src.utils.v2_progress import StageTimer, progress
from src.utils.v2_runtime import resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description="Build v2 raw IPC or strict DSDM packets.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--packet-source", type=str, choices=["raw", "strict_dsdm"], default=None)
    parser.add_argument("--agent-ids", type=str, default="all")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional strict DSDM distillation step cap.")
    parser.add_argument("--max-agents", type=int, default=None, help="Optional sender agent cap for smoke tests.")
    parser.add_argument("--max-classes", type=int, default=None, help="Optional class cap per sender for smoke tests.")
    parser.add_argument("--max-guides", type=int, default=None, help="Optional guide checkpoint cap for strict DSDM.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-download", action="store_true", help="Disable CIFAR download.")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Override cfg.dataset.root; point this to the parent of cifar-10-batches-py.",
    )
    parser.add_argument(
        "--smoke-synthetic-samples",
        type=int,
        default=None,
        help="Use a tiny synthetic CIFAR-shaped dataset for local smoke tests only.",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip packet files that already exist.")
    parser.add_argument("--no-visuals", action="store_true", help="Do not save packet PNG visualizations.")
    return parser.parse_args()


def torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_guide_models(cfg: dict, agent_id: int, device: torch.device, max_guides: int = None):
    guide_dir = get_v2_dsdm_guide_dir(cfg, agent_id)
    expected_count = cfg["dsdm"]["guide_model_number"]
    if max_guides is not None:
        expected_count = min(expected_count, max_guides)

    guide_models = []
    guide_paths = []
    missing_paths = []
    for guide_id in range(expected_count):
        ckpt_path = guide_dir / f"guide_{guide_id}.pt"
        if not ckpt_path.exists():
            missing_paths.append(ckpt_path)
            continue
        model = build_agent_model(cfg, agent_id, device)
        ckpt = torch_load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        guide_models.append(model)
        guide_paths.append(str(ckpt_path))

    if not guide_models:
        missing_text = "\n".join(str(path) for path in missing_paths[:5])
        raise FileNotFoundError(
            "No DSDM guide checkpoints found. Run Task 2 first, for example: "
            f"python -m src.main.run_pretrain_dsdm_guides --config CONFIG --agent-ids {agent_id}\n"
            f"Looked under: {guide_dir}\nMissing examples:\n{missing_text}"
        )
    if missing_paths:
        print(f"WARNING: using {len(guide_models)} existing guide checkpoints; {len(missing_paths)} missing under {guide_dir}")

    freeze_guide_pool(guide_models)
    return guide_models, guide_paths


def make_packet_meta(packet: SocialPacket, packet_source: str, sender_id: int, class_ids: List[int], ipc: int, extra: dict = None):
    meta = {
        "packet_source": packet_source,
        "sender_id": sender_id,
        "class_ids": list(class_ids),
        "ipc": ipc,
    }
    meta.update(packet.byte_stats())
    if extra:
        meta.update(extra)
    return meta


def save_packet(packet_path: Path, sender_id: int, class_ids: List[int], images: torch.Tensor, hard_labels: torch.Tensor, meta: dict):
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet = SocialPacket(
        sender_id=sender_id,
        class_ids=list(class_ids),
        images=images.cpu(),
        hard_labels=hard_labels.cpu(),
        meta={},
    )
    packet.meta = make_packet_meta(
        packet=packet,
        packet_source=meta["packet_source"],
        sender_id=sender_id,
        class_ids=class_ids,
        ipc=meta["ipc"],
        extra={key: value for key, value in meta.items() if key not in {"packet_source", "sender_id", "class_ids", "ipc"}},
    )
    torch.save(packet, packet_path)
    print(f"saved: {packet_path} images={tuple(packet.images.shape)} labels={tuple(packet.hard_labels.shape)}")
    return packet


def save_packet_visuals(cfg: dict, packet_source: str, packet: SocialPacket):
    visual_dir = get_v2_packet_visual_dir(cfg, packet_source)
    visual_dir.mkdir(parents=True, exist_ok=True)
    images = packet.images.detach().float().clamp(0.0, 1.0)
    grid_path = visual_dir / f"agent_{packet.sender_id}_packet_grid.png"
    save_image(images, grid_path, nrow=max(1, int(packet.meta.get("ipc", 10))))

    class_paths = []
    labels = packet.hard_labels.detach().cpu()
    for class_id in packet.class_ids:
        mask = labels == int(class_id)
        if mask.any():
            class_path = visual_dir / f"agent_{packet.sender_id}_class_{int(class_id)}.png"
            save_image(images[mask], class_path, nrow=max(1, int(packet.meta.get("ipc", 10))))
            class_paths.append(str(class_path))

    packet.meta["visual_grid_path"] = str(grid_path)
    packet.meta["visual_class_paths"] = class_paths
    print(f"saved visuals: {grid_path}")
    return grid_path


def packet_matches_request(packet: SocialPacket, packet_source: str, class_ids: List[int], ipc: int) -> Tuple[bool, str]:
    meta = packet.meta or {}
    if meta.get("packet_source") != packet_source:
        return False, f"packet_source {meta.get('packet_source')} != {packet_source}"
    if list(packet.class_ids) != list(class_ids):
        return False, f"class_ids {list(packet.class_ids)} != {list(class_ids)}"
    if int(meta.get("ipc", -1)) != int(ipc):
        return False, f"ipc {meta.get('ipc')} != {ipc}"
    expected_images = len(class_ids) * ipc
    if int(packet.images.shape[0]) != expected_images:
        return False, f"image_count {int(packet.images.shape[0])} != {expected_images}"
    return True, "matched"


def build_raw_packet(sender_id: int, class_ids: List[int], train_dataset, cfg: dict):
    ipc = cfg["packet"]["ipc"]
    images, hard_labels = build_raw_images(train_dataset, class_ids, ipc)
    meta = {
        "packet_source": "raw",
        "sender_id": sender_id,
        "class_ids": class_ids,
        "ipc": ipc,
    }
    return images, hard_labels, meta


def build_strict_dsdm_packet(sender_id: int, class_ids: List[int], train_dataset, cfg: dict, device: torch.device, args):
    guide_models, guide_paths = load_guide_models(cfg, sender_id, device, max_guides=args.max_guides)
    packet_cfg = dict(cfg.get("packet", {}))
    packet_cfg.update(cfg.get("dsdm", {}))
    packet_cfg["progress_desc"] = f"strict dsdm agent_{sender_id}"
    if args.max_steps is not None:
        packet_cfg["distill_steps"] = min(packet_cfg.get("distill_steps", args.max_steps), args.max_steps)

    images, hard_labels, distill_meta = distill_images_with_strict_dsdm(
        guide_models=guide_models,
        train_dataset=train_dataset,
        class_ids=class_ids,
        packet_cfg=packet_cfg,
        device=device,
    )
    meta = {
        "packet_source": "strict_dsdm",
        "sender_id": sender_id,
        "class_ids": class_ids,
        "ipc": packet_cfg["ipc"],
        "guide_checkpoint_paths": guide_paths,
        **distill_meta,
    }
    return images, hard_labels, meta


def build_sender_packet(sender_id: int, class_ids: List[int], train_dataset, cfg: dict, packet_source: str, device: torch.device, args):
    if args.max_classes is not None:
        class_ids = class_ids[: args.max_classes]
    if not class_ids:
        raise ValueError(f"no classes selected for sender agent {sender_id}")

    packet_dir = get_v2_packet_dir(cfg, packet_source)
    packet_path = packet_dir / f"agent_{sender_id}_packet.pt"
    if args.skip_existing and packet_path.exists():
        packet = torch_load(packet_path, map_location="cpu")
        matches_request, reason = packet_matches_request(packet, packet_source, class_ids, cfg["packet"]["ipc"])
        if matches_request:
            if not args.no_visuals:
                visual_path = get_v2_packet_visual_dir(cfg, packet_source) / f"agent_{sender_id}_packet_grid.png"
                if not visual_path.exists():
                    save_packet_visuals(cfg, packet_source, packet)
                    torch.save(packet, packet_path)
            print(f"skip existing: {packet_path}")
            return None
        print(f"rebuild existing packet: {packet_path} ({reason})")

    print(f"\n=== build {packet_source} packet agent_{sender_id} ===")
    print(f"class_ids: {class_ids}")
    print(f"packet_path: {packet_path}")

    if packet_source == "raw":
        images, hard_labels, meta = build_raw_packet(sender_id, class_ids, train_dataset, cfg)
    elif packet_source == "strict_dsdm":
        images, hard_labels, meta = build_strict_dsdm_packet(sender_id, class_ids, train_dataset, cfg, device, args)
    else:
        raise ValueError(f"unknown packet_source: {packet_source}")

    packet = save_packet(packet_path, sender_id, class_ids, images, hard_labels, meta)
    if not args.no_visuals:
        save_packet_visuals(cfg, packet_source, packet)
        torch.save(packet, packet_path)
    return packet


def build_dataset(cfg: dict, args):
    dataset_cfg = cfg["dataset"]
    image_size = tuple(dataset_cfg.get("image_size", (32, 32)))
    if args.smoke_synthetic_samples is not None:
        return SyntheticCIFARDataset(
            num_samples=args.smoke_synthetic_samples,
            num_classes=dataset_cfg["num_classes"],
            image_size=image_size,
        )
    return build_cifar_train_dataset(
        name=dataset_cfg["name"],
        root=dataset_cfg["root"],
        image_size=image_size,
        download=not args.no_download,
    )


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
        raise NotImplementedError("run_build_packets_v2 supports split.mode=direct only")

    selected_agent_ids = parse_agent_ids(args.agent_ids, split_cfg["num_agents"])
    if args.max_agents is not None:
        selected_agent_ids = selected_agent_ids[: args.max_agents]
    class_splits = make_direct_class_splits(
        num_classes=dataset_cfg["num_classes"],
        num_agents=split_cfg["num_agents"],
        classes_per_agent=split_cfg["classes_per_agent"],
    )
    train_dataset = build_dataset(cfg, args)

    print("=== run_build_packets_v2 ===")
    print(f"config: {args.config}")
    print(f"experiment: {cfg['experiment']['name']}")
    print(f"packet_source: {packet_source}")
    print(f"device: {device}")
    print(f"dataset_root: {dataset_cfg['root']}")
    print(f"download: {not args.no_download}")
    print(f"selected_agent_ids: {selected_agent_ids}")
    print(f"ipc: {cfg['packet']['ipc']}")
    if args.smoke_synthetic_samples is not None:
        print(f"smoke_synthetic_samples: {args.smoke_synthetic_samples}")

    with StageTimer(f"run_build_packets_v2 {packet_source} total"):
        for sender_id in progress(selected_agent_ids, desc=f"{packet_source} packet agents"):
            with StageTimer(f"build {packet_source} packet agent_{sender_id}"):
                build_sender_packet(sender_id, class_splits[sender_id], train_dataset, cfg, packet_source, device, args)


if __name__ == "__main__":
    main()
