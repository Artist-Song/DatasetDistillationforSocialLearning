"""
Build class-wise raw packets from the generalist teacher.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from src.datasets.cifar import build_cifar_train_dataset
from src.distill.simple_distiller import build_raw_images, distill_images_with_dsdm
from src.main.run_eval import build_model
from src.main.run_local_pretrain import resolve_device
from src.packet.packet_dataclass import SocialPacket
from src.utils.config import load_yaml
from src.utils.experiment import (
    get_experiment_id,
    get_experiment_metadata,
    get_experiment_root,
    get_reuse_cfg,
    get_stage_expected_experiment_id,
    get_stage_read_root,
    require_experiment_id,
    require_packet_dir,
    save_experiment_files,
    validate_reuse,
)
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


@torch.no_grad()
def build_soft_targets(model, images, temperature, device):
    model.eval()
    logits = model(images.to(device))
    return F.softmax(logits / temperature, dim=1).cpu()


def resolve_packet_source(source: str) -> str:
    if source == "global_raw":
        return "global_raw_packet"
    if source == "global_raw_packet":
        return source
    if source == "global_dsdm_packet":
        return source
    raise NotImplementedError(
        "generalist packet version only supports packet.source=global_raw_packet, "
        "global_dsdm_packet, or backward-compatible alias global_raw"
    )


def build_packet_images(model, train_dataset, class_id: int, packet_cfg, canonical_source: str, device):
    ipc = packet_cfg.get("ipc", 10)
    if canonical_source == "global_raw_packet":
        images, hard_labels = build_raw_images(train_dataset, [class_id], ipc)
        return images, hard_labels, {"packet_type": "global_raw_x_q"}
    if canonical_source == "global_dsdm_packet":
        images, hard_labels, distill_meta = distill_images_with_dsdm(
            anchor_model=model,
            train_dataset=train_dataset,
            class_ids=[class_id],
            packet_cfg=packet_cfg,
            device=device,
        )
        meta = {"packet_type": "global_dsdm_x_q"}
        meta.update(distill_meta)
        return images, hard_labels, meta
    raise ValueError(f"unsupported packet source: {canonical_source}")


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    validate_reuse(cfg, args.config)
    experiment_id = get_experiment_id(cfg, args.config)
    experiment_root = get_experiment_root(cfg, args.config)
    experiment = get_experiment_metadata(cfg, args.config)
    reuse = get_reuse_cfg(cfg)
    set_seed(cfg["seed"])
    device = resolve_device(cfg.get("device", "cpu"))

    packet_cfg = cfg.get("packet", {})
    source = packet_cfg.get("source", "global_raw_packet")
    canonical_source = resolve_packet_source(source)
    ipc = packet_cfg.get("ipc", 10)
    temperature = packet_cfg.get("temperature", 2.0)

    packet_read_root = get_stage_read_root(cfg, "packets", args.config)
    source_packet_dir = packet_read_root / "packets" / "generalist" / canonical_source
    if reuse["packets"]:
        require_packet_dir(source_packet_dir, cfg["dataset"]["num_classes"])
        print("=== run_build_generalist_packets ===")
        print(f"experiment_id: {experiment_id}")
        print(f"reuse.packets=true; skipping packet build")
        print(f"source_packet_dir: {source_packet_dir}")
        save_experiment_files(
            cfg,
            args.config,
            {"packet_read_dir": str(source_packet_dir), "packet_write_dir": None},
        )
        return

    generalist_read_root = get_stage_read_root(cfg, "generalist", args.config)
    generalist_expected_experiment_id = get_stage_expected_experiment_id(cfg, "generalist", args.config)
    ckpt_path = generalist_read_root / "checkpoints" / "generalist" / "generalist.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"generalist checkpoint not found: {ckpt_path}")
    model = build_model(cfg, device)
    ckpt = torch.load(ckpt_path, map_location=device)
    require_experiment_id(ckpt.get("experiment_id"), generalist_expected_experiment_id, ckpt_path, cfg, args.config)
    model.load_state_dict(ckpt["model_state_dict"])

    train_dataset = build_cifar_train_dataset(
        name=cfg["dataset"]["name"],
        root=cfg["dataset"]["root"],
        image_size=tuple(cfg["dataset"]["image_size"]),
        download=True,
    )
    packet_dir = experiment_root / "packets" / "generalist" / canonical_source
    packet_dir.mkdir(parents=True, exist_ok=True)
    save_experiment_files(
        cfg,
        args.config,
        {
            "generalist_checkpoint_path": str(ckpt_path),
            "packet_write_dir": str(packet_dir),
        },
    )

    print("=== run_build_generalist_packets ===")
    print(f"experiment_id: {experiment_id}")
    print(f"experiment_root: {experiment_root}")
    print(f"generalist_read_root: {generalist_read_root}")
    print(f"source: {source}")
    print(f"canonical_source: {canonical_source}")
    print(f"ipc: {ipc}")
    print(f"packet_dir: {packet_dir}")

    for class_id in range(cfg["dataset"]["num_classes"]):
        images, hard_labels, packet_meta = build_packet_images(model, train_dataset, class_id, packet_cfg, canonical_source, device)
        soft_targets = build_soft_targets(model, images, temperature, device)
        packet = SocialPacket(
            sender_id="generalist",
            class_ids=torch.tensor([class_id], dtype=torch.long),
            images=images.cpu(),
            hard_labels=hard_labels.cpu(),
            soft_targets=soft_targets,
            meta={
                **packet_meta,
                "packet_source": canonical_source,
                "packet_source_alias": source,
                "ipc": ipc,
                "temperature": temperature,
                "dataset": cfg["dataset"]["name"],
                "experiment_id": experiment_id,
                "experiment": experiment,
            },
        )
        packet_path = packet_dir / f"class_{class_id}_packet.pt"
        torch.save(packet, packet_path)
        print(f"saved: {packet_path} images={tuple(packet.images.shape)}")


if __name__ == "__main__":
    main()
