"""
Build social packets from local anchor models.

Packet images can currently come from:
- raw: real local samples, useful as an engineering baseline
- dsdm: DSDM-style prototype/semantic/history-smoothed distilled images
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from src.datasets.cifar import build_cifar_train_dataset, make_direct_class_splits
from src.distill.simple_distiller import build_raw_images, distill_images_with_dsdm
from src.main.run_eval import build_model
from src.main.run_local_pretrain import resolve_device
from src.packet.packet_dataclass import SocialPacket
from src.utils.agent_selection import parse_agent_ids
from src.utils.config import load_yaml
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="实验配置文件路径")
    parser.add_argument(
        "--agent-ids",
        type=str,
        default="all",
        help='要构建 packet 的 sender agent，例如 "all"、"0"、"0,2,4" 或 "0-3"',
    )
    return parser.parse_args()


@torch.no_grad()
def build_soft_targets(model, images: torch.Tensor, temperature: float, device: torch.device):
    model.eval()
    logits = model(images.to(device))
    return F.softmax(logits / temperature, dim=1).cpu()


def build_packet_images(model, train_dataset, class_ids, packet_cfg, device: torch.device):
    ipc = packet_cfg.get("ipc", 10)
    source = packet_cfg.get("source", "raw")
    meta = {"packet_source": source}

    if source == "raw":
        images, hard_labels = build_raw_images(train_dataset, class_ids, ipc)
        return images, hard_labels, meta

    if source == "dsdm":
        images, hard_labels, distill_meta = distill_images_with_dsdm(
            anchor_model=model,
            train_dataset=train_dataset,
            class_ids=class_ids,
            packet_cfg=packet_cfg,
            device=device,
        )
        meta.update(distill_meta)
        return images, hard_labels, meta

    raise ValueError(f"unknown packet.source: {source}")


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    set_seed(cfg["seed"])
    device = resolve_device(cfg.get("device", "cpu"))

    if cfg["split"]["mode"] != "direct":
        raise NotImplementedError("当前 run_build_packets 先支持 direct split。")

    packet_cfg = cfg.get("packet", {})
    ipc = packet_cfg.get("ipc", 10)
    temperature = packet_cfg.get("temperature", 2.0)
    source = packet_cfg.get("source", "raw")

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
    packet_dir = Path(cfg["output"]["root"]) / "packets" / run_name / source
    packet_dir.mkdir(parents=True, exist_ok=True)

    print("=== run_build_packets ===")
    print(f"dataset: {cfg['dataset']['name']}")
    print(f"device: {device}")
    print(f"ipc: {ipc}")
    print(f"temperature: {temperature}")
    print(f"packet_source: {source}")
    print(f"packet_dir: {packet_dir}")

    selected_agent_ids = parse_agent_ids(args.agent_ids, cfg["split"]["num_agents"])
    print(f"selected_agent_ids: {selected_agent_ids}")

    for agent_id in selected_agent_ids:
        class_ids = class_splits[agent_id]
        ckpt_path = ckpt_dir / f"agent_{agent_id}_anchor.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")

        model = build_model(cfg, device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        images, hard_labels, image_meta = build_packet_images(
            model=model,
            train_dataset=train_dataset,
            class_ids=class_ids,
            packet_cfg=packet_cfg,
            device=device,
        )
        soft_targets = build_soft_targets(model, images, temperature, device)

        packet = SocialPacket(
            sender_id=f"agent_{agent_id}",
            class_ids=torch.tensor(class_ids, dtype=torch.long),
            images=images.cpu(),
            hard_labels=hard_labels.cpu(),
            soft_targets=soft_targets,
            meta={
                "packet_type": f"{source}_x_q",
                "ipc": ipc,
                "temperature": temperature,
                "sender_backbone": cfg["model"]["name"],
                "dataset": cfg["dataset"]["name"],
                **image_meta,
            },
        )

        packet_path = packet_dir / f"agent_{agent_id}_packet.pt"
        torch.save(packet, packet_path)
        print(
            f"saved: {packet_path} "
            f"images={tuple(packet.images.shape)} soft_targets={tuple(packet.soft_targets.shape)}"
        )


if __name__ == "__main__":
    main()
