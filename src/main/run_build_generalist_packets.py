"""
Build class-wise raw packets from the generalist teacher.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from src.datasets.cifar import build_cifar_train_dataset
from src.distill.simple_distiller import build_raw_images
from src.main.run_eval import build_model
from src.main.run_local_pretrain import resolve_device
from src.packet.packet_dataclass import SocialPacket
from src.utils.config import load_yaml
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


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    device = resolve_device(cfg.get("device", "cpu"))

    packet_cfg = cfg.get("packet", {})
    source = packet_cfg.get("source", "global_raw")
    if source != "global_raw":
        raise NotImplementedError("first generalist packet version only supports packet.source=global_raw")
    ipc = packet_cfg.get("ipc", 10)
    temperature = packet_cfg.get("temperature", 2.0)

    ckpt_path = Path(cfg["output"]["root"]) / "checkpoints" / "generalist" / "generalist.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"generalist checkpoint not found: {ckpt_path}")
    model = build_model(cfg, device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    train_dataset = build_cifar_train_dataset(
        name=cfg["dataset"]["name"],
        root=cfg["dataset"]["root"],
        image_size=tuple(cfg["dataset"]["image_size"]),
        download=True,
    )
    packet_dir = Path(cfg["output"]["root"]) / "packets" / "generalist" / source
    packet_dir.mkdir(parents=True, exist_ok=True)

    print("=== run_build_generalist_packets ===")
    print(f"source: {source}")
    print(f"ipc: {ipc}")
    print(f"packet_dir: {packet_dir}")

    for class_id in range(cfg["dataset"]["num_classes"]):
        images, hard_labels = build_raw_images(train_dataset, [class_id], ipc)
        soft_targets = build_soft_targets(model, images, temperature, device)
        packet = SocialPacket(
            sender_id="generalist",
            class_ids=torch.tensor([class_id], dtype=torch.long),
            images=images.cpu(),
            hard_labels=hard_labels.cpu(),
            soft_targets=soft_targets,
            meta={
                "packet_type": "global_raw_x_q",
                "packet_source": source,
                "ipc": ipc,
                "temperature": temperature,
                "dataset": cfg["dataset"]["name"],
            },
        )
        packet_path = packet_dir / f"class_{class_id}_packet.pt"
        torch.save(packet, packet_path)
        print(f"saved: {packet_path} images={tuple(packet.images.shape)}")


if __name__ == "__main__":
    main()
