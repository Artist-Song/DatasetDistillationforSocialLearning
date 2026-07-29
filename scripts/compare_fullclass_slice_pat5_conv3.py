#!/usr/bin/env python3
"""Compare full-class ConvNet-3 pool slices with PAT5-local packets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
DSDM_ROOT = ROOT / "DSDM"
for path in (ROOT, DSDM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent_data import build_agent_args  # noqa: E402
from output_manager import atomic_write_json  # noqa: E402
from DSDM import Synthesizer, load_resized_data  # noqa: E402


PAT5_CONFIG = ROOT / "configs/pat_class_split/main_cifar100_pat5agent_seed0_ipc10.yaml"
OUTPUT_ROOT = ROOT / "outputs/cifar100_fullclass_slice_pat5_conv3_comparison_seed0"
SOURCES = (
    {
        "name": "full100_pure200_slice20",
        "role": "full100_then_slice",
        "guide_pool": "10_independent_epoch200",
        "packet": ROOT
        / "outputs/cifar100_fullclass_dsdm_conv3_dsdmguidee0200_ipc10_seed0"
        / "agents/agent_0/packets/dsdm_packet.pt",
    },
    {
        "name": "full100_trajectory_slice20",
        "role": "full100_then_slice",
        "guide_pool": "single_trajectory_e20_to_e200",
        "packet": ROOT
        / "outputs/cifar100_fullclass_dsdm_conv3_trajectorye0020to0200step0020_ipc10_seed0"
        / "agents/agent_0/packets/dsdm_packet.pt",
    },
    {
        "name": "pat5_local20_main",
        "role": "local20_direct_distillation",
        "guide_pool": "10_independent_epoch20",
        "packet": ROOT
        / "outputs/cifar100_pat5agent_20cls_ours_seed0_ipc10"
        / "agents/agent_0/packets/dsdm_packet.pt",
    },
    {
        "name": "pat5_local20_pure200",
        "role": "local20_direct_distillation",
        "guide_pool": "10_independent_epoch200",
        "packet": ROOT
        / "outputs/cifar100_tq_conv3_guidee0200_seed0_ipc10"
        / "agents/agent_0/packets/dsdm_packet.pt",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_packet_slice(packet_path: Path, class_ids: list[int], ipc: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
    packet = torch.load(packet_path, map_location="cpu", weights_only=False)
    images = packet.get("images")
    labels = packet.get("labels")
    if not torch.is_tensor(images) or not torch.is_tensor(labels):
        raise ValueError(f"Packet lacks image/label tensors: {packet_path}")
    if int(packet.get("factor", -1)) != 2 or packet.get("decode_type") != "single":
        raise ValueError(f"Unexpected decode protocol: {packet_path}")

    ordered_images = []
    ordered_labels = []
    for class_id in class_ids:
        indices = torch.nonzero(labels.long() == int(class_id), as_tuple=False).flatten()
        if indices.numel() != ipc:
            raise ValueError(
                f"Expected IPC={ipc} for class {class_id}, found {indices.numel()} in {packet_path}"
            )
        ordered_images.append(images[indices].float().clone())
        ordered_labels.append(labels[indices].long().clone())

    data = torch.cat(ordered_images)
    targets = torch.cat(ordered_labels)
    if not torch.isfinite(data).all():
        raise ValueError(f"Non-finite images in {packet_path}")
    if targets.tolist() != [class_id for class_id in class_ids for _ in range(ipc)]:
        raise ValueError(f"Global labels are not in requested class order: {packet_path}")
    try:
        source_packet = str(packet_path.resolve().relative_to(ROOT))
    except ValueError:
        source_packet = str(packet_path)
    metadata = {
        "source_packet": source_packet,
        "source_sha256": sha256(packet_path),
        "source_class_ids": [int(value) for value in packet.get("class_ids", [])],
        "raw_images_before_slice": int(images.shape[0]),
        "raw_images_after_slice": int(data.shape[0]),
        "factor": int(packet["factor"]),
        "decode_type": str(packet["decode_type"]),
    }
    return data, targets, metadata


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_packet(args, data: torch.Tensor, targets: torch.Tensor, val_loader, seed: int, logger) -> float:
    synthesizer = Synthesizer(args, args.nclass, args.nch, args.size, args.size, device="cuda")
    with torch.no_grad():
        synthesizer.data.copy_(data.to("cuda"))
        synthesizer.targets.copy_(targets.to("cuda"))
    seed_everything(seed)
    accuracy = float(synthesizer.test(args, val_loader, logger))
    del synthesizer
    torch.cuda.empty_cache()
    return accuracy


def main() -> None:
    cli = parse_args()
    config = yaml.safe_load(PAT5_CONFIG.read_text(encoding="utf-8"))
    eval_args = build_agent_args(config, str(PAT5_CONFIG), 0)
    class_ids = [int(value) for value in eval_args.active_class_ids]
    if len(class_ids) != 20 or len(set(class_ids)) != 20:
        raise ValueError(f"Expected 20 unique PAT5 classes, got {class_ids}")
    if eval_args.nclass != 100 or eval_args.ipc != 10:
        raise ValueError("Comparison requires a 100-way classifier and IPC=10")

    prepared = []
    for source in SOURCES:
        data, targets, metadata = load_packet_slice(source["packet"], class_ids, eval_args.ipc)
        prepared.append((source, data, targets, metadata))

    protocol = {
        "dataset": "cifar100",
        "agent": 0,
        "class_ids": class_ids,
        "global_labels_preserved": True,
        "classifier_outputs": 100,
        "raw_ipc": 10,
        "factor": 2,
        "decoded_images": 20 * 10 * 4,
        "evaluator": "fresh_convnet3w1",
        "epochs": int(eval_args.epochs),
        "batch_size": int(eval_args.batch_size),
        "lr": float(eval_args.lr),
        "dsa": bool(eval_args.dsa),
        "dsa_strategy": str(eval_args.dsa_strategy),
        "mixup": str(eval_args.mixup),
        "mix_p": float(eval_args.mix_p),
        "seed": int(cli.seed),
    }
    dry_payload = {
        "status": "validated_only" if cli.dry_run else "running",
        "protocol": protocol,
        "sources": [
            {**{key: source[key] for key in ("name", "role", "guide_pool")}, **metadata}
            for source, _, _, metadata in prepared
        ],
    }
    if cli.dry_run:
        print(json.dumps(dry_payload, indent=2, ensure_ascii=False))
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    eval_args.save_ckpt = False
    eval_args.pretrained = False
    eval_args.test = False
    _, val_loader = load_resized_data(eval_args)
    results = []
    for source, data, targets, metadata in prepared:
        source_dir = OUTPUT_ROOT / source["name"]
        source_dir.mkdir(parents=True, exist_ok=True)
        eval_args.save_dir = str(source_dir)

        def logger(message, name=source["name"]):
            print(f"[{name}] {message}", flush=True)

        logger("starting paired synthetic evaluation")
        accuracy = evaluate_packet(eval_args, data, targets, val_loader, cli.seed, logger)
        row = {
            "name": source["name"],
            "role": source["role"],
            "guide_pool": source["guide_pool"],
            "accuracy": accuracy,
            **metadata,
        }
        results.append(row)
        atomic_write_json(row, source_dir / "result.json")
        logger(f"completed accuracy={accuracy:.4f}")

    by_name = {row["name"]: row["accuracy"] for row in results}
    comparisons = {
        "pure200_full100_slice_minus_local20": (
            by_name["full100_pure200_slice20"] - by_name["pat5_local20_pure200"]
        ),
        "trajectory_full100_slice_minus_local20_main": (
            by_name["full100_trajectory_slice20"] - by_name["pat5_local20_main"]
        ),
    }
    payload = {
        "status": "complete",
        "protocol": protocol,
        "results": results,
        "comparisons_percentage_points": comparisons,
    }
    atomic_write_json(payload, OUTPUT_ROOT / "summary.json")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
