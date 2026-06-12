from pathlib import Path

import torch


def _decode_dsdm_images(args, packet):
    """复用 DSDM 原 decode_fn 解码 factorized synthetic data。"""
    from test import decode_fn

    images = packet["images"]
    labels = packet["labels"]
    meta = packet.get("meta", {})
    factor = int(packet.get("factor", meta.get("factor", getattr(args, "factor", 1))))
    decode_type = packet.get("decode_type", meta.get("decode_type", getattr(args, "decode_type", "single")))
    if factor <= 1:
        return images, labels

    decoded_images = []
    decoded_labels = []
    class_ids = packet.get("class_ids", sorted({int(x) for x in labels.tolist()}))
    for class_id in class_ids:
        mask = labels == int(class_id)
        data_c = images[mask].detach()
        target_c = labels[mask].detach()
        data_c, target_c = decode_fn(
            data_c,
            target_c,
            factor,
            decode_type,
            bound=getattr(args, "batch_syn_max", 128),
        )
        decoded_images.append(data_c.cpu())
        decoded_labels.append(target_c.cpu())
    return torch.cat(decoded_images), torch.cat(decoded_labels)


def consume_packet_for_training(args, packet_path):
    """读取 packet，并转换为 receiver 可直接训练的 images/labels。"""
    packet = torch.load(Path(packet_path), map_location="cpu")
    source = packet.get("source", "")
    if source == "dsdm":
        images, labels = _decode_dsdm_images(args, packet)
        decoded_for_training = bool(images.shape[0] != packet["images"].shape[0])
    elif source in {"heuristic", "importance"}:
        images, labels = packet["images"].cpu(), packet["labels"].cpu()
        decoded_for_training = False
    else:
        raise ValueError(f"不支持的 packet source: {source}")
    return {
        "images": images.float(),
        "labels": labels.long(),
        "raw_images": int(packet["images"].shape[0]),
        "num_images": int(images.shape[0]),
        "decoded_for_training": decoded_for_training,
        "source": source,
        "class_ids": packet.get("class_ids", []),
        "meta": packet.get("meta", {}),
    }


def consume_manifest_packets(args, manifest_rows):
    """读取 manifest 中的全部 packet 并拼接成训练张量。"""
    packets = []
    for row in manifest_rows:
        consumed = consume_packet_for_training(args, row["packet_path"])
        consumed["sender_agent"] = int(row["sender_agent"])
        packets.append(consumed)
    images = torch.cat([p["images"] for p in packets])
    labels = torch.cat([p["labels"] for p in packets])
    return images, labels, packets
