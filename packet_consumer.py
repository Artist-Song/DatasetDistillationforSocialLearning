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


def consume_packet_for_training(args, packet_path, require_sender_logits=False, require_generalist_logits=False):
    """读取 packet，并转换为 receiver 可直接训练的 images/labels/logits。"""
    packet = torch.load(Path(packet_path), map_location="cpu")
    source = packet.get("source", "")
    if source == "dsdm":
        images, labels = _decode_dsdm_images(args, packet)
        decoded_for_training = bool(images.shape[0] != packet["images"].shape[0])
    elif source in {"heuristic", "importance", "full_real"}:
        images, labels = packet["images"].cpu(), packet["labels"].cpu()
        decoded_for_training = False
    else:
        raise ValueError(f"不支持的 packet source: {source}")
    has_sender_logits = bool(packet.get("has_sender_logits", False))
    sender_logits = None
    sender_logit_class_ids = None
    sender_logit_dim = 0
    sender_logit_bytes = 0
    if has_sender_logits:
        sender_logits = packet["sender_logits"].cpu()
        if sender_logits.shape[0] != images.shape[0]:
            raise ValueError("sender_logits 数量与训练 images 数量不一致")
        class_ids = packet["sender_logit_class_ids"].long().cpu()
        sender_logit_class_ids = class_ids.view(1, -1).repeat(images.shape[0], 1)
        sender_logit_dim = int(sender_logits.shape[1])
        sender_logit_bytes = int(sender_logits.numel() * sender_logits.element_size())
    elif require_sender_logits:
        raise ValueError(f"packet 缺少 sender_logits: {packet_path}")

    has_generalist_logits = bool(packet.get("has_generalist_logits", False))
    generalist_logits = None
    generalist_logit_dim = 0
    generalist_logit_bytes = 0
    if has_generalist_logits:
        generalist_logits = packet["generalist_logits"].cpu()
        if generalist_logits.shape[0] != images.shape[0]:
            raise ValueError("generalist_logits 数量与训练 images 数量不一致")
        generalist_logit_dim = int(generalist_logits.shape[1])
        generalist_logit_bytes = int(generalist_logits.numel() * generalist_logits.element_size())
    elif require_generalist_logits:
        raise ValueError(f"packet 缺少 generalist_logits: {packet_path}")

    return {
        "images": images.float(),
        "labels": labels.long(),
        "raw_images": int(packet["images"].shape[0]),
        "num_images": int(images.shape[0]),
        "decoded_for_training": decoded_for_training,
        "source": source,
        "class_ids": packet.get("class_ids", []),
        "meta": packet.get("meta", {}),
        "has_sender_logits": has_sender_logits,
        "sender_logits": sender_logits,
        "sender_logit_class_ids": sender_logit_class_ids,
        "sender_logit_dim": sender_logit_dim,
        "sender_logit_bytes": sender_logit_bytes,
        "has_generalist_logits": has_generalist_logits,
        "generalist_logits": generalist_logits,
        "generalist_logit_dim": generalist_logit_dim,
        "generalist_logit_bytes": generalist_logit_bytes,
    }


def consume_manifest_packets(args, manifest_rows, require_logits=False, require_generalist_logits=False):
    """读取 manifest 中的全部 packet 并拼接成训练张量和可选 logits。"""
    packets = []
    sender_agent_chunks = []
    for row in manifest_rows:
        consumed = consume_packet_for_training(args, row["packet_path"], require_sender_logits=require_logits, require_generalist_logits=require_generalist_logits)
        consumed["sender_agent"] = int(row["sender_agent"])
        packets.append(consumed)
        sender_agent_chunks.append(torch.full((consumed["images"].shape[0],), int(row["sender_agent"]), dtype=torch.long))
    images = torch.cat([p["images"] for p in packets])
    labels = torch.cat([p["labels"] for p in packets])
    sender_agents = torch.cat(sender_agent_chunks)
    has_any_logits = any(p["has_sender_logits"] for p in packets)
    has_all_logits = all(p["has_sender_logits"] for p in packets)
    sender_logits = None
    sender_logit_class_ids = None
    if require_logits:
        if not has_all_logits:
            raise ValueError("部分 packet 缺少 sender_logits，请为全部 packet 运行 attach_logits")
        sender_logits = torch.cat([p["sender_logits"] for p in packets])
        sender_logit_class_ids = torch.cat([p["sender_logit_class_ids"] for p in packets])

    has_all_generalist_logits = all(p["has_generalist_logits"] for p in packets)
    generalist_logits = None
    if require_generalist_logits:
        if not has_all_generalist_logits:
            raise ValueError("部分 packet 缺少 generalist_logits，请先运行 attach_generalist_logits")
        generalist_logits = torch.cat([p["generalist_logits"] for p in packets])
    return {
        "images": images,
        "labels": labels,
        "sender_logits": sender_logits,
        "sender_logit_class_ids": sender_logit_class_ids,
        "generalist_logits": generalist_logits,
        "sender_agents": sender_agents,
        "packets": packets,
    }
