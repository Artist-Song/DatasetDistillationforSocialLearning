from pathlib import Path

import torch


def _empty_packet_like(source, images, labels):
    """构造非通信来源的训练数据块元信息。"""
    return {
        "images": images.float(),
        "labels": labels.long(),
        "raw_images": 0,
        "num_images": int(images.shape[0]),
        "decoded_for_training": False,
        "source": source,
        "class_ids": sorted({int(x) for x in labels.tolist()}),
        "meta": {},
        "has_sender_logits": False,
        "sender_logits": None,
        "sender_logit_class_ids": None,
        "sender_logit_dim": 0,
        "sender_logit_bytes": 0,
        "has_generalist_logits": False,
        "generalist_logits": None,
        "generalist_logit_dim": 0,
        "generalist_logit_bytes": 0,
    }


def _load_self_real_training_data(args, per_class_limit=None):
    """加载 receiver 本地真实 expert 数据，用作 self replay 且不计通信量。"""
    from agent_data import ActiveClassDataset, get_num_classes, get_train_dataset

    dataset = get_train_dataset(args, normalize=False, augment=False)
    active_classes = [int(c) for c in getattr(args, "active_class_ids", [])]
    subset = ActiveClassDataset(dataset, active_classes, num_classes=get_num_classes(args))
    if per_class_limit is None:
        per_class_limit = int(getattr(args, "self_real_per_class", 0) or 0)
    else:
        per_class_limit = int(per_class_limit)
    per_class_seen = {class_id: 0 for class_id in active_classes}
    images = []
    labels = []
    for index in range(len(subset)):
        image, label = subset[index]
        label = int(label)
        if per_class_limit > 0:
            if per_class_seen[label] >= per_class_limit:
                continue
            per_class_seen[label] += 1
        images.append(image)
        labels.append(label)
    if not images:
        raise ValueError("self_data_mode=real 但 receiver 本地真实数据为空")
    return _empty_packet_like("self_real", torch.stack(images), torch.tensor(labels, dtype=torch.long))


def load_receiver_local_real_data(args):
    """Public entry point for the DKP receiver's complete local-real stream."""
    return _load_self_real_training_data(args, per_class_limit=0)


def _decode_dsdm_images(args, packet):
    """复用 DSDM 原 decode_fn 解码 factorized synthetic data。"""
    from dsdm_decode import decode_fn

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
    elif source in {"heuristic", "importance", "fast", "full_real"}:
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


def _fill_missing_logits_for_self_data(packets, require_logits=False, require_generalist_logits=False, num_classes=0):
    """为 self real 数据补齐不会参与 KD 的占位 logits。"""
    if require_logits:
        logit_dim = next((p["sender_logit_dim"] for p in packets if p["has_sender_logits"]), 0)
        if logit_dim <= 0:
            raise ValueError("require_logits=true，但没有可用 sender logits")
        for packet in packets:
            if packet["has_sender_logits"]:
                continue
            if packet["source"] != "self_real":
                raise ValueError("部分 packet 缺少 sender_logits，请为全部通信 packet 运行 attach_logits")
            packet["sender_logits"] = torch.zeros((packet["num_images"], logit_dim), dtype=torch.float)
            packet["sender_logit_class_ids"] = torch.zeros((packet["num_images"], logit_dim), dtype=torch.long)
            packet["sender_logit_dim"] = int(logit_dim)
    if require_generalist_logits:
        logit_dim = next((p["generalist_logit_dim"] for p in packets if p["has_generalist_logits"]), int(num_classes))
        if logit_dim <= 0:
            raise ValueError("require_generalist_logits=true，但无法确定 generalist logits 维度")
        for packet in packets:
            if packet["has_generalist_logits"]:
                continue
            if packet["source"] != "self_real":
                raise ValueError("部分 packet 缺少 generalist_logits，请先运行 attach_generalist_logits")
            packet["generalist_logits"] = torch.zeros((packet["num_images"], logit_dim), dtype=torch.float)
            packet["generalist_logit_dim"] = int(logit_dim)


def consume_manifest_packets(args, manifest_rows, require_logits=False, require_generalist_logits=False, receiver_agent=None):
    """读取 manifest 中的 packet，并按需用本地真实数据替换 self packet。"""
    packets = []
    sender_agent_chunks = []
    receiver_agent = int(getattr(args, "agent_id", receiver_agent if receiver_agent is not None else -1))
    self_data_mode = str(getattr(args, "self_data_mode", "packet")).lower()
    if self_data_mode not in {"packet", "real"}:
        raise ValueError(f"不支持的 self_data_mode: {self_data_mode}")
    for row in manifest_rows:
        sender_agent = int(row["sender_agent"])
        if self_data_mode == "real" and sender_agent == receiver_agent:
            consumed = _load_self_real_training_data(args)
            consumed["sender_agent"] = sender_agent
            packets.append(consumed)
            sender_agent_chunks.append(torch.full((consumed["images"].shape[0],), sender_agent, dtype=torch.long))
            continue
        consumed = consume_packet_for_training(args, row["packet_path"], require_sender_logits=require_logits, require_generalist_logits=require_generalist_logits)
        consumed["sender_agent"] = sender_agent
        packets.append(consumed)
        sender_agent_chunks.append(torch.full((consumed["images"].shape[0],), sender_agent, dtype=torch.long))
    _fill_missing_logits_for_self_data(
        packets,
        require_logits=require_logits,
        require_generalist_logits=require_generalist_logits,
        num_classes=int(getattr(args, "num_classes", getattr(args, "nclass", 0))),
    )
    images = torch.cat([p["images"] for p in packets])
    labels = torch.cat([p["labels"] for p in packets])
    sender_agents = torch.cat(sender_agent_chunks)
    sender_logits = None
    sender_logit_class_ids = None
    if require_logits:
        sender_logits = torch.cat([p["sender_logits"] for p in packets])
        sender_logit_class_ids = torch.cat([p["sender_logit_class_ids"] for p in packets])

    generalist_logits = None
    if require_generalist_logits:
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


def consume_external_manifest_packets(args, manifest_rows, receiver_agent, require_logits=False):
    """Consume only the other senders' packets for the DKP-SL external stream."""
    receiver_agent = int(receiver_agent)
    packets = []
    sender_agent_chunks = []
    for row in manifest_rows:
        sender_agent = int(row["sender_agent"])
        if sender_agent == receiver_agent:
            continue
        consumed = consume_packet_for_training(
            args,
            row["packet_path"],
            require_sender_logits=require_logits,
            require_generalist_logits=False,
        )
        consumed["sender_agent"] = sender_agent
        packets.append(consumed)
        sender_agent_chunks.append(
            torch.full((consumed["num_images"],), sender_agent, dtype=torch.long)
        )
    if not packets:
        raise ValueError(f"receiver {receiver_agent} has no external DKP packets")

    result = {
        "images": torch.cat([packet["images"] for packet in packets]),
        "labels": torch.cat([packet["labels"] for packet in packets]),
        "sender_agents": torch.cat(sender_agent_chunks),
        "sender_logits": None,
        "sender_logit_class_ids": None,
        "packets": packets,
    }
    if require_logits:
        result["sender_logits"] = torch.cat([packet["sender_logits"] for packet in packets])
        result["sender_logit_class_ids"] = torch.cat(
            [packet["sender_logit_class_ids"] for packet in packets]
        )
    return result


def consume_receiver_manifest_packet(args, manifest_rows, receiver_agent):
    """Load the receiver's own DKP without counting it as external communication."""
    receiver_agent = int(receiver_agent)
    matching_rows = [
        row for row in manifest_rows if int(row["sender_agent"]) == receiver_agent
    ]
    if len(matching_rows) != 1:
        raise ValueError(
            f"receiver {receiver_agent} must have exactly one self DKP manifest row; "
            f"found {len(matching_rows)}"
        )
    row = matching_rows[0]
    consumed = consume_packet_for_training(
        args,
        row["packet_path"],
        require_sender_logits=False,
        require_generalist_logits=False,
    )
    consumed["sender_agent"] = receiver_agent
    consumed["packet_path"] = str(Path(row["packet_path"]).resolve())
    consumed["manifest_packet_sha256"] = str(row.get("packet_sha256", ""))
    return consumed
