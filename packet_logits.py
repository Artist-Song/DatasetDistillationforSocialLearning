from pathlib import Path

import torch

from agent_data import get_agent_dir, get_num_classes
from output_manager import atomic_torch_save
from packet_consumer import _decode_dsdm_images


def _ensure_dsdm_path():
    """确保 DSDM 源码目录可被导入。"""
    import sys

    root = Path(__file__).resolve().parent
    dsdm_root = root / "DSDM"
    if str(dsdm_root) not in sys.path:
        sys.path.insert(0, str(dsdm_root))


def _normalize_images(args, images, device):
    """按 DSDM 数据集均值方差归一化图片。"""
    from data import MEANS, STDS

    mean = torch.tensor(MEANS[args.dataset], device=device).view(1, -1, 1, 1)
    std = torch.tensor(STDS[args.dataset], device=device).view(1, -1, 1, 1)
    return (images.to(device).float() - mean) / std


def _load_sender_model(args, expert_path, device):
    """加载 sender expert model 并返回 eval 模型。"""
    from train import define_model

    model = define_model(args, get_num_classes(args)).to(device)
    model.load_state_dict(torch.load(expert_path, map_location=device))
    model.eval()
    return model


def _packet_training_images(args, packet):
    """返回与 receiver 训练样本对齐的 packet 图片。"""
    if packet.get("source") == "dsdm":
        images, _ = _decode_dsdm_images(args, packet)
        return images
    if packet.get("source") in {"heuristic", "importance", "fast", "full_real"}:
        return packet["images"].detach().cpu()
    raise ValueError(f"不支持的 packet source: {packet.get('source')}")


def attach_sender_logits_to_packet(args, agent_id, packet_method):
    """为指定 agent 的 packet 附加 sender expert-class logits。"""
    _ensure_dsdm_path()
    agent_dir = get_agent_dir(args, agent_id)
    packet_path = agent_dir / "packets" / f"{packet_method}_packet.pt"
    expert_path = agent_dir / "checkpoints" / "expert_model.pt"
    if not packet_path.exists():
        raise FileNotFoundError(f"缺少 agent packet: {packet_path}")
    if not expert_path.exists():
        raise FileNotFoundError(f"缺少 sender expert model: {expert_path}")

    packet = torch.load(packet_path, map_location="cpu")
    if packet.get("has_sender_logits", False):
        print(f"[attach_logits] 覆盖已有 sender_logits: {packet_path}")
    sender_class_ids = [int(c) for c in packet.get("class_ids", [])]
    if not sender_class_ids:
        raise ValueError(f"packet 缺少 class_ids: {packet_path}")

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = _load_sender_model(args, expert_path, device)
    images = _packet_training_images(args, packet)
    batch_size = int(getattr(args, "batch_size", 64))
    class_index = torch.tensor(sender_class_ids, device=device, dtype=torch.long)
    logits_chunks = []
    with torch.no_grad():
        for start in range(0, images.shape[0], batch_size):
            batch = _normalize_images(args, images[start : start + batch_size], device)
            full_logits = model(batch)
            logits_chunks.append(full_logits[:, class_index].detach().cpu())
    sender_logits = torch.cat(logits_chunks, dim=0)

    packet["has_sender_logits"] = True
    packet["sender_logits"] = sender_logits.half()
    packet["sender_logit_class_ids"] = torch.tensor(sender_class_ids, dtype=torch.long)
    packet["sender_logit_dim"] = len(sender_class_ids)
    packet["sender_logit_num_images"] = int(sender_logits.shape[0])
    packet["sender_logit_dtype"] = "float16"
    atomic_torch_save(packet, packet_path)
    print(
        f"[attach_logits] agent={agent_id} method={packet_method} "
        f"shape={tuple(packet['sender_logits'].shape)} path={packet_path}"
    )
    return packet_path
