from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from agent_data import AGENT_CLASS_SPLIT, get_agent_dir, get_agent_train_dataset


def _ensure_dsdm_path():
    """确保 DSDM 源码目录可被导入。"""
    import sys

    root = Path(__file__).resolve().parent
    dsdm_root = root / "DSDM"
    if str(dsdm_root) not in sys.path:
        sys.path.insert(0, str(dsdm_root))


def train_agent_experts(args, agent_id, resume=False, overwrite=False):
    """训练单个 agent 的 guide model pool，并保存最后一个为 expert_model。"""
    _ensure_dsdm_path()
    from train import define_model

    agent_dir = get_agent_dir(args, agent_id)
    ckpt_dir = agent_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    expert_path = ckpt_dir / "expert_model.pt"
    if expert_path.exists() and resume and not overwrite:
        return expert_path

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    dataset = get_agent_train_dataset(args, agent_id, normalize=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    last_path = None
    for model_idx in range(int(args.pretrained_model_number)):
        model = define_model(args, 10).to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        model.train()
        for _ in range(int(args.pretrained_epochs)):
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
        last_path = ckpt_dir / f"guide_model_{model_idx}.pt"
        torch.save(model.state_dict(), last_path)
    if last_path is None:
        raise RuntimeError("pretrained_model_number 必须大于 0")
    torch.save(torch.load(last_path, map_location="cpu"), expert_path)
    return expert_path


def prepare_agent_pretrained_dir(args, agent_id):
    """把 guide_model 命名映射为 DSDM 期望的 cifar10_model_i.pth。"""
    ckpt_dir = get_agent_dir(args, agent_id) / "checkpoints"
    dsdm_dir = ckpt_dir / "dsdm_pretrained"
    dsdm_dir.mkdir(parents=True, exist_ok=True)
    for model_idx in range(int(args.pretrained_model_number)):
        src = ckpt_dir / f"guide_model_{model_idx}.pt"
        dst = dsdm_dir / f"{args.dataset}_model_{model_idx}.pth"
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            torch.save(torch.load(src, map_location="cpu"), dst)
    return dsdm_dir


def describe_agent(agent_id):
    """返回 agent 的固定类别说明。"""
    return f"agent_{agent_id}: classes={AGENT_CLASS_SPLIT[int(agent_id)]}"
