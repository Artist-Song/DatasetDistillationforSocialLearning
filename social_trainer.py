from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from agent_data import AGENT_CLASS_SPLIT, AGENT_MODEL_SPLIT, get_agent_dir, get_receiver_dir
from packet_consumer import consume_manifest_packets
from social_metrics import evaluate_receiver_model


def _ensure_dsdm_path():
    """确保 DSDM 源码目录可被导入。"""
    import sys

    root = Path(__file__).resolve().parent
    dsdm_root = root / "DSDM"
    if str(dsdm_root) not in sys.path:
        sys.path.insert(0, str(dsdm_root))


def _build_balanced_loader(args, images, labels):
    """根据类别频次构建 class-balanced sampler。"""
    from data import MEANS, STDS

    mean = torch.tensor(MEANS[args.dataset]).view(1, -1, 1, 1)
    std = torch.tensor(STDS[args.dataset]).view(1, -1, 1, 1)
    images = (images - mean) / std
    counts = torch.bincount(labels, minlength=10).float()
    weights = torch.tensor([1.0 / max(1.0, counts[int(y)].item()) for y in labels], dtype=torch.float)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    dataset = TensorDataset(images.float(), labels.long())
    return DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)


class SocialTrainer:
    """负责 receiver agent 的二轮社会化学习。"""

    def __init__(self, args, receiver_agent, manifest_rows):
        """初始化 receiver、packet 和训练设备。"""
        _ensure_dsdm_path()
        self.args = args
        self.receiver_agent = int(receiver_agent)
        self.manifest_rows = manifest_rows
        self.expert_classes = AGENT_CLASS_SPLIT[self.receiver_agent]
        self.device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    def _build_models(self):
        """构建 before/after 模型并加载 receiver expert 权重。"""
        from train import define_model

        self.args.net_type = AGENT_MODEL_SPLIT[self.receiver_agent]
        model_old = define_model(self.args, 10).to(self.device)
        model_new = define_model(self.args, 10).to(self.device)
        expert_path = get_agent_dir(self.args, self.receiver_agent) / "checkpoints" / "expert_model.pt"
        state = torch.load(expert_path, map_location=self.device)
        model_old.load_state_dict(state)
        model_new.load_state_dict(state)
        for param in model_old.parameters():
            param.requires_grad = False
        model_old.eval()
        return model_old, model_new

    def _compute_fr_loss(self, model_old, model_new, images, labels):
        """只对 receiver expert 类样本计算 expert logits MSE。"""
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for class_id in self.expert_classes:
            mask |= labels == int(class_id)
        if not mask.any():
            return torch.tensor(0.0, device=self.device)
        class_index = torch.tensor(self.expert_classes, device=self.device, dtype=torch.long)
        with torch.no_grad():
            old_logits = model_old(images[mask])[:, class_index]
        new_logits = model_new(images[mask])[:, class_index]
        return nn.functional.mse_loss(new_logits, old_logits)

    def train(self):
        """执行 receiver 二轮训练并返回结果指标。"""
        model_old, model_new = self._build_models()
        receiver_dir = get_receiver_dir(self.args, self.receiver_agent)
        (receiver_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        torch.save(model_old.state_dict(), receiver_dir / "checkpoints" / "before_social.pt")

        before = evaluate_receiver_model(self.args, model_old, self.receiver_agent, self.device)
        images, labels, packets = consume_manifest_packets(self.args, self.manifest_rows)
        loader = _build_balanced_loader(self.args, images, labels)
        optimizer = optim.SGD(model_new.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        last_cls = 0.0
        last_fr = 0.0
        model_new.train()
        for _ in range(int(self.args.epochs)):
            for batch_images, batch_labels in loader:
                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.to(self.device)
                optimizer.zero_grad()
                logits = model_new(batch_images)
                loss_cls = criterion(logits, batch_labels)
                loss_fr = self._compute_fr_loss(model_old, model_new, batch_images, batch_labels)
                loss = loss_cls + 0.05 * loss_fr
                loss.backward()
                optimizer.step()
                last_cls = float(loss_cls.detach().cpu())
                last_fr = float(loss_fr.detach().cpu())
        torch.save(model_new.state_dict(), receiver_dir / "checkpoints" / "after_social.pt")
        after = evaluate_receiver_model(self.args, model_new, self.receiver_agent, self.device)
        external_raw = sum(p["raw_images"] for p in packets if p["sender_agent"] != self.receiver_agent)
        return {
            "receiver_agent": self.receiver_agent,
            "receiver_model": AGENT_MODEL_SPLIT[self.receiver_agent],
            "expert_classes": ",".join(str(c) for c in self.expert_classes),
            "method": "DSDM",
            "ipc": int(self.args.ipc),
            "external_comm_images": int(external_raw),
            "acc_global_before": before["acc_global"],
            "acc_expert_before": before["acc_expert"],
            "acc_global_after": after["acc_global"],
            "acc_expert_after": after["acc_expert"],
            "acc_new_after": after["acc_new"],
            "forgetting": before["acc_expert"] - after["acc_expert"],
            "loss_cls": last_cls,
            "loss_fr": last_fr,
        }
