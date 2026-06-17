from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from agent_data import get_agent_class_split, get_agent_dir, get_agent_model_split, get_num_classes, get_receiver_dir
from packet_consumer import consume_manifest_packets
from social_metrics import evaluate_receiver_model


def _ensure_dsdm_path():
    """确保 DSDM 源码目录可被导入。"""
    import sys

    root = Path(__file__).resolve().parent
    dsdm_root = root / "DSDM"
    if str(dsdm_root) not in sys.path:
        sys.path.insert(0, str(dsdm_root))


def _build_balanced_loader(args, images, labels, sender_logits=None, sender_logit_class_ids=None, sender_agents=None):
    """根据类别频次构建 class-balanced sampler，并可选携带 sender logits。"""
    from data import MEANS, STDS

    mean = torch.tensor(MEANS[args.dataset]).view(1, -1, 1, 1)
    std = torch.tensor(STDS[args.dataset]).view(1, -1, 1, 1)
    images = (images - mean) / std
    counts = torch.bincount(labels, minlength=get_num_classes(args)).float()
    weights = torch.tensor([1.0 / max(1.0, counts[int(y)].item()) for y in labels], dtype=torch.float)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    if sender_logits is None:
        dataset = TensorDataset(images.float(), labels.long())
    else:
        dataset = TensorDataset(
            images.float(),
            labels.long(),
            sender_logits.float(),
            sender_logit_class_ids.long(),
            sender_agents.long(),
        )
    return DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)


class SocialTrainer:
    """负责 receiver agent 的二轮社会化学习。"""

    def __init__(self, args, receiver_agent, manifest_rows):
        """初始化 receiver、packet 和训练设备。"""
        _ensure_dsdm_path()
        self.args = args
        self.receiver_agent = int(receiver_agent)
        self.manifest_rows = manifest_rows
        self.class_split = get_agent_class_split(args)
        self.model_split = get_agent_model_split(args)
        self.expert_classes = self.class_split[self.receiver_agent]
        self.device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    def _build_models(self):
        """构建 before/after 模型并加载 receiver expert 权重。"""
        from train import define_model

        self.args.net_type = self.model_split[self.receiver_agent]
        model_old = define_model(self.args, get_num_classes(self.args)).to(self.device)
        model_new = define_model(self.args, get_num_classes(self.args)).to(self.device)
        if getattr(self.args, "init_mode", "expert") == "expert":
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

    def _compute_kd_loss(self, student_logits, teacher_logits, teacher_class_ids, sender_agents):
        """只对 external sender packet 计算 temperature KL KD。"""
        mask = sender_agents != self.receiver_agent
        if not mask.any():
            return torch.tensor(0.0, device=self.device)
        temperature = float(getattr(self, "kd_temperature", 2.0))
        student_selected = student_logits[mask].gather(1, teacher_class_ids[mask])
        teacher_selected = teacher_logits[mask]
        teacher_p = torch.softmax(teacher_selected / temperature, dim=1)
        student_log_p = torch.log_softmax(student_selected / temperature, dim=1)
        return temperature * temperature * nn.functional.kl_div(student_log_p, teacher_p, reduction="batchmean")

    def train(self):
        """执行 receiver 二轮训练并返回结果指标。"""
        model_old, model_new = self._build_models()
        receiver_dir = get_receiver_dir(self.args, self.receiver_agent)
        (receiver_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        torch.save(model_old.state_dict(), receiver_dir / "checkpoints" / "before_social.pt")

        before = evaluate_receiver_model(self.args, model_old, self.receiver_agent, self.device)
        use_logits = bool(getattr(self.args, "use_logits", False))
        consumed = consume_manifest_packets(self.args, self.manifest_rows, require_logits=use_logits)
        packets = consumed["packets"]
        if use_logits and consumed["sender_logits"] is None:
            raise ValueError("logits.enabled=true，但 packet 中没有 sender_logits，请先运行 attach_logits。")
        loader = _build_balanced_loader(
            self.args,
            consumed["images"],
            consumed["labels"],
            consumed["sender_logits"] if use_logits else None,
            consumed["sender_logit_class_ids"] if use_logits else None,
            consumed["sender_agents"] if use_logits else None,
        )
        receiver_lr = float(getattr(self.args, "receiver_lr", self.args.lr))
        receiver_epochs = int(getattr(self.args, "receiver_epochs", self.args.epochs))
        init_mode = getattr(self.args, "init_mode", "expert")
        use_fr = bool(getattr(self.args, "use_fr", init_mode == "expert"))
        lambda_fr = float(getattr(self.args, "lambda_fr", 0.05)) if use_fr else 0.0
        self.kd_temperature = float(getattr(self.args, "kd_temperature", 2.0))
        lambda_kd = float(getattr(self.args, "lambda_kd", 0.0)) if use_logits else 0.0
        optimizer = optim.SGD(model_new.parameters(), lr=receiver_lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        last_cls = 0.0
        last_fr = 0.0
        last_kd = 0.0
        model_new.train()
        for _ in range(receiver_epochs):
            for batch in loader:
                if use_logits:
                    batch_images, batch_labels, batch_teacher_logits, batch_teacher_class_ids, batch_sender_agents = batch
                    batch_teacher_logits = batch_teacher_logits.to(self.device)
                    batch_teacher_class_ids = batch_teacher_class_ids.to(self.device)
                    batch_sender_agents = batch_sender_agents.to(self.device)
                else:
                    batch_images, batch_labels = batch
                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.to(self.device)
                optimizer.zero_grad()
                logits = model_new(batch_images)
                loss_cls = criterion(logits, batch_labels)
                loss_fr = self._compute_fr_loss(model_old, model_new, batch_images, batch_labels) if use_fr else torch.tensor(0.0, device=self.device)
                loss_kd = (
                    self._compute_kd_loss(logits, batch_teacher_logits, batch_teacher_class_ids, batch_sender_agents)
                    if use_logits
                    else torch.tensor(0.0, device=self.device)
                )
                loss = loss_cls + lambda_fr * loss_fr + lambda_kd * loss_kd
                loss.backward()
                optimizer.step()
                last_cls = float(loss_cls.detach().cpu())
                last_fr = float(loss_fr.detach().cpu())
                last_kd = float(loss_kd.detach().cpu())
        torch.save(model_new.state_dict(), receiver_dir / "checkpoints" / "after_social.pt")
        after = evaluate_receiver_model(self.args, model_new, self.receiver_agent, self.device)
        external_raw = sum(p["raw_images"] for p in packets if p["sender_agent"] != self.receiver_agent)
        external_logit_bytes = sum(p["sender_logit_bytes"] for p in packets if p["sender_agent"] != self.receiver_agent)
        base_method = getattr(self.args, "packet_method", "dsdm").upper()
        method = f"{base_method}_LOGIT" if use_logits else base_method
        return {
            "receiver_agent": self.receiver_agent,
            "receiver_model": self.model_split[self.receiver_agent],
            "expert_classes": ",".join(str(c) for c in self.expert_classes),
            "packet_method": getattr(self.args, "packet_method", "dsdm"),
            "method": method,
            "init_mode": init_mode,
            "use_fr": str(use_fr).lower(),
            "lambda_fr": lambda_fr,
            "use_logits": str(use_logits).lower(),
            "lambda_kd": lambda_kd,
            "kd_temperature": self.kd_temperature,
            "ipc": int(self.args.ipc),
            "external_comm_images": int(external_raw),
            "external_comm_logit_bytes": int(external_logit_bytes),
            "acc_global_before": before["acc_global"],
            "acc_expert_before": before["acc_expert"],
            "acc_global_after": after["acc_global"],
            "acc_expert_after": after["acc_expert"],
            "acc_new_after": after["acc_new"],
            "forgetting": before["acc_expert"] - after["acc_expert"],
            "loss_cls": last_cls,
            "loss_fr": last_fr,
            "loss_kd": last_kd,
        }
