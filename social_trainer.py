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


def _build_balanced_loader(args, images, labels, sender_logits=None, sender_logit_class_ids=None, sender_agents=None, generalist_logits=None):
    """根据类别频次构建 class-balanced sampler，并可选携带 sender/generalist logits。"""
    from data import MEANS, STDS

    mean = torch.tensor(MEANS[args.dataset]).view(1, -1, 1, 1)
    std = torch.tensor(STDS[args.dataset]).view(1, -1, 1, 1)
    images = (images - mean) / std
    counts = torch.bincount(labels, minlength=get_num_classes(args)).float()
    expert_classes = {int(c) for c in getattr(args, "active_class_ids", [])}
    self_class_weight = float(getattr(args, "self_class_weight", 1.0))
    weights = []
    for label in labels:
        base_weight = 1.0 / max(1.0, counts[int(label)].item())
        if int(label) in expert_classes:
            base_weight *= self_class_weight
        weights.append(base_weight)
    weights = torch.tensor(weights, dtype=torch.float)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    if sender_logits is None and generalist_logits is None:
        dataset = TensorDataset(images.float(), labels.long())
    else:
        tensors = [images.float(), labels.long()]
        if sender_logits is not None:
            tensors.extend([sender_logits.float(), sender_logit_class_ids.long()])
        if generalist_logits is not None:
            tensors.append(generalist_logits.float())
        tensors.append(sender_agents.long())
        dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)


def _augment_cifar_batch(images, args):
    """对已归一化的 CIFAR batch 应用逐样本 crop/flip。"""
    if not bool(getattr(args, "receiver_augment", False)):
        return images
    if str(getattr(args, "dataset", "")).lower() not in {"cifar10", "cifar100"}:
        return images
    _ensure_dsdm_path()
    from data import MEANS, STDS

    padding = 4
    batch, channels, height, width = images.shape
    mean = torch.tensor(MEANS[args.dataset], device=images.device, dtype=images.dtype)
    std = torch.tensor(STDS[args.dataset], device=images.device, dtype=images.dtype)
    raw_zero = (-mean / std).view(1, channels, 1, 1)
    padded = raw_zero.expand(batch, channels, height + 2 * padding, width + 2 * padding).clone()
    padded[:, :, padding:padding + height, padding:padding + width] = images
    augmented = torch.empty_like(images)
    offsets = torch.randint(0, 2 * padding + 1, (batch, 2), device=images.device)
    for index in range(batch):
        top, left = offsets[index].tolist()
        augmented[index] = padded[index, :, top:top + height, left:left + width]
    flip_mask = torch.rand(batch, device=images.device) < 0.5
    augmented[flip_mask] = torch.flip(augmented[flip_mask], dims=(-1,))
    return augmented


def _freeze_batchnorm_stats(model):
    """冻结 BatchNorm running statistics，同时保留 affine 参数可训练。"""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


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
        if args.device == "cuda" and torch.cuda.is_available():
            gpu_id = int(getattr(args, "gpu_id", 0) or 0)
            if gpu_id >= torch.cuda.device_count():
                raise ValueError(f"配置 gpu_id={gpu_id}，但当前只检测到 {torch.cuda.device_count()} 张 GPU")
            # DSDM/train.py 内部存在 device='cuda'，这里先设置当前 GPU，保证导入后落到目标卡。
            torch.cuda.set_device(gpu_id)
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")

    def _build_models(self):
        """构建 before/after 模型并加载 receiver expert 权重。"""
        from train import define_model

        # receiver_args 已在 build_agent_args 中解析为 DSDM 可识别的 family/depth/width。
        self.args.model_name = self.model_split[self.receiver_agent]
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

    # ──────────────────────────────────────────────────────────────────────────
    # 改进 Loss 方法（新增，不修改原有方法）
    # ──────────────────────────────────────────────────────────────────────────

    def _get_penultimate_feature(self, model, images):
        """提取 penultimate 特征向量，兼容 ConvNet/VGG/AlexNet/ResNet 的不同返回格式。"""
        f_idx = int(getattr(self.args, "idx_from", 2))
        result = model.get_feature(images, f_idx, f_idx)
        # VGG 返回 (list, None)，其他返回 list
        if isinstance(result, tuple):
            result = result[0]
        feat = result[0] if isinstance(result, list) else result
        # 展平为 [B, D]
        if feat.dim() > 2:
            feat = feat.reshape(feat.size(0), -1)
        return feat

    def _compute_fr_loss_kl(self, model_old, model_new, images, labels):
        """KL 散度版 FR loss：约束 expert 类概率分布，不受 logit 尺度影响。
        用法：config 中设 fr_loss_type: 'kl'"""
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for class_id in self.expert_classes:
            mask |= labels == int(class_id)
        if not mask.any():
            return torch.tensor(0.0, device=self.device)
        class_index = torch.tensor(self.expert_classes, device=self.device, dtype=torch.long)
        temperature = float(getattr(self.args, "fr_kl_temperature", 2.0))
        with torch.no_grad():
            old_logits = model_old(images[mask])[:, class_index]
        new_logits = model_new(images[mask])[:, class_index]
        old_p = torch.softmax(old_logits / temperature, dim=1)
        new_log_p = torch.log_softmax(new_logits / temperature, dim=1)
        return temperature * temperature * nn.functional.kl_div(new_log_p, old_p, reduction="batchmean")

    def _compute_fr_feat_loss(self, model_old, model_new, images, labels):
        """特征层余弦一致性 FR：约束中间层方向，防止浅层漂移导致遗忘。
        用法：config 中设 use_fr_feat: true"""
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for class_id in self.expert_classes:
            mask |= labels == int(class_id)
        if not mask.any():
            return torch.tensor(0.0, device=self.device)
        imgs_expert = images[mask]
        with torch.no_grad():
            old_feat = self._get_penultimate_feature(model_old, imgs_expert)
        new_feat = self._get_penultimate_feature(model_new, imgs_expert)
        cos_sim = nn.functional.cosine_similarity(new_feat, old_feat, dim=1)
        return 1.0 - cos_sim.mean()

    def _compute_self_kd_loss(self, model_old, model_new, images):
        """全 batch 自蒸馏：用旧模型对所有数据做软标签约束，防止自身知识退化。
        用法：config 中设 use_self_kd: true"""
        temperature = float(getattr(self.args, "self_kd_temperature", 2.0))
        with torch.no_grad():
            old_logits = model_old(images)
        new_logits = model_new(images)
        old_p = torch.softmax(old_logits / temperature, dim=1)
        new_log_p = torch.log_softmax(new_logits / temperature, dim=1)
        return temperature * temperature * nn.functional.kl_div(new_log_p, old_p, reduction="batchmean")

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

    def _compute_generalist_kd_loss(self, student_logits, generalist_logits, sender_agents):
        """只对 external packet 计算 full-class generalist KD。"""
        mask = sender_agents != self.receiver_agent
        if not mask.any():
            return torch.tensor(0.0, device=self.device)
        temperature = float(getattr(self, "kd_temperature", 2.0))
        teacher_p = torch.softmax(generalist_logits[mask] / temperature, dim=1)
        student_log_p = torch.log_softmax(student_logits[mask] / temperature, dim=1)
        return temperature * temperature * nn.functional.kl_div(student_log_p, teacher_p, reduction="batchmean")

    def _build_receiver_scheduler(self, optimizer, receiver_epochs):
        """按配置构建 receiver 端学习率调度器。"""
        scheduler_name = str(getattr(self.args, "receiver_scheduler", "none")).lower()
        if scheduler_name in {"", "none"}:
            return None
        if scheduler_name == "multistep":
            milestones = getattr(self.args, "receiver_scheduler_milestones", None)
            if not milestones:
                milestones = [int(0.65 * receiver_epochs), int(0.85 * receiver_epochs)]
            milestones = [max(1, int(v)) for v in milestones]
            gamma = float(getattr(self.args, "receiver_scheduler_gamma", 0.2))
            return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        if scheduler_name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(receiver_epochs))
        raise ValueError(f"不支持的 receiver scheduler: {scheduler_name}")

    def _current_loss_weights(self, epoch, receiver_epochs, base_lambda_fr, base_lambda_kd):
        """根据可选两段式 schedule 返回当前 FR/KD 权重。"""
        schedule = str(getattr(self.args, "lambda_schedule", "none")).lower()
        if schedule in {"", "none"}:
            return base_lambda_fr, base_lambda_kd
        if schedule != "late_fr":
            raise ValueError(f"不支持的 lambda_schedule: {schedule}")
        switch_ratio = float(getattr(self.args, "lambda_schedule_switch", 0.7))
        if epoch < int(receiver_epochs * switch_ratio):
            return base_lambda_fr, base_lambda_kd
        fr_multiplier = float(getattr(self.args, "lambda_fr_late_multiplier", 1.5))
        kd_multiplier = float(getattr(self.args, "lambda_kd_late_multiplier", 0.7))
        return base_lambda_fr * fr_multiplier, base_lambda_kd * kd_multiplier

    def train(self):
        """执行 receiver 二轮训练并返回结果指标。"""
        model_old, model_new = self._build_models()
        receiver_dir = get_receiver_dir(self.args, self.receiver_agent)
        method_tag = getattr(self.args, "packet_method", "dsdm").lower()
        if bool(getattr(self.args, "use_generalist_logits", False)):
            method_tag = f"{method_tag}_packet_generalist"
        elif bool(getattr(self.args, "use_logits", False)):
            method_tag = f"{method_tag}_logit"
        method_tag = f"{method_tag}_{getattr(self.args, 'init_mode', 'expert')}"
        checkpoint_dir = receiver_dir / "checkpoints" / method_tag
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model_old.state_dict(), checkpoint_dir / "before_social.pt")

        before = evaluate_receiver_model(self.args, model_old, self.receiver_agent, self.device)
        use_logits = bool(getattr(self.args, "use_logits", False))
        use_generalist_logits = bool(getattr(self.args, "use_generalist_logits", False))
        consumed = consume_manifest_packets(
            self.args,
            self.manifest_rows,
            require_logits=use_logits,
            require_generalist_logits=use_generalist_logits,
            receiver_agent=self.receiver_agent,
        )
        packets = consumed["packets"]
        if use_logits and consumed["sender_logits"] is None:
            raise ValueError("logits.enabled=true，但 packet 中没有 sender_logits，请先运行 attach_logits。")
        if use_generalist_logits and consumed["generalist_logits"] is None:
            raise ValueError("use_generalist_logits=true，但 packet 中没有 generalist_logits，请先运行 attach_generalist_logits。")
        loader = _build_balanced_loader(
            self.args,
            consumed["images"],
            consumed["labels"],
            consumed["sender_logits"] if use_logits else None,
            consumed["sender_logit_class_ids"] if use_logits else None,
            consumed["sender_agents"],
            consumed["generalist_logits"] if use_generalist_logits else None,
        )
        receiver_lr = float(getattr(self.args, "receiver_lr", self.args.lr))
        receiver_epochs = int(getattr(self.args, "receiver_epochs", self.args.epochs))
        init_mode = getattr(self.args, "init_mode", "expert")
        use_fr = bool(getattr(self.args, "use_fr", init_mode == "expert"))
        base_lambda_fr = float(getattr(self.args, "lambda_fr", 0.05)) if use_fr else 0.0
        self.kd_temperature = float(getattr(self.args, "kd_temperature", 2.0))
        base_lambda_kd = float(getattr(self.args, "lambda_kd", 0.0)) if (use_logits or use_generalist_logits) else 0.0
        kd_mix_beta = float(getattr(self.args, "kd_mix_beta", 0.5)) if use_generalist_logits else 0.0
        # 改进 loss 控制参数（opt-in，默认与原始行为完全相同）
        fr_loss_type = str(getattr(self.args, "fr_loss_type", "mse")).lower()
        use_fr_feat = bool(getattr(self.args, "use_fr_feat", False))
        alpha_fr_feat = float(getattr(self.args, "alpha_fr_feat", 0.5))
        use_self_kd = bool(getattr(self.args, "use_self_kd", False))
        lambda_self_kd = float(getattr(self.args, "lambda_self_kd", 0.3)) if use_self_kd else 0.0
        optimizer = optim.SGD(model_new.parameters(), lr=receiver_lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        scheduler = self._build_receiver_scheduler(optimizer, receiver_epochs)
        criterion = nn.CrossEntropyLoss()
        last_cls = 0.0
        last_fr = 0.0
        last_kd = 0.0
        last_sender_kd = 0.0
        last_generalist_kd = 0.0
        last_self_kd = 0.0
        model_new.train()
        if bool(getattr(self.args, "freeze_bn_stats", False)):
            _freeze_batchnorm_stats(model_new)
        for epoch in range(receiver_epochs):
            lambda_fr, lambda_kd = self._current_loss_weights(epoch, receiver_epochs, base_lambda_fr, base_lambda_kd)
            for batch in loader:
                if use_logits and use_generalist_logits:
                    batch_images, batch_labels, batch_teacher_logits, batch_teacher_class_ids, batch_generalist_logits, batch_sender_agents = batch
                    batch_teacher_logits = batch_teacher_logits.to(self.device)
                    batch_teacher_class_ids = batch_teacher_class_ids.to(self.device)
                    batch_generalist_logits = batch_generalist_logits.to(self.device)
                    batch_sender_agents = batch_sender_agents.to(self.device)
                elif use_logits:
                    batch_images, batch_labels, batch_teacher_logits, batch_teacher_class_ids, batch_sender_agents = batch
                    batch_teacher_logits = batch_teacher_logits.to(self.device)
                    batch_teacher_class_ids = batch_teacher_class_ids.to(self.device)
                    batch_sender_agents = batch_sender_agents.to(self.device)
                    batch_generalist_logits = None
                elif use_generalist_logits:
                    batch_images, batch_labels, batch_generalist_logits, batch_sender_agents = batch
                    batch_generalist_logits = batch_generalist_logits.to(self.device)
                    batch_sender_agents = batch_sender_agents.to(self.device)
                    batch_teacher_logits = None
                    batch_teacher_class_ids = None
                else:
                    batch_images, batch_labels = batch
                    batch_sender_agents = None
                    batch_teacher_logits = None
                    batch_teacher_class_ids = None
                    batch_generalist_logits = None
                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.to(self.device)
                batch_images = _augment_cifar_batch(batch_images, self.args)
                optimizer.zero_grad()
                logits = model_new(batch_images)
                loss_cls = criterion(logits, batch_labels)
                # FR loss：通过 fr_loss_type 选择 mse（原始）或 kl（改进）
                if use_fr:
                    if fr_loss_type == "kl":
                        loss_fr = self._compute_fr_loss_kl(model_old, model_new, batch_images, batch_labels)
                    else:
                        loss_fr = self._compute_fr_loss(model_old, model_new, batch_images, batch_labels)
                    if use_fr_feat:
                        loss_fr = loss_fr + alpha_fr_feat * self._compute_fr_feat_loss(model_old, model_new, batch_images, batch_labels)
                else:
                    loss_fr = torch.tensor(0.0, device=self.device)
                # Self-KD loss（全 batch 旧模型软标签约束）
                loss_self_kd = self._compute_self_kd_loss(model_old, model_new, batch_images) if use_self_kd else torch.tensor(0.0, device=self.device)
                loss_sender_kd = (
                    self._compute_kd_loss(logits, batch_teacher_logits, batch_teacher_class_ids, batch_sender_agents)
                    if use_logits
                    else torch.tensor(0.0, device=self.device)
                )
                loss_generalist_kd = (
                    self._compute_generalist_kd_loss(logits, batch_generalist_logits, batch_sender_agents)
                    if use_generalist_logits
                    else torch.tensor(0.0, device=self.device)
                )
                if use_generalist_logits and use_logits:
                    loss_kd = (1.0 - kd_mix_beta) * loss_sender_kd + kd_mix_beta * loss_generalist_kd
                elif use_generalist_logits:
                    loss_kd = loss_generalist_kd
                else:
                    loss_kd = loss_sender_kd
                loss = loss_cls + lambda_fr * loss_fr + lambda_kd * loss_kd + lambda_self_kd * loss_self_kd
                loss.backward()
                optimizer.step()
                last_cls = float(loss_cls.detach().cpu())
                last_fr = float(loss_fr.detach().cpu())
                last_kd = float(loss_kd.detach().cpu())
                last_sender_kd = float(loss_sender_kd.detach().cpu())
                last_generalist_kd = float(loss_generalist_kd.detach().cpu())
                last_self_kd = float(loss_self_kd.detach().cpu())
            if scheduler is not None:
                scheduler.step()
        torch.save(model_new.state_dict(), checkpoint_dir / "after_social.pt")
        after = evaluate_receiver_model(self.args, model_new, self.receiver_agent, self.device)
        external_raw = sum(p["raw_images"] for p in packets if p["sender_agent"] != self.receiver_agent)
        self_real_images = sum(p["num_images"] for p in packets if p.get("source") == "self_real")
        external_logit_bytes = (
            sum(p["sender_logit_bytes"] for p in packets if p["sender_agent"] != self.receiver_agent)
            if use_logits
            else 0
        )
        external_generalist_logit_bytes = (
            sum(p["generalist_logit_bytes"] for p in packets if p["sender_agent"] != self.receiver_agent)
            if use_generalist_logits
            else 0
        )
        base_method = getattr(self.args, "packet_method", "dsdm").upper()
        if use_generalist_logits:
            method = f"{base_method}_PACKET_GENERALIST"
        elif use_logits:
            method = f"{base_method}_LOGIT"
        else:
            method = base_method
        return {
            "receiver_agent": self.receiver_agent,
            "receiver_model": getattr(self.args, "model_name", self.model_split[self.receiver_agent]),
            "expert_classes": ",".join(str(c) for c in self.expert_classes),
            "packet_method": getattr(self.args, "packet_method", "dsdm"),
            "method": method,
            "init_mode": init_mode,
            "self_data_mode": getattr(self.args, "self_data_mode", "packet"),
            "self_real_per_class": int(getattr(self.args, "self_real_per_class", 0) or 0),
            "use_fr": str(use_fr).lower(),
            "lambda_fr": base_lambda_fr,
            "use_logits": str(use_logits).lower(),
            "communication_mode": getattr(self.args, "communication_mode", "direct"),
            "use_generalist_logits": str(use_generalist_logits).lower(),
            "kd_mix_beta": kd_mix_beta,
            "lambda_kd": base_lambda_kd,
            "kd_temperature": self.kd_temperature,
            "ipc": int(self.args.ipc),
            "self_real_images": int(self_real_images),
            "external_comm_images": int(external_raw),
            "external_comm_logit_bytes": int(external_logit_bytes),
            "external_comm_generalist_logit_bytes": int(external_generalist_logit_bytes),
            "acc_global_before": before["acc_global"],
            "acc_expert_before": before["acc_expert"],
            "acc_global_after": after["acc_global"],
            "acc_expert_after": after["acc_expert"],
            "acc_new_after": after["acc_new"],
            "forgetting": before["acc_expert"] - after["acc_expert"],
            "loss_cls": last_cls,
            "loss_fr": last_fr,
            "loss_kd": last_kd,
            "loss_sender_kd": last_sender_kd,
            "loss_generalist_kd": last_generalist_kd,
            "loss_self_kd": last_self_kd,
            "fr_loss_type": fr_loss_type,
            "use_fr_feat": str(use_fr_feat).lower(),
            "use_self_kd": str(use_self_kd).lower(),
            "receiver_augment": str(bool(getattr(self.args, "receiver_augment", False))).lower(),
            "freeze_bn_stats": str(bool(getattr(self.args, "freeze_bn_stats", False))).lower(),
        }
