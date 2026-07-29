from pathlib import Path
import math
import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from agent_data import get_agent_class_split, get_agent_dir, get_agent_model_split, get_num_classes, get_receiver_dir
from dkp_receiver import CyclingLoader, build_complete_balanced_loader, mean_loss_totals, supervised_contrastive_loss
from packet_consumer import (
    consume_external_manifest_packets,
    consume_manifest_packets,
    consume_receiver_manifest_packet,
    load_receiver_local_real_data,
)
from social_metrics import evaluate_receiver_model


_DKP_ABLATION_VARIANT = re.compile(r"^ablation_fr([01])_kd([01])_sc([01])$")


def resolve_dkp_loss_switches(variant, explicit_switches=None):
    """Resolve endpoint or diagnostic-ablation loss switches without inference from weights."""
    normalized = str(variant).strip().lower()
    if normalized == "ce_only":
        resolved = {"fr": False, "kd": False, "supcon": False}
    elif normalized == "full":
        resolved = {"fr": True, "kd": True, "supcon": True}
    else:
        match = _DKP_ABLATION_VARIANT.fullmatch(normalized)
        if match is None:
            raise ValueError(f"unsupported dkp_variant: {normalized}")
        resolved = {
            "fr": match.group(1) == "1",
            "kd": match.group(2) == "1",
            "supcon": match.group(3) == "1",
        }
        if resolved in (
            {"fr": False, "kd": False, "supcon": False},
            {"fr": True, "kd": True, "supcon": True},
        ):
            raise ValueError("loss-ablation variants must use ce_only/full for the 000/111 endpoints")

    if explicit_switches is not None:
        if not isinstance(explicit_switches, dict):
            raise ValueError("receiver.loss_switches must be a mapping")
        if set(explicit_switches) != {"fr", "kd", "supcon"}:
            raise ValueError("receiver.loss_switches must contain exactly fr/kd/supcon")
        if any(not isinstance(value, bool) for value in explicit_switches.values()):
            raise ValueError("receiver.loss_switches values must be booleans")
        if explicit_switches != resolved:
            raise ValueError(
                f"receiver.loss_switches={explicit_switches} conflicts with dkp_variant={normalized}"
            )
    return resolved


def resolve_class_balanced_ce_weights(expert_classes, num_classes):
    """Weight local/external CE by their shares of the global class universe."""
    num_classes = int(num_classes)
    class_ids = [int(class_id) for class_id in expert_classes]
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than one")
    if not class_ids or len(class_ids) != len(set(class_ids)):
        raise ValueError("expert_classes must be non-empty and unique")
    if min(class_ids) < 0 or max(class_ids) >= num_classes:
        raise ValueError("expert_classes fall outside the global class range")
    if len(class_ids) >= num_classes:
        raise ValueError("receiver must have at least one external class")
    local_weight = len(class_ids) / float(num_classes)
    external_weight = 1.0 - local_weight
    return local_weight, external_weight


def _ensure_dsdm_path():
    """确保 DSDM 源码目录可被导入。"""
    import sys

    root = Path(__file__).resolve().parent
    dsdm_root = root / "DSDM"
    if str(dsdm_root) not in sys.path:
        sys.path.append(str(dsdm_root))


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


def _augment_cifar_batch(images, args, force=False):
    """对已归一化的 CIFAR batch 应用逐样本 crop/flip。"""
    if not force and not bool(getattr(args, "receiver_augment", False)):
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
        scheduler_unit = str(getattr(self.args, "receiver_scheduler_unit", "epoch")).lower()
        if scheduler_unit not in {"epoch", "optimizer_step"}:
            raise ValueError(f"unsupported receiver scheduler unit: {scheduler_unit}")
        if scheduler_name in {"", "none"}:
            return None
        if scheduler_name == "multistep":
            if scheduler_unit == "optimizer_step":
                milestones = getattr(self.args, "receiver_scheduler_step_milestones", None)
                if not milestones:
                    raise ValueError(
                        "optimizer-step MultiStepLR requires receiver.scheduler_step_milestones"
                    )
            else:
                milestones = getattr(self.args, "receiver_scheduler_milestones", None)
                if not milestones:
                    milestones = [int(0.65 * receiver_epochs), int(0.85 * receiver_epochs)]
            milestones = [max(1, int(v)) for v in milestones]
            gamma = float(getattr(self.args, "receiver_scheduler_gamma", 0.2))
            return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        if scheduler_name == "cosine":
            if scheduler_unit == "optimizer_step":
                optimizer_steps = getattr(self.args, "receiver_optimizer_steps", None)
                if optimizer_steps is None:
                    raise ValueError(
                        "optimizer-step cosine scheduler requires receiver.optimizer_steps"
                    )
                horizon = int(optimizer_steps)
            else:
                horizon = int(receiver_epochs)
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=horizon)
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

    def _normalize_images(self, images):
        """Normalize raw [0, 1] packet or CIFAR images for the receiver model."""
        from data import MEANS, STDS

        mean = torch.tensor(MEANS[self.args.dataset], dtype=images.dtype).view(1, -1, 1, 1)
        std = torch.tensor(STDS[self.args.dataset], dtype=images.dtype).view(1, -1, 1, 1)
        return ((images.cpu() - mean) / std).float()

    def _forward_with_penultimate(self, model, images):
        """Capture the exact feature tensor consumed by the task classifier."""
        from models.cosine_classifier import get_output_classifier

        classifier = get_output_classifier(model)
        captured = []

        def capture_input(_module, inputs):
            captured.append(inputs[0])

        hook = classifier.register_forward_pre_hook(capture_input)
        try:
            logits = model(images)
        finally:
            hook.remove()
        if len(captured) != 1:
            raise RuntimeError(f"output classifier was called {len(captured)} times in one forward pass")
        feature = captured[0]
        if feature.ndim > 2:
            feature = feature.reshape(feature.shape[0], -1)
        return logits, feature

    def _initialize_external_prototypes(self, model_old, model_new, images, labels):
        """Initialize external head rows from deterministic decoded DKP features."""
        from models.cosine_classifier import (
            get_output_classifier,
            get_output_classifier_type,
            set_classifier_weight_rows,
        )

        num_classes = get_num_classes(self.args)
        local_ids = sorted(int(class_id) for class_id in self.expert_classes)
        external_ids = sorted(set(range(num_classes)) - set(local_ids))
        actual_external_ids = sorted(int(class_id) for class_id in torch.unique(labels).tolist())
        if actual_external_ids != external_ids:
            raise ValueError(
                "external DKP class set does not equal the receiver's non-local classes: "
                f"actual={actual_external_ids} expected={external_ids}"
            )

        expected_per_class = int(getattr(self.args, "prototype_decoded_per_class", 40))
        if expected_per_class <= 0:
            raise ValueError("prototype_decoded_per_class must be positive")
        classifier = get_output_classifier(model_new)
        classifier_type = get_output_classifier_type(model_new)
        if int(classifier.out_features) != num_classes:
            raise ValueError(
                f"receiver classifier output is {classifier.out_features}; expected {num_classes}"
            )
        local_index = torch.tensor(local_ids, device=classifier.weight.device, dtype=torch.long)
        local_weight_before = classifier.weight.detach().index_select(0, local_index).clone()
        local_bias_before = None
        if classifier.bias is not None:
            local_bias_before = classifier.bias.detach().index_select(0, local_index).clone()

        model_old.eval()
        prototypes = []
        prototype_batch_size = int(getattr(self.args, "prototype_batch_size", 128))
        with torch.no_grad():
            for class_id in external_ids:
                class_images = images[labels == class_id]
                if class_images.shape[0] != expected_per_class:
                    raise ValueError(
                        f"external class {class_id} has {class_images.shape[0]} decoded images; "
                        f"expected {expected_per_class}"
                    )
                feature_chunks = []
                for start in range(0, class_images.shape[0], prototype_batch_size):
                    batch = class_images[start:start + prototype_batch_size].to(self.device)
                    _, feature = self._forward_with_penultimate(model_old, batch)
                    feature_chunks.append(feature.detach())
                class_feature = torch.cat(feature_chunks, dim=0).mean(dim=0)
                if not torch.isfinite(class_feature).all() or class_feature.norm(p=2).item() <= 0.0:
                    raise ValueError(
                        f"prototype source feature for external class {class_id} must be finite and non-zero"
                    )
                prototype = nn.functional.normalize(class_feature, p=2, dim=0)
                if not torch.isfinite(prototype).all():
                    raise ValueError(f"prototype for external class {class_id} is not finite")
                prototypes.append(prototype)

        prototype_tensor = torch.stack(prototypes)
        if not torch.isfinite(prototype_tensor).all():
            raise ValueError("external prototypes contain non-finite values")
        external_index = torch.tensor(external_ids, device=classifier.weight.device, dtype=torch.long)
        if classifier_type == "cosine":
            set_classifier_weight_rows(model_new, external_ids, prototype_tensor)
            prototype_mode = "cosine_unit_weight_rows"
            alpha = None
            beta = None
        else:
            local_norms = local_weight_before.norm(p=2, dim=1)
            if not torch.isfinite(local_norms).all():
                raise ValueError("local linear classifier row norms contain non-finite values")
            alpha_tensor = local_norms.mean()
            if not torch.isfinite(alpha_tensor) or alpha_tensor.item() <= 0.0:
                raise ValueError(
                    "linear prototype alpha must be finite and positive; "
                    f"actual={alpha_tensor.item()}"
                )
            alpha = float(alpha_tensor.cpu())
            if local_bias_before is None:
                beta_tensor = torch.zeros((), device=classifier.weight.device, dtype=classifier.weight.dtype)
            else:
                if not torch.isfinite(local_bias_before).all():
                    raise ValueError("local linear classifier biases contain non-finite values")
                beta_tensor = local_bias_before.mean()
            if not torch.isfinite(beta_tensor):
                raise ValueError("linear prototype beta must be finite")
            beta = float(beta_tensor.cpu())
            values = prototype_tensor.to(
                device=classifier.weight.device,
                dtype=classifier.weight.dtype,
            ) * alpha_tensor
            with torch.no_grad():
                classifier.weight.index_copy_(0, external_index, values)
                if classifier.bias is not None:
                    classifier.bias.index_fill_(0, external_index, beta_tensor)
            prototype_mode = "linear_local_row_norm_bias_mean"

        local_weight_after = classifier.weight.detach().index_select(0, local_index)
        if not torch.equal(local_weight_before, local_weight_after):
            raise RuntimeError("prototype initialization overwrote local expert-class weights")
        if local_bias_before is not None:
            local_bias_after = classifier.bias.detach().index_select(0, local_index)
            if not torch.equal(local_bias_before, local_bias_after):
                raise RuntimeError("prototype initialization overwrote local expert-class biases")
        external_weight = classifier.weight.detach().index_select(0, external_index)
        if not torch.isfinite(external_weight).all():
            raise RuntimeError("prototype-initialized classifier weights are not finite")
        expected_value = 1.0 if classifier_type == "cosine" else alpha
        expected_norm = torch.full(
            (len(external_ids),),
            float(expected_value),
            device=external_weight.device,
            dtype=external_weight.dtype,
        )
        if not torch.allclose(external_weight.norm(p=2, dim=1), expected_norm, atol=1e-6, rtol=0):
            raise RuntimeError("prototype-initialized classifier weights have incorrect norms")
        if classifier.bias is not None:
            external_bias = classifier.bias.detach().index_select(0, external_index)
            if not torch.isfinite(external_bias).all():
                raise RuntimeError("prototype-initialized classifier biases are not finite")
            if classifier_type == "linear" and not torch.allclose(
                external_bias,
                torch.full_like(external_bias, beta),
                atol=1e-7,
                rtol=0,
            ):
                raise RuntimeError("prototype-initialized linear biases do not equal beta")
        external_norms = external_weight.norm(p=2, dim=1)
        self._prototype_init_stats = {
            "classifier_type": classifier_type,
            "mode": prototype_mode,
            "alpha": alpha,
            "beta": beta,
            "external_weight_norm_min": float(external_norms.min().cpu()),
            "external_weight_norm_max": float(external_norms.max().cpu()),
            "external_weight_norm_mean": float(external_norms.mean().cpu()),
            "local_rows_preserved": True,
            "local_bias_preserved": True,
        }
        return external_ids

    def _receiver_checkpoint_retention(self):
        """Resolve retention, defaulting to the historical three-checkpoint behavior."""
        from config_adapter import normalize_receiver_checkpoint_retention

        return normalize_receiver_checkpoint_retention(
            getattr(self.args, "receiver_checkpoint_retention", "all")
        )

    def _validate_dkp_streams(self, local, external, require_logits, self_packet=None):
        """Validate receiver-local and external stream invariants before training."""
        local_labels = local["labels"].long()
        local_ids = sorted(int(class_id) for class_id in torch.unique(local_labels).tolist())
        expected_local_ids = sorted(int(class_id) for class_id in self.expert_classes)
        if local_ids != expected_local_ids:
            raise ValueError(f"local real class set mismatch: actual={local_ids} expected={expected_local_ids}")
        actual_senders = sorted(int(sender) for sender in torch.unique(external["sender_agents"]).tolist())
        expected_senders = sorted(set(self.class_split) - {self.receiver_agent})
        if actual_senders != expected_senders:
            raise ValueError(
                f"external sender set mismatch: actual={actual_senders} expected={expected_senders}"
            )
        if not torch.isfinite(local["images"]).all() or not torch.isfinite(external["images"]).all():
            raise ValueError("receiver streams contain non-finite images")
        if self_packet is not None:
            self_labels = self_packet["labels"].long()
            self_ids, self_counts = torch.unique(
                self_labels, sorted=True, return_counts=True
            )
            actual_self_ids = [int(class_id) for class_id in self_ids.tolist()]
            if actual_self_ids != expected_local_ids:
                raise ValueError(
                    f"self DKP class set mismatch: actual={actual_self_ids} "
                    f"expected={expected_local_ids}"
                )
            expected_raw_per_class = int(
                getattr(self.args, "receiver_packet_raw_per_class", self.args.ipc)
            )
            if expected_raw_per_class <= 0:
                raise ValueError("receiver packet_raw_per_class must be positive")
            expected_raw = len(expected_local_ids) * expected_raw_per_class
            if int(self_packet["raw_images"]) != expected_raw:
                raise ValueError(
                    f"self DKP raw image count is {self_packet['raw_images']}; "
                    f"expected {expected_raw}"
                )
            expected_decoded_per_class = int(
                getattr(self.args, "prototype_decoded_per_class", 40)
            )
            expected_decoded = len(expected_local_ids) * expected_decoded_per_class
            if int(self_packet["num_images"]) != expected_decoded:
                raise ValueError(
                    f"self DKP decoded image count is {self_packet['num_images']}; "
                    f"expected {expected_decoded}"
                )
            if not torch.all(self_counts == expected_decoded_per_class):
                count_map = {
                    int(class_id): int(count)
                    for class_id, count in zip(self_ids.tolist(), self_counts.tolist())
                }
                raise ValueError(
                    f"self DKP decoded class counts are not {expected_decoded_per_class}: "
                    f"{count_map}"
                )
            if self_packet["images"].shape[0] != self_labels.shape[0]:
                raise ValueError("self DKP image and label rows do not align")
            if not torch.isfinite(self_packet["images"]).all():
                raise ValueError("self DKP contains non-finite images")
            if self_labels.min().item() < 0 or self_labels.max().item() >= get_num_classes(self.args):
                raise ValueError("self DKP labels fall outside the global classifier range")
        if require_logits:
            teacher_logits = external["sender_logits"]
            teacher_class_ids = external["sender_logit_class_ids"]
            if teacher_logits is None or teacher_class_ids is None:
                raise ValueError("full DKP-SL requires sender logits and sender class ids")
            if teacher_logits.shape != teacher_class_ids.shape:
                raise ValueError("sender logits and sender class ids must have identical shapes")
            if teacher_logits.shape[1] != len(self.expert_classes):
                raise ValueError(
                    f"sender logit dimension is {teacher_logits.shape[1]}; "
                    f"expected {len(self.expert_classes)}"
                )
            if not torch.isfinite(teacher_logits).all():
                raise ValueError("external sender logits contain non-finite values")
            if teacher_class_ids.min().item() < 0 or teacher_class_ids.max().item() >= get_num_classes(self.args):
                raise ValueError("external sender class ids fall outside the global classifier range")
            for sender_agent in actual_senders:
                sender_mask = external["sender_agents"] == sender_agent
                expected_class_ids = torch.tensor(
                    self.class_split[sender_agent],
                    dtype=torch.long,
                ).view(1, -1)
                actual_class_ids = teacher_class_ids[sender_mask]
                if actual_class_ids.shape[1] != expected_class_ids.shape[1] or not torch.equal(
                    actual_class_ids,
                    expected_class_ids.expand_as(actual_class_ids),
                ):
                    raise ValueError(
                        f"sender {sender_agent} logit class ids do not match its configured class order"
                    )
                sender_label_ids = sorted(
                    int(class_id)
                    for class_id in torch.unique(external["labels"][sender_mask]).tolist()
                )
                if sender_label_ids != sorted(self.class_split[sender_agent]):
                    raise ValueError(
                        f"sender {sender_agent} packet labels do not match its configured class set"
                    )

    def _receiver_expert_checkpoint_sha(self, require_manifest_match):
        """Resolve and validate the checkpoint shared by init and the frozen FR teacher."""
        from packet_integrity import (
            file_sha256,
            strict_packet_validation_enabled,
            validate_receiver_expert_provenance,
        )

        expert_path = get_agent_dir(self.args, self.receiver_agent) / "checkpoints" / "expert_model.pt"
        strict = strict_packet_validation_enabled(self.args)
        if not expert_path.is_file():
            if strict:
                raise FileNotFoundError(f"receiver expert checkpoint is missing: {expert_path}")
            return ""
        if strict:
            previous_use_logits = getattr(self.args, "use_logits", False)
            self.args.use_logits = bool(require_manifest_match)
            try:
                return validate_receiver_expert_provenance(
                    self.args,
                    self.receiver_agent,
                    self.manifest_rows,
                    expert_path,
                )
            finally:
                self.args.use_logits = previous_use_logits
        return file_sha256(expert_path)

    def _set_receiver_seed(self):
        """Seed every RNG used by the receiver and return the resolved run seed."""
        resolved_seed = int(getattr(self.args, "seed", 0)) + 100000 + self.receiver_agent
        random.seed(resolved_seed)
        np.random.seed(resolved_seed % (2 ** 32))
        torch.manual_seed(resolved_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(resolved_seed)
        return resolved_seed

    def _train_dkp_sl_v1(self):
        """Run the opt-in two-stream DKP-SL receiver protocol."""
        from models.cosine_classifier import sgd_parameter_groups
        from output_manager import atomic_torch_save, atomic_write_json
        from packet_integrity import file_sha256

        variant = str(getattr(self.args, "dkp_variant", "full")).strip().lower()
        loss_switches = resolve_dkp_loss_switches(
            variant,
            getattr(self.args, "dkp_loss_switches", None),
        )
        use_fr = loss_switches["fr"]
        use_kd = loss_switches["kd"]
        use_supcon = loss_switches["supcon"]
        require_logits = use_kd
        if bool(getattr(self.args, "use_logits", False)) is not use_kd:
            raise ValueError(
                f"dkp_variant={variant} requires communication.use_sender_logits={str(use_kd).lower()}"
            )
        if bool(getattr(self.args, "use_generalist_logits", False)):
            raise ValueError("DKP-SL v1 does not use generalist logits")
        if str(getattr(self.args, "init_mode", "expert")).lower() != "expert":
            raise ValueError("DKP-SL v1 requires the receiver expert checkpoint as initialization")
        local_ce_source = str(
            getattr(self.args, "receiver_local_ce_source", "real")
        ).strip().lower()
        if local_ce_source not in {
            "real",
            "packet",
            "real_packet_50_50",
            "real_packet_mix",
        }:
            raise ValueError(f"unsupported receiver local_ce_source: {local_ce_source}")
        uses_self_packet = local_ce_source in {
            "packet",
            "real_packet_50_50",
            "real_packet_mix",
        }
        if local_ce_source == "real":
            local_ce_real_fraction = 1.0
        elif local_ce_source == "packet":
            local_ce_real_fraction = 0.0
        elif local_ce_source == "real_packet_50_50":
            local_ce_real_fraction = 0.5
        else:
            local_ce_real_fraction = float(
                getattr(self.args, "receiver_local_ce_real_fraction", float("nan"))
            )
            if not math.isfinite(local_ce_real_fraction) or not 0.0 < local_ce_real_fraction < 1.0:
                raise ValueError(
                    "real_packet_mix requires a finite local CE real fraction strictly between 0 and 1"
                )
        ce_local_weight, ce_external_weight = resolve_class_balanced_ce_weights(
            self.expert_classes,
            get_num_classes(self.args),
        )

        receiver_seed = self._set_receiver_seed()
        receiver_expert_sha = self._receiver_expert_checkpoint_sha(
            require_manifest_match=require_logits,
        )
        model_old, model_new = self._build_models()
        receiver_dir = get_receiver_dir(self.args, self.receiver_agent)
        checkpoint_dir = receiver_dir / "checkpoints" / f"dkp_sl_v1_{variant}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_retention = self._receiver_checkpoint_retention()
        checkpoint_artifacts = {}
        if checkpoint_retention == "all":
            before_path = atomic_torch_save(
                model_old.state_dict(), checkpoint_dir / "before_social.pt"
            )
            checkpoint_artifacts["before_social"] = {
                "path": str(before_path.resolve()),
                "sha256": file_sha256(before_path),
            }
        before = evaluate_receiver_model(
            self.args,
            model_old,
            self.receiver_agent,
            self.device,
            expert_mask=True,
        )

        local = load_receiver_local_real_data(self.args)
        external = consume_external_manifest_packets(
            self.args,
            self.manifest_rows,
            self.receiver_agent,
            require_logits=require_logits,
        )
        self_packet = None
        self_packet_sha = ""
        if uses_self_packet:
            self_packet = consume_receiver_manifest_packet(
                self.args,
                self.manifest_rows,
                self.receiver_agent,
            )
            self_packet_sha = file_sha256(self_packet["packet_path"])
            manifest_packet_sha = self_packet["manifest_packet_sha256"]
            if manifest_packet_sha and manifest_packet_sha != self_packet_sha:
                raise ValueError(
                    "self DKP packet SHA-256 does not match its immutable manifest row"
                )
        self._validate_dkp_streams(
            local,
            external,
            require_logits=require_logits,
            self_packet=self_packet,
        )
        local_images = self._normalize_images(local["images"])
        external_images = self._normalize_images(external["images"])
        self_packet_images = (
            self._normalize_images(self_packet["images"])
            if self_packet is not None
            else None
        )
        external_ids = self._initialize_external_prototypes(
            model_old,
            model_new,
            external_images,
            external["labels"],
        )
        prototype_stats = dict(self._prototype_init_stats)
        if checkpoint_retention == "all":
            prototype_path = atomic_torch_save(
                model_new.state_dict(), checkpoint_dir / "after_prototype_init.pt"
            )
            checkpoint_artifacts["after_prototype_init"] = {
                "path": str(prototype_path.resolve()),
                "sha256": file_sha256(prototype_path),
            }

        local_batch_size = int(getattr(self.args, "receiver_local_batch_size", 64))
        external_batch_size = int(getattr(self.args, "receiver_external_batch_size", 64))
        self_packet_batch_size = int(
            getattr(self.args, "receiver_self_packet_batch_size", 64)
        )
        local_generator = torch.Generator().manual_seed(receiver_seed + 1)
        external_generator = torch.Generator().manual_seed(receiver_seed + 2)
        self_packet_generator = torch.Generator().manual_seed(receiver_seed + 3)
        local_loader = build_complete_balanced_loader(
            local_images,
            local["labels"],
            batch_size=local_batch_size,
            shuffle=True,
            generator=local_generator,
            pad_to_full_batch=True,
        )
        if require_logits:
            external_loader = build_complete_balanced_loader(
                external_images,
                external["labels"],
                external["sender_logits"].float(),
                external["sender_logit_class_ids"].long(),
                external["sender_agents"].long(),
                batch_size=external_batch_size,
                shuffle=True,
                generator=external_generator,
            )
        else:
            external_loader = build_complete_balanced_loader(
                external_images,
                external["labels"],
                external["sender_agents"].long(),
                batch_size=external_batch_size,
                shuffle=True,
                generator=external_generator,
            )
        external_cycle = CyclingLoader(external_loader)
        local_cycle = CyclingLoader(local_loader)
        self_packet_cycle = None
        if self_packet is not None:
            self_packet_loader = build_complete_balanced_loader(
                self_packet_images,
                self_packet["labels"],
                batch_size=self_packet_batch_size,
                shuffle=True,
                generator=self_packet_generator,
                pad_to_full_batch=True,
            )
            self_packet_cycle = CyclingLoader(self_packet_loader)

        receiver_lr = float(getattr(self.args, "receiver_lr", self.args.lr))
        receiver_epochs = int(getattr(self.args, "receiver_epochs", self.args.epochs))
        configured_optimizer_steps = getattr(
            self.args, "receiver_optimizer_steps", None
        )
        if configured_optimizer_steps is None:
            target_optimizer_steps = receiver_epochs * len(local_loader)
        else:
            target_optimizer_steps = int(configured_optimizer_steps)
            if target_optimizer_steps <= 0:
                raise ValueError("receiver optimizer_steps must be positive")
        scheduler_unit = str(
            getattr(self.args, "receiver_scheduler_unit", "epoch")
        ).lower()
        lambda_fr = float(getattr(self.args, "lambda_fr", 0.2 if use_fr else 0.0))
        lambda_kd = float(getattr(self.args, "lambda_kd", 0.6 if use_kd else 0.0))
        lambda_sc = float(getattr(self.args, "lambda_sc", 0.1 if use_supcon else 0.0))
        for name, enabled, value in (
            ("FR", use_fr, lambda_fr),
            ("KD", use_kd, lambda_kd),
            ("SupCon", use_supcon, lambda_sc),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"lambda_{name} must be finite and non-negative")
            if enabled and value <= 0.0:
                raise ValueError(f"enabled {name} requires a positive loss weight")
            if not enabled and value != 0.0:
                raise ValueError(f"disabled {name} requires a zero loss weight")
        self.kd_temperature = float(getattr(self.args, "kd_temperature", 2.0))
        supcon_temperature = float(getattr(self.args, "supcon_temperature", 0.07))
        parameter_groups = sgd_parameter_groups(model_new, self.args.weight_decay)
        optimizer = optim.SGD(parameter_groups, lr=receiver_lr, momentum=self.args.momentum)
        scheduler = self._build_receiver_scheduler(optimizer, receiver_epochs)
        criterion = nn.CrossEntropyLoss()
        totals = {
            "loss": 0.0,
            "loss_cls": 0.0,
            "loss_ce_local": 0.0,
            "loss_ce_local_real": 0.0,
            "loss_ce_local_packet": 0.0,
            "loss_ce_external": 0.0,
            "loss_fr": 0.0,
            "loss_kd": 0.0,
            "loss_sc": 0.0,
        }
        optimizer_steps = 0
        training_started = time.perf_counter()
        model_new.train()
        if bool(getattr(self.args, "freeze_bn_stats", False)):
            _freeze_batchnorm_stats(model_new)
        while optimizer_steps < target_optimizer_steps:
                local_batch_images, local_batch_labels = local_cycle.next()
                external_batch = external_cycle.next()
                self_packet_batch = (
                    self_packet_cycle.next() if self_packet_cycle is not None else None
                )
                if require_logits:
                    (
                        external_batch_images,
                        external_batch_labels,
                        teacher_logits,
                        teacher_class_ids,
                        sender_agents,
                    ) = external_batch
                    teacher_logits = teacher_logits.to(self.device)
                    teacher_class_ids = teacher_class_ids.to(self.device)
                    sender_agents = sender_agents.to(self.device)
                else:
                    external_batch_images, external_batch_labels, _sender_agents = external_batch
                    teacher_logits = None
                    teacher_class_ids = None
                    sender_agents = None

                local_batch_images = local_batch_images.to(self.device)
                local_batch_labels = local_batch_labels.to(self.device)
                external_batch_images = external_batch_images.to(self.device)
                external_batch_labels = external_batch_labels.to(self.device)
                local_view = _augment_cifar_batch(local_batch_images, self.args, force=True)
                external_view1 = _augment_cifar_batch(external_batch_images, self.args, force=True)
                external_view2 = _augment_cifar_batch(external_batch_images, self.args, force=True)
                if self_packet_batch is not None:
                    self_packet_batch_images, self_packet_batch_labels = self_packet_batch
                    self_packet_batch_images = self_packet_batch_images.to(self.device)
                    self_packet_batch_labels = self_packet_batch_labels.to(self.device)
                    self_packet_view1 = _augment_cifar_batch(
                        self_packet_batch_images, self.args, force=True
                    )
                    self_packet_view2 = _augment_cifar_batch(
                        self_packet_batch_images, self.args, force=True
                    )
                else:
                    self_packet_view1 = None
                    self_packet_view2 = None
                    self_packet_batch_labels = None

                optimizer.zero_grad()
                local_logits = None
                local_feature = None
                if local_ce_source != "packet" or use_fr:
                    local_logits, local_feature = self._forward_with_penultimate(
                        model_new, local_view
                    )
                external_logits1, external_feature1 = self._forward_with_penultimate(model_new, external_view1)
                external_logits2, external_feature2 = self._forward_with_penultimate(model_new, external_view2)
                zero = torch.zeros((), device=self.device)
                if local_ce_source in {
                    "real",
                    "real_packet_50_50",
                    "real_packet_mix",
                }:
                    loss_ce_local_real = criterion(local_logits, local_batch_labels)
                else:
                    loss_ce_local_real = zero
                if self_packet_view1 is not None:
                    self_packet_logits1, _ = self._forward_with_penultimate(
                        model_new, self_packet_view1
                    )
                    self_packet_logits2, _ = self._forward_with_penultimate(
                        model_new, self_packet_view2
                    )
                    loss_ce_local_packet = 0.5 * (
                        criterion(self_packet_logits1, self_packet_batch_labels)
                        + criterion(self_packet_logits2, self_packet_batch_labels)
                    )
                else:
                    loss_ce_local_packet = zero
                if local_ce_source == "real":
                    loss_ce_local = loss_ce_local_real
                elif local_ce_source == "packet":
                    loss_ce_local = loss_ce_local_packet
                else:
                    loss_ce_local = (
                        local_ce_real_fraction * loss_ce_local_real
                        + (1.0 - local_ce_real_fraction) * loss_ce_local_packet
                    )
                loss_ce_external = 0.5 * (
                    criterion(external_logits1, external_batch_labels)
                    + criterion(external_logits2, external_batch_labels)
                )
                loss_cls = (
                    ce_local_weight * loss_ce_local
                    + ce_external_weight * loss_ce_external
                )

                if use_fr:
                    with torch.no_grad():
                        _, old_local_feature = self._forward_with_penultimate(model_old, local_view)
                    loss_fr = 1.0 - nn.functional.cosine_similarity(
                        local_feature,
                        old_local_feature,
                        dim=1,
                    ).mean()
                else:
                    loss_fr = zero
                if use_kd:
                    deterministic_logits = model_new(external_batch_images)
                    loss_kd = self._compute_kd_loss(
                        deterministic_logits,
                        teacher_logits,
                        teacher_class_ids,
                        sender_agents,
                    )
                else:
                    loss_kd = zero
                if use_supcon:
                    loss_sc = supervised_contrastive_loss(
                        external_feature1,
                        external_feature2,
                        external_batch_labels,
                        temperature=supcon_temperature,
                    )
                else:
                    loss_sc = zero
                loss = loss_cls + lambda_fr * loss_fr + lambda_kd * loss_kd + lambda_sc * loss_sc
                loss.backward()
                optimizer.step()

                optimizer_steps += 1
                if scheduler is not None and scheduler_unit == "optimizer_step":
                    scheduler.step()
                step_values = {
                    "loss": loss,
                    "loss_cls": loss_cls,
                    "loss_ce_local": loss_ce_local,
                    "loss_ce_local_real": loss_ce_local_real,
                    "loss_ce_local_packet": loss_ce_local_packet,
                    "loss_ce_external": loss_ce_external,
                    "loss_fr": loss_fr,
                    "loss_kd": loss_kd,
                    "loss_sc": loss_sc,
                }
                for name, value in step_values.items():
                    totals[name] += float(value.detach().cpu())
                if (
                    scheduler is not None
                    and scheduler_unit == "epoch"
                    and optimizer_steps % len(local_loader) == 0
                ):
                    scheduler.step()
        if optimizer_steps != target_optimizer_steps:
            raise RuntimeError(
                f"receiver executed {optimizer_steps} optimizer steps; "
                f"expected {target_optimizer_steps}"
            )
        training_seconds = time.perf_counter() - training_started
        means = mean_loss_totals(totals, optimizer_steps)

        after_social_path = atomic_torch_save(
            model_new.state_dict(), checkpoint_dir / "after_social.pt"
        )
        after_social_sha = file_sha256(after_social_path)
        checkpoint_artifacts["after_social"] = {
            "path": str(after_social_path.resolve()),
            "sha256": after_social_sha,
        }
        after = evaluate_receiver_model(self.args, model_new, self.receiver_agent, self.device)
        external_raw = sum(packet["raw_images"] for packet in external["packets"])
        external_logit_bytes = (
            sum(packet["sender_logit_bytes"] for packet in external["packets"])
            if require_logits
            else 0
        )
        if variant == "ce_only":
            method = "DKP_CE_ONLY"
        elif variant == "full":
            method = "DKP_SL"
        else:
            method = "DKP_SL_ABLATION"
        result = {
            "receiver_agent": self.receiver_agent,
            "receiver_model": getattr(self.args, "model_name", self.model_split[self.receiver_agent]),
            "expert_classes": ",".join(str(class_id) for class_id in self.expert_classes),
            "packet_method": getattr(self.args, "packet_method", "dsdm"),
            "method": method,
            "protocol": "dkp_sl_v1",
            "dkp_variant": variant,
            "init_mode": "expert",
            "self_data_mode": "real",
            "local_ce_source": local_ce_source,
            "local_ce_real_fraction": local_ce_real_fraction,
            "ce_local_weight": ce_local_weight,
            "ce_external_weight": ce_external_weight,
            "self_real_per_class": 0,
            "self_packet_raw_images": int(self_packet["raw_images"]) if self_packet else 0,
            "self_packet_decoded_images": int(self_packet["num_images"]) if self_packet else 0,
            "self_packet_path": self_packet["packet_path"] if self_packet else "",
            "self_packet_sha256": self_packet_sha,
            "use_fr": str(use_fr).lower(),
            "lambda_fr": lambda_fr,
            "use_logits": str(use_kd).lower(),
            "communication_mode": getattr(self.args, "communication_mode", "all_share_once"),
            "use_generalist_logits": "false",
            "kd_mix_beta": 0.0,
            "lambda_kd": lambda_kd,
            "kd_temperature": self.kd_temperature,
            "lambda_sc": lambda_sc,
            "supcon_temperature": supcon_temperature,
            "ipc": int(self.args.ipc),
            "self_real_images": int(local["num_images"]),
            "external_comm_images": int(external_raw),
            "external_comm_logit_bytes": int(external_logit_bytes),
            "external_comm_generalist_logit_bytes": 0,
            "acc_global_before": before["acc_global"],
            "acc_expert_before": before["acc_expert"],
            "acc_global_after": after["acc_global"],
            "acc_expert_after": after["acc_expert"],
            "acc_new_after": after["acc_new"],
            "forgetting": before["acc_expert"] - after["acc_expert"],
            **means,
            "loss_sender_kd": means["loss_kd"],
            "loss_generalist_kd": 0.0,
            "loss_self_kd": 0.0,
            "optimizer_steps": int(optimizer_steps),
            "target_optimizer_steps": int(target_optimizer_steps),
            "scheduler_unit": scheduler_unit,
            "scheduler_step_milestones": ";".join(
                str(value)
                for value in getattr(
                    self.args, "receiver_scheduler_step_milestones", []
                )
            ),
            "training_seconds": float(training_seconds),
            "prototype_initialized_classes": len(external_ids),
            "classifier_type": prototype_stats["classifier_type"],
            "prototype_init_mode": prototype_stats["mode"],
            "prototype_alpha": "" if prototype_stats["alpha"] is None else prototype_stats["alpha"],
            "prototype_beta": "" if prototype_stats["beta"] is None else prototype_stats["beta"],
            "prototype_weight_norm_min": prototype_stats["external_weight_norm_min"],
            "prototype_weight_norm_max": prototype_stats["external_weight_norm_max"],
            "checkpoint_retention": checkpoint_retention,
            "after_social_checkpoint_sha256": after_social_sha,
            "receiver_seed": receiver_seed,
            "receiver_init_checkpoint_sha256": receiver_expert_sha,
            "fr_teacher_checkpoint_sha256": receiver_expert_sha,
            "fr_loss_type": "feature_cosine",
            "use_fr_feat": str(use_fr).lower(),
            "use_self_kd": "false",
            "receiver_augment": "true",
            "freeze_bn_stats": str(bool(getattr(self.args, "freeze_bn_stats", False))).lower(),
        }
        provenance_path = checkpoint_dir / "receiver_provenance.json"
        provenance = {
            "protocol": "dkp_sl_v1",
            "dkp_variant": variant,
            "local_ce_source": local_ce_source,
            "local_ce_real_fraction": local_ce_real_fraction,
            "loss_switches": loss_switches,
            "receiver_agent": self.receiver_agent,
            "receiver_model": result["receiver_model"],
            "receiver_seed": receiver_seed,
            "receiver_init_checkpoint_sha256": receiver_expert_sha,
            "fr_teacher_checkpoint_sha256": receiver_expert_sha,
            "classifier_type": prototype_stats["classifier_type"],
            "prototype_initialization": {
                **prototype_stats,
                "initialized_classes": [int(class_id) for class_id in external_ids],
            },
            "checkpoint_retention": checkpoint_retention,
            "checkpoint_artifacts": checkpoint_artifacts,
            "training_schedule": {
                "target_optimizer_steps": int(target_optimizer_steps),
                "actual_optimizer_steps": int(optimizer_steps),
                "scheduler_unit": scheduler_unit,
                "scheduler": str(
                    getattr(self.args, "receiver_scheduler", "none")
                ),
                "scheduler_step_milestones": [
                    int(value)
                    for value in getattr(
                        self.args, "receiver_scheduler_step_milestones", []
                    )
                ],
            },
            "classification_balance": {
                "num_global_classes": int(get_num_classes(self.args)),
                "num_local_classes": len(self.expert_classes),
                "num_external_classes": int(get_num_classes(self.args)) - len(self.expert_classes),
                "ce_local_weight": ce_local_weight,
                "ce_external_weight": ce_external_weight,
            },
            "local_data": {
                "real_images": int(local["num_images"]),
                "self_packet_raw_images": (
                    int(self_packet["raw_images"]) if self_packet else 0
                ),
                "self_packet_decoded_images": (
                    int(self_packet["num_images"]) if self_packet else 0
                ),
                "self_packet_path": (
                    self_packet["packet_path"] if self_packet else ""
                ),
                "self_packet_sha256": self_packet_sha,
                "counted_as_external_communication": False,
                "used_for_kd": False,
                "used_for_supcon": False,
            },
            "statistics": {
                "metrics_before": before,
                "metrics_after": after,
                "loss_means": means,
                "optimizer_steps": int(optimizer_steps),
                "training_seconds": float(training_seconds),
                "external_comm_images": int(external_raw),
                "external_comm_logit_bytes": int(external_logit_bytes),
            },
        }
        atomic_write_json(provenance, provenance_path)
        result["receiver_provenance_path"] = str(provenance_path.resolve())
        result["receiver_provenance_sha256"] = file_sha256(provenance_path)
        return result

    def train(self):
        """执行 receiver 二轮训练并返回结果指标。"""
        protocol = str(getattr(self.args, "receiver_protocol", "legacy")).strip().lower()
        if protocol == "dkp_sl_v1":
            return self._train_dkp_sl_v1()
        if protocol not in {"", "legacy"}:
            raise ValueError(f"unsupported receiver_protocol: {protocol}")
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
