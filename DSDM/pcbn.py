import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PCBNRegularizer:
    """按类别匹配 BatchNorm 输入统计的可选正则项。"""

    def __init__(self, args, logger=None):
        """读取 PCBN 配置并准备 hook 状态。"""
        self.enabled = bool(getattr(args, "pcbn_enabled", False))
        self.weight = float(getattr(args, "pcbn_weight", 0.0))
        self.layers = self._parse_layers(getattr(args, "pcbn_layers", "all"))
        self.normalize_layers = bool(getattr(args, "pcbn_normalize_layers", True))
        self.logger = logger
        self.handles = []
        self.records = []
        self.collecting = False
        self.layer_names = []
        self.logged = False
        if not math.isfinite(self.weight):
            raise ValueError(f"PCBN weight must be finite, got {self.weight}")
        if self.weight <= 0:
            self.enabled = False

    @staticmethod
    def _parse_layers(layers):
        """解析 PCBN 层选择，None 表示使用全部 BN 层。"""
        if layers is None:
            return None
        if isinstance(layers, str):
            if layers.strip().lower() in {"", "all", "*"}:
                return None
            items = [item.strip() for item in layers.split(",") if item.strip()]
        elif isinstance(layers, (list, tuple, set)):
            items = [str(item).strip() for item in layers if str(item).strip()]
        else:
            items = [str(layers).strip()]
        return set(items) if items else None

    def _use_layer(self, name, index):
        """判断指定 BN 层是否参与 PCBN 正则。"""
        if self.layers is None:
            return True
        return str(index) in self.layers or name in self.layers

    def _hook(self, module, inputs):
        """在 BN 前向前记录输入激活。"""
        if self.collecting and inputs:
            self.records.append(inputs[0])

    def attach(self, model):
        """给当前 guide model 注册 BN 输入 hook。"""
        self.close()
        if not self.enabled:
            return 0
        bn_index = 0
        matched_tokens = set()
        for name, module in model.named_modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                if self._use_layer(name, bn_index):
                    self.handles.append(module.register_forward_pre_hook(self._hook))
                    self.layer_names.append(name)
                    if self.layers is not None:
                        if name in self.layers:
                            matched_tokens.add(name)
                        if str(bn_index) in self.layers:
                            matched_tokens.add(str(bn_index))
                bn_index += 1
        if self.layers is not None:
            unmatched = self.layers - matched_tokens
            if unmatched:
                self.close()
                raise ValueError(f"PCBN layers did not match BatchNorm modules: {sorted(unmatched)}")
        if not self.handles:
            raise RuntimeError("PCBN is enabled but the guide model has no selected BatchNorm layers")
        if self.logger is not None and not self.logged:
            self.logger(
                f"[PCBN] enabled weight={self.weight} "
                f"layers={self.layer_names if self.layers is not None else 'all'} "
                f"hooked={len(self.handles)}"
            )
            self.logged = True
        return len(self.handles)

    def close(self):
        """移除当前 guide model 上注册的 hook。"""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.records = []
        self.collecting = False
        self.layer_names = []

    @staticmethod
    def _stats(activation, detach=False):
        """计算单个 BN 输入激活的通道均值和方差。"""
        dims = [0] + list(range(2, activation.dim()))
        mean = activation.mean(dim=dims)
        var = activation.var(dim=dims, unbiased=False)
        if detach:
            mean = mean.detach()
            var = var.detach()
        return mean, var

    def _collect_stats(self, model, images, detach=False):
        """前向一次模型并收集所有目标 BN 层统计。"""
        self.records = []
        self.collecting = True
        try:
            if detach:
                with torch.no_grad():
                    model(images)
            else:
                model(images)
            stats = [self._stats(record, detach=detach) for record in self.records]
            if len(stats) != len(self.handles):
                raise RuntimeError(
                    "PCBN hook collection mismatch: "
                    f"expected {len(self.handles)} activations, collected {len(stats)}"
                )
            return stats
        finally:
            self.collecting = False

    def loss(self, model, real_images, syn_images):
        """计算真实类 batch 与合成类 batch 的 PCBN 匹配损失。"""
        if not self.enabled:
            return None
        if not self.handles:
            self.attach(model)

        real_stats = self._collect_stats(model, real_images, detach=True)
        syn_stats = self._collect_stats(model, syn_images, detach=False)
        if len(real_stats) != len(syn_stats):
            raise RuntimeError(
                f"PCBN real/synthetic statistic mismatch: {len(real_stats)} != {len(syn_stats)}"
            )

        loss = syn_images.new_tensor(0.0)
        for (real_mean, real_var), (syn_mean, syn_var) in zip(real_stats, syn_stats):
            loss = loss + F.mse_loss(syn_mean, real_mean) + F.mse_loss(syn_var, real_var)
        if self.normalize_layers:
            loss = loss / len(real_stats)
        return loss * self.weight
