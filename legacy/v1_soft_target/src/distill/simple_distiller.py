"""
DSDM-style image distillation utilities for social packets.

The packet distiller keeps the social-learning pipeline unchanged: sender
anchors produce distilled images here, and run_build_packets.py later attaches
soft targets. This module follows the DSDM matching flow with topology and
random-walk losses removed:
- class-wise real/synthetic sampling
- differentiable augmentation
- feature prototype matching
- feature covariance matching
- historical prototype smoothing
"""

from math import ceil
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from tqdm import tqdm


def select_ipc_indices(dataset, class_ids, ipc: int):
    return select_n_per_class_indices(dataset, class_ids, ipc)


def select_n_per_class_indices(dataset, class_ids, n_per_class: int):
    selected = []
    counts = {class_id: 0 for class_id in class_ids}

    for idx, target in enumerate(dataset.targets):
        if target in counts and counts[target] < n_per_class:
            selected.append(idx)
            counts[target] += 1
        if all(count == n_per_class for count in counts.values()):
            break

    missing = {class_id: n_per_class - count for class_id, count in counts.items() if count < n_per_class}
    if missing:
        raise RuntimeError(f"not enough samples for n_per_class={n_per_class}: {missing}")

    return selected


def stack_subset_samples(dataset, indices):
    images = []
    labels = []
    subset = Subset(dataset, indices)
    for image, label in subset:
        images.append(image)
        labels.append(label)

    return torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.long)


def build_raw_images(dataset, class_ids, ipc: int):
    indices = select_ipc_indices(dataset, class_ids, ipc)
    return stack_subset_samples(dataset, indices)


def freeze_model(model) -> None:
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)


class DiffAug:
    """Differentiable augmentation from DSDM without project-specific imports."""

    def __init__(
        self,
        strategy: str = "color_crop_cutout",
        batch: bool = True,
        ratio_cutout: float = 0.5,
        single: bool = False,
    ):
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = ratio_cutout
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5
        self.batch = batch

        self.aug = strategy != "" and strategy.lower() != "none"
        self.strategy = []
        self.flip = False
        self.color = False
        self.cutout = False
        if self.aug:
            for aug in strategy.lower().split("_"):
                if aug == "flip" and not single:
                    self.flip = True
                elif aug == "color" and not single:
                    self.color = True
                elif aug == "cutout" and not single:
                    self.cutout = True
                else:
                    self.strategy.append(aug)

        self.aug_fn = {
            "color": [self.brightness_fn, self.saturation_fn, self.contrast_fn],
            "crop": [self.crop_fn],
            "cutout": [self.cutout_fn],
            "flip": [self.flip_fn],
            "scale": [self.scale_fn],
            "rotate": [self.rotate_fn],
            "translate": [self.translate_fn],
        }

    def __call__(self, x, single_aug: bool = True, seed: int = -1):
        if not self.aug:
            return x

        if self.flip:
            self.set_seed(seed)
            x = self.flip_fn(x, self.batch)
        if self.color:
            for fn in self.aug_fn["color"]:
                self.set_seed(seed)
                x = fn(x, self.batch)
        if self.strategy:
            if single_aug:
                aug_name = self.strategy[np.random.randint(len(self.strategy))]
                for fn in self.aug_fn[aug_name]:
                    self.set_seed(seed)
                    x = fn(x, self.batch)
            else:
                for aug_name in self.strategy:
                    for fn in self.aug_fn[aug_name]:
                        self.set_seed(seed)
                        x = fn(x, self.batch)
        if self.cutout:
            self.set_seed(seed)
            x = self.cutout_fn(x, self.batch)
        return x.contiguous()

    def set_seed(self, seed: int):
        if seed > 0:
            np.random.seed(seed)
            torch.random.manual_seed(seed)

    def scale_fn(self, x, batch=True):
        ratio = self.ratio_scale
        if batch:
            sx = np.random.uniform() * (ratio - 1.0 / ratio) + 1.0 / ratio
            sy = np.random.uniform() * (ratio - 1.0 / ratio) + 1.0 / ratio
            theta = torch.tensor([[sx, 0, 0], [0, sy, 0]], dtype=torch.float, device=x.device)
            theta = theta.expand(x.shape[0], 2, 3)
        else:
            sx = np.random.uniform(size=x.shape[0]) * (ratio - 1.0 / ratio) + 1.0 / ratio
            sy = np.random.uniform(size=x.shape[0]) * (ratio - 1.0 / ratio) + 1.0 / ratio
            theta = torch.tensor(
                [[[sx[i], 0, 0], [0, sy[i], 0]] for i in range(x.shape[0])],
                dtype=torch.float,
                device=x.device,
            )
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)

    def rotate_fn(self, x, batch=True):
        ratio = self.ratio_rotate
        if batch:
            theta_value = (np.random.uniform() - 0.5) * 2 * ratio / 180 * float(np.pi)
            theta = torch.tensor(
                [[np.cos(theta_value), np.sin(-theta_value), 0], [np.sin(theta_value), np.cos(theta_value), 0]],
                dtype=torch.float,
                device=x.device,
            )
            theta = theta.expand(x.shape[0], 2, 3)
        else:
            theta_value = (np.random.uniform(size=x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
            theta = torch.tensor(
                [
                    [[np.cos(theta_value[i]), np.sin(-theta_value[i]), 0], [np.sin(theta_value[i]), np.cos(theta_value[i]), 0]]
                    for i in range(x.shape[0])
                ],
                dtype=torch.float,
                device=x.device,
            )
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)

    def flip_fn(self, x, batch=True):
        if batch:
            return x.flip(3) if np.random.uniform() < self.prob_flip else x
        randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
        return torch.where(randf < self.prob_flip, x.flip(3), x)

    def brightness_fn(self, x, batch=True):
        randb = np.random.uniform() if batch else torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        return x + (randb - 0.5) * self.brightness

    def saturation_fn(self, x, batch=True):
        x_mean = x.mean(dim=1, keepdim=True)
        rands = np.random.uniform() if batch else torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        return (x - x_mean) * (rands * self.saturation) + x_mean

    def contrast_fn(self, x, batch=True):
        x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
        randc = np.random.uniform() if batch else torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        return (x - x_mean) * (randc + self.contrast) + x_mean

    def translate_fn(self, x, batch=True):
        ratio = self.ratio_crop_pad
        shift_y = int(x.size(3) * ratio + 0.5)
        if batch:
            translation_y = np.random.randint(-shift_y, shift_y + 1)
        else:
            translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)

        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(x.size(2), dtype=torch.long, device=x.device),
            torch.arange(x.size(3), dtype=torch.long, device=x.device),
            indexing="ij",
        )
        grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
        x_pad = F.pad(x, (1, 1))
        return x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)

    def crop_fn(self, x, batch=True):
        ratio = self.ratio_crop_pad
        shift_x = int(x.size(2) * ratio + 0.5)
        shift_y = int(x.size(3) * ratio + 0.5)
        if batch:
            translation_x = np.random.randint(-shift_x, shift_x + 1)
            translation_y = np.random.randint(-shift_y, shift_y + 1)
        else:
            translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
            translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)

        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(x.size(2), dtype=torch.long, device=x.device),
            torch.arange(x.size(3), dtype=torch.long, device=x.device),
            indexing="ij",
        )
        grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
        grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
        x_pad = F.pad(x, (1, 1, 1, 1))
        return x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)

    def cutout_fn(self, x, batch=True):
        cutout_size = int(x.size(2) * self.ratio_cutout + 0.5), int(x.size(3) * self.ratio_cutout + 0.5)
        if batch:
            offset_x = np.random.randint(0, x.size(2) + (1 - cutout_size[0] % 2))
            offset_y = np.random.randint(0, x.size(3) + (1 - cutout_size[1] % 2))
        else:
            offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
            offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)

        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
            indexing="ij",
        )
        grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
        grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
        mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
        mask[grid_batch, grid_x, grid_y] = 0
        return x * mask.unsqueeze(1)


class ClassDatasetSampler:
    """Class-wise sampler matching DSDM's ClassDataLoader.class_sample API."""

    def __init__(self, dataset, class_ids: Sequence[int], device: torch.device):
        self.dataset = dataset
        self.class_ids = list(class_ids)
        self.device = device
        self.cls_idx: Dict[int, List[int]] = {class_id: [] for class_id in self.class_ids}
        for idx, target in enumerate(dataset.targets):
            if target in self.cls_idx:
                self.cls_idx[target].append(idx)
        for class_id, indices in self.cls_idx.items():
            if not indices:
                raise RuntimeError(f"no samples found for class {class_id}")

    def class_sample(self, class_id: int, batch_size: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = self.cls_idx[class_id]
        if batch_size > 0:
            replace = len(indices) < batch_size
            selected = np.random.choice(indices, size=batch_size, replace=replace).tolist()
        else:
            selected = indices

        images = []
        labels = []
        for idx in selected:
            image, label = self.dataset[idx]
            images.append(image)
            labels.append(label)
        return torch.stack(images).to(self.device), torch.tensor(labels, dtype=torch.long, device=self.device)


class Synthesizer:
    """Condensed data holder following DSDM's class-grouped synthetic layout."""

    def __init__(
        self,
        class_ids: Sequence[int],
        ipc: int,
        nchannel: int,
        image_size: Tuple[int, int],
        factor: int,
        decode_type: str,
        device: torch.device,
    ):
        self.class_ids = list(class_ids)
        self.ipc = ipc
        self.nclass = len(self.class_ids)
        self.nchannel = nchannel
        self.size = image_size
        self.device = device

        hs, ws = image_size
        self.data = torch.randn(
            size=(self.nclass * self.ipc, self.nchannel, hs, ws),
            dtype=torch.float,
            requires_grad=True,
            device=device,
        )
        self.data.data = torch.clamp(self.data.data / 4 + 0.5, min=0.0, max=1.0)
        targets = []
        for class_id in self.class_ids:
            targets.extend([class_id] * self.ipc)
        self.targets = torch.tensor(targets, dtype=torch.long, requires_grad=False, device=device)

        self.factor = max(1, factor)
        self.decode_type = decode_type
        self.resize = nn.Upsample(size=self.size, mode="bilinear", align_corners=False)

    def init(self, loader: ClassDatasetSampler, init_type: str = "random"):
        if init_type == "noise":
            return
        if init_type == "random":
            for class_pos, class_id in enumerate(self.class_ids):
                images, _ = loader.class_sample(class_id, self.ipc)
                self.data.data[self.ipc * class_pos : self.ipc * (class_pos + 1)] = images.data
            return
        if init_type == "mix":
            for class_pos, class_id in enumerate(self.class_ids):
                images, _ = loader.class_sample(class_id, self.ipc * self.factor**2)
                s = self.size[0] // self.factor
                remained = self.size[0] % self.factor
                k = 0
                n = self.ipc
                h_loc = 0
                for i in range(self.factor):
                    h_r = s + 1 if i < remained else s
                    w_loc = 0
                    for j in range(self.factor):
                        w_r = s + 1 if j < remained else s
                        img_part = F.interpolate(images[k * n : (k + 1) * n], size=(h_r, w_r), mode="bilinear")
                        self.data.data[n * class_pos : n * (class_pos + 1), :, h_loc : h_loc + h_r, w_loc : w_loc + w_r] = img_part
                        w_loc += w_r
                        k += 1
                    h_loc += h_r
            return
        raise ValueError(f"unknown dsdm_init: {init_type}")

    def parameters(self):
        return [self.data]

    def decode_zoom(self, images: torch.Tensor, labels: torch.Tensor, factor: int):
        h = images.shape[-1]
        remained = h % factor
        if remained > 0:
            images = F.pad(images, pad=(0, factor - remained, 0, factor - remained), value=0.5)
        s_crop = ceil(h / factor)
        n_crop = factor**2

        cropped = []
        for i in range(factor):
            for j in range(factor):
                h_loc = i * s_crop
                w_loc = j * s_crop
                cropped.append(images[:, :, h_loc : h_loc + s_crop, w_loc : w_loc + s_crop])
        data_dec = self.resize(torch.cat(cropped))
        target_dec = torch.cat([labels for _ in range(n_crop)])
        return data_dec, target_dec

    def decode_zoom_multi(self, images: torch.Tensor, labels: torch.Tensor, factor_max: int):
        data_multi = []
        target_multi = []
        for factor in range(1, factor_max + 1):
            data, target = self.decode_zoom(images, labels, factor)
            data_multi.append(data)
            target_multi.append(target)
        return torch.cat(data_multi), torch.cat(target_multi)

    def decode_zoom_bound(self, images: torch.Tensor, labels: torch.Tensor, factor_max: int, bound: int = 128):
        bound_cur = bound - len(images)
        budget = len(images)
        data_multi = []
        target_multi = []
        idx = 0
        decoded_total = 0
        for factor in range(factor_max, 0, -1):
            decode_size = factor**2
            n = min(bound_cur // decode_size, budget) if factor > 1 else budget
            data, target = self.decode_zoom(images[idx : idx + n], labels[idx : idx + n], factor)
            data_multi.append(data)
            target_multi.append(target)
            idx += n
            budget -= n
            decoded_total += n * decode_size
            bound_cur = bound - decoded_total - budget
            if budget == 0:
                break
        return torch.cat(data_multi), torch.cat(target_multi)

    def decode(self, images: torch.Tensor, labels: torch.Tensor, bound: int = 128):
        if self.factor <= 1:
            return images, labels
        if self.decode_type == "multi":
            return self.decode_zoom_multi(images, labels, self.factor)
        if self.decode_type == "bound":
            return self.decode_zoom_bound(images, labels, self.factor, bound=bound)
        return self.decode_zoom(images, labels, self.factor)

    def subsample(self, images: torch.Tensor, labels: torch.Tensor, max_size: int = -1):
        if images.shape[0] > max_size and max_size > 0:
            indices = torch.randperm(images.shape[0], device=images.device)[:max_size]
            images = images[indices]
            labels = labels[indices]
        return images, labels

    def sample(self, class_pos: int, max_size: int = 128):
        idx_from = self.ipc * class_pos
        idx_to = self.ipc * (class_pos + 1)
        images = self.data[idx_from:idx_to]
        labels = self.targets[idx_from:idx_to]
        images, labels = self.decode(images, labels, bound=max_size)
        return self.subsample(images, labels, max_size=max_size)


def dist(x: torch.Tensor, y: torch.Tensor, method: str = "mse") -> torch.Tensor:
    if method == "mse":
        return (x - y).pow(2).sum()
    if method == "l1":
        return (x - y).abs().sum()
    if method == "l1_mean":
        n_b = x.shape[0]
        return (x - y).abs().reshape(n_b, -1).mean(-1).sum()
    if method == "cos":
        x = x.reshape(x.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        return torch.sum(1 - torch.sum(x * y, dim=-1) / (torch.norm(x, dim=-1) * torch.norm(y, dim=-1) + 1e-6))
    if method == "l2":
        return torch.sqrt((x - y).pow(2).sum() + 1e-12)
    raise ValueError(f"unknown DSDM metric: {method}")


def get_feature_list(model, images: torch.Tensor, idx_from: int, idx_to: int):
    backbone = model.get_backbone() if hasattr(model, "get_backbone") else model
    output = backbone.get_feature(images, idx_from, idx_to)
    if isinstance(output, tuple):
        return output[0]
    return output


def matchloss(
    img_real: torch.Tensor,
    img_syn: torch.Tensor,
    model,
    idx_from: int,
    idx_to: int,
    metric: str,
    cov_weight: float,
    h_p_weight: float,
    h_p: torch.Tensor = None,
):
    with torch.no_grad():
        feat_tg = get_feature_list(model, img_real, idx_from, idx_to)
    feat = get_feature_list(model, img_syn, idx_from, idx_to)

    feat_real = feat_tg[-1]
    feat_syn = feat[-1]
    proto_loss = dist(feat_real.mean(0), feat_syn.mean(0), method=metric)

    proto_tg = feat_real.mean(0).view(feat_real.mean(0).shape[0], -1).reshape(-1)
    proto_syn = feat_syn.mean(0).view(feat_syn.mean(0).shape[0], -1).reshape(-1)
    feat_tg_view = feat_real.view(feat_real.size(0), -1)
    feat_view = feat_syn.view(feat_syn.size(0), -1)

    centered_real = feat_tg_view - proto_tg
    centered_syn = feat_view - proto_syn
    cov_real = torch.matmul(centered_real.t(), centered_real) / max(feat_tg_view.size(0) - 1, 1)
    cov_syn = torch.matmul(centered_syn.t(), centered_syn) / max(feat_view.size(0) - 1, 1)
    semantic_loss = dist(cov_syn, cov_real, method=metric) / proto_syn.shape[0] * cov_weight
    loss = proto_loss + semantic_loss

    loss_info = {
        "Proto": float(proto_loss.detach().cpu()),
        "Sem": float(semantic_loss.detach().cpu()),
        "Mem": 0.0,
    }

    if h_p is not None and h_p_weight > 0:
        h_p_loss = dist(feat_syn.mean(0), h_p, method=metric) / proto_syn.shape[0] * h_p_weight
        loss = loss + h_p_loss
        loss_info["Mem"] = float(h_p_loss.detach().cpu())

    return loss, loss_info


def distill_images_with_dsdm(anchor_model, train_dataset, class_ids, packet_cfg, device: torch.device):
    freeze_model(anchor_model)

    ipc = packet_cfg.get("ipc", 10)
    steps = packet_cfg.get("distill_steps", packet_cfg.get("niter", 100))
    lr_img = packet_cfg.get("distill_lr", packet_cfg.get("lr_img", 5e-3))
    mom_img = packet_cfg.get("mom_img", 0.5)
    batch_real = packet_cfg.get("batch_real", 256)
    batch_syn_max = packet_cfg.get("batch_syn_max", 256)
    init_type = packet_cfg.get("dsdm_init", packet_cfg.get("init", "random"))
    factor = packet_cfg.get("factor", 1)
    decode_type = packet_cfg.get("decode_type", "single")
    aug_type = packet_cfg.get("aug_type", "color_crop_cutout")
    idx_from = packet_cfg.get("idx_from", 0)
    idx_to = packet_cfg.get("idx_to", -1)
    metric = packet_cfg.get("metric", "l1")
    cov_weight = packet_cfg.get("cov_weight", 50.0)
    smooth_iter = packet_cfg.get("smooth_iter", 2000)
    smooth_factor = packet_cfg.get("smooth_factor", 0.99)
    h_p_weight = packet_cfg.get("h_p_weight", packet_cfg.get("mem_weight", 0.2))

    sample_image, _ = train_dataset[0]
    nchannel = sample_image.shape[0]
    image_size = tuple(sample_image.shape[-2:])
    loader_real = ClassDatasetSampler(train_dataset, class_ids, device)
    synset = Synthesizer(
        class_ids=class_ids,
        ipc=ipc,
        nchannel=nchannel,
        image_size=image_size,
        factor=factor,
        decode_type=decode_type,
        device=device,
    )
    synset.init(loader_real, init_type=init_type)
    aug = DiffAug(strategy=aug_type, batch=True)
    optimizer = torch.optim.SGD(synset.parameters(), lr=lr_img, momentum=mom_img)

    smooth_syns = [None] * len(class_ids)
    h_p = [None] * len(class_ids)
    loss_dict_avg = {"Proto": 0.0, "Sem": 0.0, "Mem": 0.0}
    last_loss = 0.0

    progress = tqdm(range(steps), desc="dsdm distill", leave=False)
    for it in progress:
        loss_total = 0.0
        loss_dict_avg = {"Proto": 0.0, "Sem": 0.0, "Mem": 0.0}
        synset.data.data = torch.clamp(synset.data.data, min=0.0, max=1.0)

        for class_pos, class_id in enumerate(class_ids):
            img_real, _ = loader_real.class_sample(class_id, batch_real)
            img_syn, _ = synset.sample(class_pos, max_size=batch_syn_max)
            n_real = img_real.shape[0]
            img_aug = aug(torch.cat([img_real, img_syn]))

            optimizer.zero_grad(set_to_none=True)
            class_h_p = h_p[class_pos] if it > smooth_iter else None
            loss, loss_info = matchloss(
                img_real=img_aug[:n_real],
                img_syn=img_aug[n_real:],
                model=anchor_model,
                idx_from=idx_from,
                idx_to=idx_to,
                metric=metric,
                cov_weight=cov_weight,
                h_p_weight=h_p_weight,
                h_p=class_h_p,
            )
            loss_total += float(loss.detach().cpu())
            for key, value in loss_info.items():
                loss_dict_avg[key] = loss_dict_avg.get(key, 0.0) + value
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            for class_pos, _ in enumerate(class_ids):
                img_syn, _ = synset.sample(class_pos, max_size=batch_syn_max)
                syn_img_aug = aug(img_syn)
                feature_h = get_feature_list(anchor_model, syn_img_aug, idx_from, idx_to)
                smooth_syns[class_pos] = feature_h[-1].mean(0)
                if it == 0 or h_p[class_pos] is None:
                    h_p[class_pos] = smooth_syns[class_pos]
                else:
                    h_p[class_pos] = (1 - smooth_factor) * smooth_syns[class_pos] + smooth_factor * h_p[class_pos]

        last_loss = loss_total / len(class_ids)
        progress.set_postfix(
            loss=f"{last_loss:.4f}",
            proto=f"{loss_dict_avg.get('Proto', 0.0) / max(len(class_ids), 1):.4f}",
            sem=f"{loss_dict_avg.get('Sem', 0.0) / max(len(class_ids), 1):.4f}",
            mem=f"{loss_dict_avg.get('Mem', 0.0) / max(len(class_ids), 1):.4f}",
        )

    images = synset.data.detach().clamp(0.0, 1.0).cpu()
    hard_labels = synset.targets.detach().cpu()
    normalizer = max(len(class_ids), 1)
    meta = {
        "distill_method": "dsdm_proto_sem_mem_no_topology",
        "distill_steps": steps,
        "distill_lr": lr_img,
        "mom_img": mom_img,
        "batch_real": batch_real,
        "batch_syn_max": batch_syn_max,
        "dsdm_init": init_type,
        "factor": factor,
        "decode_type": decode_type,
        "aug_type": aug_type,
        "idx_from": idx_from,
        "idx_to": idx_to,
        "metric": metric,
        "cov_weight": cov_weight,
        "smooth_iter": smooth_iter,
        "smooth_factor": smooth_factor,
        "h_p_weight": h_p_weight,
        "distill_final_loss": last_loss,
        "distill_final_proto_loss": loss_dict_avg.get("Proto", 0.0) / normalizer,
        "distill_final_sem_loss": loss_dict_avg.get("Sem", 0.0) / normalizer,
        "distill_final_mem_loss": loss_dict_avg.get("Mem", 0.0) / normalizer,
    }
    return images, hard_labels, meta
