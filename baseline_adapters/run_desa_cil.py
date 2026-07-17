#!/usr/bin/env python3
"""DeSA adaptation for class-incremental heterogeneous socialized learning.

The original DeSA implementation averages same-label anchor slots across
clients.  That operation is undefined when clients have disjoint class sets.
DeSA-CIL keeps the official random-ConvNet distribution-matching anchors,
iterative KD and supervised-contrastive regularizer, but concatenates the
class-disjoint anchors and obtains every anchor target only from its owner.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[1]
DSDM_ROOT = ROOT / "DSDM"
DESA_ROOT = ROOT / "external_baselines/repos/DESA"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DSDM_ROOT))

from train import define_model  # noqa: E402
from agent_data import build_agent_args, get_agent_dir  # noqa: E402
from config_adapter import load_config  # noqa: E402


MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


def load_external_module(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.append(str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DESA_UTILS = load_external_module("desa_official_utils", DESA_ROOT / "utils.py")
DESA_LOSS = load_external_module("desa_official_loss", DESA_ROOT / "loss_fn.py")


@dataclass(frozen=True)
class Settings:
    config: str
    ipc: int
    output_root: str
    anchor_source: str | None
    rounds: int
    anchor_iters: int
    batch_size: int
    kd_batch_size: int
    eval_batch_size: int
    workers: int
    lr_kd: float
    lr_img: float
    lambda_ori: float
    lambda_kd: float
    lambda_reg: float
    temperature: float
    distil_weight: float
    device: str
    seed: int
    checkpoint_every: int
    smoke: bool
    resume: bool
    force_anchors: bool


class ProjectModel(nn.Module):
    """Expose DeSA's (feature, logits)/embed API for an exact project model."""

    def __init__(self, model: nn.Module, feature_index: int, model_name: str):
        super().__init__()
        self.model = model
        self.feature_index = int(feature_index)
        self.model_name = model_name

    def _final_classifier(self) -> nn.Module:
        if hasattr(self.model, "fc"):
            return self.model.fc
        classifier = self.model.classifier
        if isinstance(classifier, nn.Sequential):
            return classifier[-1]
        return classifier

    def _forward_with_feature(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        captured = []

        def capture_input(_module, inputs):
            captured.append(inputs[0])

        handle = self._final_classifier().register_forward_pre_hook(capture_input)
        try:
            logits = self.model(images)
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError(f"failed to capture penultimate feature for {self.model_name}")
        return captured[0].flatten(1), logits

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        return self._forward_with_feature(images)[0]

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._forward_with_feature(images)


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset, indices: list[int]):
        self.dataset = dataset
        self.indices = [int(index) for index in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]


class AnchorTargetDataset(Dataset):
    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        owners: torch.Tensor,
        targets: torch.Tensor,
    ):
        self.images = images.cpu()
        self.labels = labels.cpu()
        self.owners = owners.cpu()
        self.targets = targets.cpu()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index], self.owners[index], self.targets[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run class-incremental DeSA with exact project backbones.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ipc", type=int, choices=[10, 50], required=True)
    parser.add_argument("--output-root", default=str(ROOT / "external_baselines/outputs/desa_cil"))
    parser.add_argument("--anchor-source", default=None, help="Optional compatible legacy anchors.pt")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--anchor-iters", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--kd-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr-kd", type=float, default=0.01)
    parser.add_argument("--lr-img", type=float, default=1.0)
    parser.add_argument("--lambda-ori", type=float, default=1.0)
    parser.add_argument("--lambda-kd", type=float, default=1.0)
    parser.add_argument("--lambda-reg", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=20.0)
    parser.add_argument("--distil-weight", type=float, default=0.9)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-anchors", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")


def build_datasets(data_dir: str):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    static_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train = datasets.CIFAR100(data_dir, train=True, transform=train_transform, download=False)
    train_static = datasets.CIFAR100(data_dir, train=True, transform=static_transform, download=False)
    test = datasets.CIFAR100(data_dir, train=False, transform=static_transform, download=False)
    return train, train_static, test


def class_indices(targets, classes: list[int]) -> list[int]:
    allowed = set(classes)
    return [index for index, target in enumerate(targets) if int(target) in allowed]


def load_models(config_path: str, cfg_dict: dict, device: torch.device) -> tuple[dict[int, ProjectModel], dict[int, list[int]], dict[int, str]]:
    class_split = {
        int(key.replace("agent_", "")): [int(value) for value in values]
        for key, values in cfg_dict["agents"]["class_split"].items()
    }
    model_split = {
        int(key.replace("agent_", "")): str(value)
        for key, value in cfg_dict["agents"]["model_split"].items()
    }
    models = {}
    for agent in sorted(class_split):
        args = build_agent_args(cfg_dict, config_path, agent)
        model = define_model(args, 100)
        checkpoint = get_agent_dir(args, agent) / "checkpoints/expert_model.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"missing project expert checkpoint: {checkpoint}")
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        wrapped = ProjectModel(model, args.idx_from, model_split[agent]).to(device)
        models[agent] = wrapped
        print(
            f"[model/a{agent}] {model_split[agent]} family={args.net_type} depth={args.depth} "
            f"width={args.width} feature_idx={args.idx_from} checkpoint={checkpoint}"
        )
    return models, class_split, model_split


def materialize(dataset: Dataset, indices: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    data_loader = DataLoader(IndexedDataset(dataset, indices), batch_size=512, shuffle=False, num_workers=2)
    images, labels = [], []
    for batch_images, batch_labels in data_loader:
        images.append(batch_images)
        labels.append(batch_labels)
    return torch.cat(images).to(device), torch.cat(labels).long().to(device)


def generate_sender_anchors(
    dataset: Dataset,
    indices: list[int],
    classes: list[int],
    ipc: int,
    iterations: int,
    lr_img: float,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    real_images, real_labels = materialize(dataset, indices, device)
    positions = {class_id: torch.nonzero(real_labels == class_id, as_tuple=False).flatten() for class_id in classes}
    generator = torch.Generator(device=device).manual_seed(seed)
    images = torch.randn(len(classes) * ipc, 3, 32, 32, generator=generator, device=device, requires_grad=True)
    labels = torch.tensor([class_id for class_id in classes for _ in range(ipc)], device=device, dtype=torch.long)
    optimizer = torch.optim.SGD([images], lr=lr_img, momentum=0.5)
    augmentation = DESA_UTILS.ParamDiffAug()
    for iteration in range(iterations):
        torch.manual_seed(seed * 100000 + iteration)
        network = DESA_UTILS.get_network("ConvNet", 3, len(classes), (32, 32)).to(device).train()
        embed = network.module.embed if hasattr(network, "module") else network.embed
        loss = torch.zeros((), device=device)
        for offset, class_id in enumerate(classes):
            class_positions = positions[class_id]
            sampled = class_positions[torch.randperm(len(class_positions), generator=generator, device=device)[:256]]
            augmentation_seed = seed * 100000 + iteration * len(classes) + offset
            real = DESA_UTILS.DiffAugment(
                real_images[sampled], "color_crop_cutout_flip_scale_rotate", seed=augmentation_seed, param=augmentation
            )
            synthetic = DESA_UTILS.DiffAugment(
                images[offset * ipc : (offset + 1) * ipc],
                "color_crop_cutout_flip_scale_rotate",
                seed=augmentation_seed,
                param=augmentation,
            )
            loss = loss + (embed(real).detach().mean(0) - embed(synthetic).mean(0)).square().sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if iteration == 0 or (iteration + 1) % 100 == 0 or iteration + 1 == iterations:
            print(f"[anchor] iter={iteration + 1}/{iterations} loss={float(loss.detach()) / len(classes):.4f}")
        del network
    return images.detach().cpu(), labels.detach().cpu()


def validate_anchor_dict(anchors: dict[int, tuple[torch.Tensor, torch.Tensor]], classes: dict[int, list[int]], ipc: int) -> None:
    if set(anchors) != set(classes):
        raise ValueError(f"anchor owners {sorted(anchors)} do not match agents {sorted(classes)}")
    for owner in classes:
        images, labels = anchors[owner]
        expected = len(classes[owner]) * ipc
        if len(images) != expected or len(labels) != expected:
            raise ValueError(f"owner {owner}: expected {expected} anchors, got {len(images)}")
        counts = torch.bincount(labels.long(), minlength=100)
        for class_id in classes[owner]:
            if int(counts[class_id]) != ipc:
                raise ValueError(f"owner {owner} class {class_id}: expected IPC={ipc}, got {int(counts[class_id])}")


def load_legacy_anchors(path: Path) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if "anchors" in value:
        value = value["anchors"]
    return {int(owner): (pair[0].float().cpu(), pair[1].long().cpu()) for owner, pair in value.items()}


def prepare_anchors(
    cfg: Settings,
    dataset: Dataset,
    train_indices: dict[int, list[int]],
    classes: dict[int, list[int]],
    run_root: Path,
    device: torch.device,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    path = run_root / "anchors.pt"
    if path.exists() and not cfg.force_anchors:
        anchors = load_legacy_anchors(path)
        validate_anchor_dict(anchors, classes, cfg.ipc)
        print(f"[anchors] reuse {path}")
        return anchors
    if cfg.anchor_source:
        source = Path(cfg.anchor_source)
        anchors = load_legacy_anchors(source)
        validate_anchor_dict(anchors, classes, cfg.ipc)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"anchors": anchors, "source": str(source.resolve()), "ipc": cfg.ipc, "seed": cfg.seed}, path)
        print(f"[anchors] imported compatible DeSA anchors from {source}")
        return anchors

    anchors = {}
    iterations = 1 if cfg.smoke else cfg.anchor_iters
    for owner in sorted(classes):
        print(f"[anchors/a{owner}] classes={classes[owner][0]}-{classes[owner][-1]} ipc={cfg.ipc}")
        anchors[owner] = generate_sender_anchors(
            dataset,
            train_indices[owner],
            classes[owner],
            cfg.ipc,
            iterations,
            cfg.lr_img,
            cfg.seed + owner * 1009,
            device,
        )
    validate_anchor_dict(anchors, classes, cfg.ipc)
    torch.save({"anchors": anchors, "source": "generated", "ipc": cfg.ipc, "seed": cfg.seed}, path)
    return anchors


@torch.no_grad()
def owner_targets(
    models: dict[int, ProjectModel],
    anchors: dict[int, tuple[torch.Tensor, torch.Tensor]],
    classes: dict[int, list[int]],
    device: torch.device,
    batch_size: int,
) -> dict[int, torch.Tensor]:
    targets = {}
    for owner, model in models.items():
        model.eval()
        outputs = []
        data_loader = DataLoader(TensorDataset(anchors[owner][0]), batch_size=batch_size, shuffle=False)
        columns = torch.tensor(classes[owner], device=device)
        for (images,) in data_loader:
            _, logits = model(images.to(device))
            outputs.append(logits.index_select(1, columns).cpu())
        targets[owner] = torch.cat(outputs)
    return targets


def make_external_anchor_dataset(
    receiver: int,
    anchors: dict[int, tuple[torch.Tensor, torch.Tensor]],
    targets: dict[int, torch.Tensor],
) -> AnchorTargetDataset:
    images, labels, owners, logits = [], [], [], []
    for owner in sorted(anchors):
        if owner == receiver:
            continue
        owner_images, owner_labels = anchors[owner]
        images.append(owner_images)
        labels.append(owner_labels)
        owners.append(torch.full((len(owner_labels),), owner, dtype=torch.long))
        logits.append(targets[owner])
    return AnchorTargetDataset(torch.cat(images), torch.cat(labels), torch.cat(owners), torch.cat(logits))


def restricted_owner_kd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    owners: torch.Tensor,
    classes: dict[int, list[int]],
    temperature: float,
    distil_weight: float,
) -> torch.Tensor:
    total = torch.zeros((), device=student_logits.device)
    for owner in owners.unique(sorted=True):
        mask = owners == owner
        columns = torch.tensor(classes[int(owner)], device=student_logits.device)
        student_local = student_logits[mask].index_select(1, columns)
        teacher_local = teacher_logits[mask]
        hard = F.cross_entropy(student_logits[mask], labels[mask])
        soft = F.kl_div(
            F.log_softmax(student_local / temperature, dim=1),
            F.softmax(teacher_local / temperature, dim=1),
            reduction="batchmean",
        )
        group_loss = (1.0 - distil_weight) * hard + distil_weight * temperature * temperature * soft
        total = total + group_loss * (int(mask.sum()) / len(labels))
    return total


def next_or_restart(iterator, data_loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(data_loader)
        return next(iterator), iterator


def train_receiver_round(
    model: ProjectModel,
    optimizer: torch.optim.Optimizer,
    local_loader: DataLoader,
    anchor_loader: DataLoader,
    classes: dict[int, list[int]],
    cfg: Settings,
    device: torch.device,
    distance_loss: nn.Module,
) -> dict[str, float]:
    model.train()
    anchor_iterator = iter(anchor_loader)
    sums = {"loss": 0.0, "ori": 0.0, "kd": 0.0, "reg": 0.0, "correct": 0.0, "count": 0.0}
    for images, labels in local_loader:
        (anchor_images, anchor_labels, owners, teacher_logits), anchor_iterator = next_or_restart(anchor_iterator, anchor_loader)
        images, labels = images.to(device), labels.long().to(device)
        anchor_images = anchor_images.to(device)
        anchor_labels = anchor_labels.long().to(device)
        owners = owners.long().to(device)
        teacher_logits = teacher_logits.to(device)

        features, logits = model(images)
        _, anchor_logits = model(anchor_images)
        loss_ori = F.cross_entropy(logits, labels)
        loss_kd = restricted_owner_kd(
            anchor_logits, teacher_logits, anchor_labels, owners, classes, cfg.temperature, cfg.distil_weight
        )
        anchor_features = model.embed(anchor_images).detach()
        loss_reg = distance_loss(features, anchor_features, labels, anchor_labels)
        loss = cfg.lambda_ori * loss_ori + cfg.lambda_kd * loss_kd + cfg.lambda_reg * loss_reg
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        size = labels.numel()
        sums["loss"] += float(loss.detach()) * size
        sums["ori"] += float(loss_ori.detach()) * size
        sums["kd"] += float(loss_kd.detach()) * size
        sums["reg"] += float(loss_reg.detach()) * size
        sums["correct"] += int((logits.argmax(1) == labels).sum())
        sums["count"] += size
    count = max(1.0, sums["count"])
    return {key: value / count for key, value in sums.items() if key != "count"}


@torch.no_grad()
def evaluate(model: ProjectModel, dataset: Dataset, indices: list[int], device: torch.device, batch_size: int) -> float:
    model.eval()
    data_loader = DataLoader(IndexedDataset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=2)
    correct = total = 0
    for images, labels in data_loader:
        _, logits = model(images.to(device))
        labels = labels.to(device)
        correct += int((logits.argmax(1) == labels).sum())
        total += labels.numel()
    return 100.0 * correct / max(1, total)


def checkpoint_state(
    path: Path,
    round_index: int,
    models: dict[int, ProjectModel],
    optimizers: dict[int, torch.optim.Optimizer],
    cfg: Settings,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "round": round_index,
            "models": {agent: model.state_dict() for agent, model in models.items()},
            "optimizers": {agent: optimizer.state_dict() for agent, optimizer in optimizers.items()},
            "settings": asdict(cfg),
        },
        path,
    )


def write_results(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(cfg: Settings) -> Path:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    config_path = str(Path(cfg.config).resolve())
    project_cfg = load_config(config_path)
    run_name = project_cfg["project"]["run_name"]
    configured_ipc = int(project_cfg.get("distillation", {}).get("ipc", cfg.ipc))
    if configured_ipc != cfg.ipc:
        raise ValueError(f"config IPC={configured_ipc} does not match requested IPC={cfg.ipc}")
    run_root = Path(cfg.output_root) / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "resolved_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    log_path = run_root / "run.log"

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    log(f"[start] DeSA-CIL config={config_path} ipc={cfg.ipc} seed={cfg.seed} device={device}")
    train, train_static, test = build_datasets(project_cfg["dataset"].get("data_dir", "./data"))
    models, classes, model_names = load_models(config_path, project_cfg, device)
    train_indices = {agent: class_indices(train.targets, agent_classes) for agent, agent_classes in classes.items()}
    anchors = prepare_anchors(cfg, train_static, train_indices, classes, run_root, device)

    all_test = list(range(len(test)))
    before = {
        agent: evaluate(
            model,
            test,
            class_indices(test.targets, classes[agent])[: cfg.eval_batch_size] if cfg.smoke else class_indices(test.targets, classes[agent]),
            device,
            cfg.eval_batch_size,
        )
        for agent, model in models.items()
    }
    optimizers = {agent: torch.optim.SGD(model.parameters(), lr=cfg.lr_kd) for agent, model in models.items()}
    start_round = 0
    checkpoint = run_root / "checkpoint.pt"
    if cfg.resume and checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        for agent in models:
            models[agent].load_state_dict(state["models"][agent])
            optimizers[agent].load_state_dict(state["optimizers"][agent])
        start_round = int(state["round"])
        log(f"[resume] round={start_round} checkpoint={checkpoint}")

    rounds = 1 if cfg.smoke else cfg.rounds
    distance_loss = DESA_LOSS.Distance_loss(device=device)
    for round_index in range(start_round, rounds):
        started = time.time()
        targets = owner_targets(models, anchors, classes, device, cfg.kd_batch_size)
        for receiver in sorted(models):
            local_indices = train_indices[receiver]
            if cfg.smoke:
                local_indices = local_indices[: 2 * cfg.batch_size]
            local_loader = DataLoader(
                IndexedDataset(train, local_indices),
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=0 if cfg.smoke else cfg.workers,
                pin_memory=device.type == "cuda",
            )
            external = make_external_anchor_dataset(receiver, anchors, targets)
            anchor_loader = DataLoader(external, batch_size=cfg.kd_batch_size, shuffle=True, num_workers=0)
            metrics = train_receiver_round(
                models[receiver], optimizers[receiver], local_loader, anchor_loader, classes, cfg, device, distance_loss
            )
            log(
                f"[round={round_index + 1}/{rounds} a{receiver}] loss={metrics['loss']:.4f} "
                f"ori={metrics['ori']:.4f} kd={metrics['kd']:.4f} reg={metrics['reg']:.4f} acc={metrics['correct']:.4f}"
            )
        if (round_index + 1) % cfg.checkpoint_every == 0 or round_index + 1 == rounds:
            checkpoint_state(checkpoint, round_index + 1, models, optimizers, cfg)
        log(f"[round={round_index + 1}] seconds={time.time() - started:.1f}")

    rows = []
    total_logit_bytes = cfg.rounds * (len(models) - 1) * sum(
        (len(images) * 25 * 4) for images, _ in anchors.values()
    )
    per_receiver_logit_bytes = cfg.rounds * 3 * 25 * cfg.ipc * 25 * 4
    for receiver, model in models.items():
        own = class_indices(test.targets, classes[receiver])
        novel = [index for index in all_test if int(test.targets[index]) not in set(classes[receiver])]
        global_indices = all_test
        if cfg.smoke:
            own = own[: cfg.eval_batch_size]
            novel = novel[: cfg.eval_batch_size]
            global_indices = all_test[: cfg.eval_batch_size]
        expert_acc = evaluate(model, test, own, device, cfg.eval_batch_size)
        new_acc = evaluate(model, test, novel, device, cfg.eval_batch_size)
        global_acc = evaluate(model, test, global_indices, device, cfg.eval_batch_size)
        rows.append(
            {
                "method": "DeSA-CIL",
                "ipc": cfg.ipc,
                "seed": cfg.seed,
                "receiver": receiver,
                "backbone": model_names[receiver],
                "acc_global": global_acc,
                "acc_new": new_acc,
                "acc_expert": expert_acc,
                "forgetting": before[receiver] - expert_acc,
                "expert_before": before[receiver],
                "external_comm_images": 3 * 25 * cfg.ipc,
                "iterative_owner_logit_bytes_per_receiver": per_receiver_logit_bytes,
                "iterative_owner_logit_bytes_all_agents": total_logit_bytes,
            }
        )
        log(f"[result/a{receiver}] global={global_acc:.2f} new={new_acc:.2f} expert={expert_acc:.2f} forgetting={before[receiver] - expert_acc:.2f}")
    average = dict(rows[0])
    average["receiver"] = "avg"
    average["backbone"] = "heterogeneous-average"
    for key in ["acc_global", "acc_new", "acc_expert", "forgetting", "expert_before"]:
        average[key] = sum(float(row[key]) for row in rows) / len(rows)
    rows.append(average)
    output = run_root / "social_results.csv"
    write_results(output, rows)
    log(f"[done] {output}")
    return output


def main() -> None:
    args = parse_args()
    project_cfg = load_config(args.config)
    seed = int(project_cfg.get("runtime", {}).get("seed", 0) if args.seed is None else args.seed)
    cfg = Settings(
        config=args.config,
        ipc=args.ipc,
        output_root=args.output_root,
        anchor_source=args.anchor_source,
        rounds=args.rounds,
        anchor_iters=args.anchor_iters,
        batch_size=args.batch_size,
        kd_batch_size=args.kd_batch_size,
        eval_batch_size=args.eval_batch_size,
        workers=args.workers,
        lr_kd=args.lr_kd,
        lr_img=args.lr_img,
        lambda_ori=args.lambda_ori,
        lambda_kd=args.lambda_kd,
        lambda_reg=args.lambda_reg,
        temperature=args.temperature,
        distil_weight=args.distil_weight,
        device=args.device,
        seed=seed,
        checkpoint_every=args.checkpoint_every,
        smoke=args.smoke,
        resume=args.resume,
        force_anchors=args.force_anchors,
    )
    run(cfg)


if __name__ == "__main__":
    main()
