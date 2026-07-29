#!/usr/bin/env python3
"""Complete MASC reproduction for CIFAR-100 socialized learning.

The released MASC repository contains the CIFAR-100 architectures and inference
code, but not the complete training entry/checkpoints.  This adapter implements
the paper's CC, RA and HFE stages while changing only the number of real images
made available to collective cultivation (full data, IPC=10, or IPC=50).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MASC_ROOT = ROOT / "external_baselines/repos/MASC_SL/MASC_SL_inference_CIFAR100_raw_4_25"

from agent_data import get_agent_class_split  # noqa: E402
from config_adapter import load_config  # noqa: E402


def _load_masc_network_module():
    path = MASC_ROOT / "utils/network_wider_cifar100.py"
    spec = importlib.util.spec_from_file_location("_masc_network_wider_cifar100", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load MASC network module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MASC_NETWORKS = _load_masc_network_module()
Netwider = _MASC_NETWORKS.Netwider
Netwider_multi = _MASC_NETWORKS.Netwider_multi


LEGACY_CLASS_SPLIT = {agent: list(range(agent * 25, (agent + 1) * 25)) for agent in range(4)}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Settings:
    config: str | None
    config_sha256: str | None
    source_run_name: str
    class_split: dict[int, list[int]]
    budget: str
    seed: int
    data_dir: str
    output_root: str
    teacher_epochs: int
    cc_epochs: int
    ra_epochs: int
    batch_size: int
    eval_batch_size: int
    workers: int
    lr: float
    momentum: float
    weight_decay: float
    lambda_kd: float
    lambda_align: float
    energy_anchor: float
    temperature: float
    device: str
    smoke: bool
    force: bool

    @property
    def ipc(self) -> int:
        return 500 if self.budget == "full" else int(self.budget)


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset, indices: list[int]):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]


class LogitHead(nn.Module):
    """MASC official three-layer classifier without the erroneous final Softmax."""

    def __init__(self, in_features: int = 512, num_classes: int = 100):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.layers(feature)


class Expert(nn.Module):
    """Official homogeneous MASC expert backbone and classifier."""

    def __init__(self):
        super().__init__()
        official = Netwider(13)
        self.backbone = official.layers
        self.head = nn.Sequential(*list(official.classifier[:-1]))

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        output = images
        for layer in self.backbone:
            output = layer(output)
        return output.flatten(1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(images))


class CollectiveStudent(nn.Module):
    """Official wide student plus teacher-head projections used during CC."""

    def __init__(self, num_agents: int):
        super().__init__()
        official = Netwider_multi(13)
        self.backbone = official.layers
        self.general_head = nn.Sequential(*list(official.classifier[:-1]))
        self.teacher_heads = nn.ModuleList(LogitHead() for _ in range(int(num_agents)))

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        output = images
        for layer in self.backbone:
            output = layer(output)
        return output.flatten(1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.embed(images)
        return feature, self.general_head(feature)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce complete MASC with IPC-limited CC images.")
    parser.add_argument(
        "--config",
        default=None,
        help="Project config supplying the current class-disjoint agent split; omitted only for legacy 4x25 runs.",
    )
    parser.add_argument("--budget", nargs="+", choices=["full", "10", "50"], default=["full", "10", "50"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--output-root", default=str(ROOT / "external_baselines/outputs/masc_complete"))
    parser.add_argument("--teacher-epochs", type=int, default=500)
    parser.add_argument("--cc-epochs", type=int, default=500)
    parser.add_argument("--ra-epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--lambda-kd", type=float, default=1.0)
    parser.add_argument("--lambda-align", type=float, default=0.1)
    parser.add_argument("--energy-anchor", type=float, default=-20.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
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


def load_data(data_dir: str):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train = datasets.CIFAR100(data_dir, train=True, transform=train_transform, download=False)
    test = datasets.CIFAR100(data_dir, train=False, transform=eval_transform, download=False)
    return train, test


def indices_for_classes(targets, classes: list[int]) -> list[int]:
    allowed = set(classes)
    return [index for index, target in enumerate(targets) if int(target) in allowed]


def validate_class_split(class_split: dict[int, list[int]]) -> None:
    if sorted(class_split) != list(range(len(class_split))):
        raise ValueError("MASC agent IDs must be contiguous from zero")
    flattened = [int(class_id) for classes in class_split.values() for class_id in classes]
    if len(flattened) != 100 or sorted(flattened) != list(range(100)):
        raise ValueError("MASC class sets must be a disjoint global-label cover of CIFAR-100")
    if len({len(classes) for classes in class_split.values()}) != 1:
        raise ValueError("MASC scaling protocol requires balanced class counts")


def class_membership_mask(labels: torch.Tensor, classes: list[int]) -> torch.Tensor:
    allowed = torch.tensor(classes, dtype=labels.dtype, device=labels.device)
    return (labels[:, None] == allowed[None, :]).any(dim=1)


def communication_indices(
    targets,
    ipc: int,
    seed: int,
    class_split: dict[int, list[int]],
) -> dict[int, list[int]]:
    rng = np.random.default_rng(seed)
    selected: dict[int, list[int]] = {}
    target_array = np.asarray(targets)
    for agent, classes in class_split.items():
        selected[agent] = []
        for class_id in classes:
            candidates = np.flatnonzero(target_array == class_id)
            choice = rng.choice(candidates, size=ipc, replace=False)
            selected[agent].extend(int(index) for index in choice)
    return selected


def loader(dataset, indices: list[int], cfg: Settings, shuffle: bool = True) -> DataLoader:
    if cfg.smoke:
        indices = indices[: min(len(indices), 2 * cfg.batch_size)]
    return DataLoader(
        IndexedDataset(dataset, indices),
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=0 if cfg.smoke else cfg.workers,
        pin_memory=cfg.device.startswith("cuda"),
    )


def free_energy(logits: torch.Tensor) -> torch.Tensor:
    return -torch.logsumexp(logits, dim=1)


def energy_alignment(logits: torch.Tensor, anchor: float) -> torch.Tensor:
    return (free_energy(logits) - anchor).square().mean()


def kd_loss(student: torch.Tensor, teacher: torch.Tensor, temperature: float) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(student / temperature, dim=1),
        F.softmax(teacher / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature * temperature)


def train_expert(model: Expert, data_loader: DataLoader, cfg: Settings, device: torch.device, log) -> None:
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    for epoch in range(cfg.teacher_epochs):
        model.train()
        totals = [0.0, 0]
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels) + cfg.lambda_align * energy_alignment(logits, cfg.energy_anchor)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals[0] += float(loss.detach()) * labels.numel()
            totals[1] += labels.numel()
        if epoch == 0 or (epoch + 1) % 25 == 0 or epoch + 1 == cfg.teacher_epochs:
            log(f"[teacher] epoch={epoch + 1}/{cfg.teacher_epochs} loss={totals[0] / max(1, totals[1]):.4f}")


def train_cc(
    student: CollectiveStudent,
    experts: dict[int, Expert],
    class_split: dict[int, list[int]],
    data_loader: DataLoader,
    cfg: Settings,
    device: torch.device,
    log,
) -> None:
    student.to(device)
    for expert in experts.values():
        expert.to(device).eval().requires_grad_(False)
    optimizer = torch.optim.SGD(student.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    for epoch in range(cfg.cc_epochs):
        student.train()
        totals = [0.0, 0.0, 0.0, 0]
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            features, general_logits = student(images)
            ce = F.cross_entropy(general_logits, labels)
            kd = torch.zeros((), device=device)
            for agent, classes in class_split.items():
                mask = class_membership_mask(labels, classes)
                if mask.any():
                    with torch.no_grad():
                        teacher_logits = experts[agent](images[mask])
                    projected_logits = student.teacher_heads[agent](features[mask])
                    kd = kd + kd_loss(projected_logits, teacher_logits, cfg.temperature)
            kd = kd / len(experts)
            align = energy_alignment(general_logits, cfg.energy_anchor)
            loss = ce + cfg.lambda_kd * kd + cfg.lambda_align * align
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals[0] += float(ce.detach()) * labels.numel()
            totals[1] += float(kd.detach()) * labels.numel()
            totals[2] += float(align.detach()) * labels.numel()
            totals[3] += labels.numel()
        if epoch == 0 or (epoch + 1) % 25 == 0 or epoch + 1 == cfg.cc_epochs:
            count = max(1, totals[3])
            log(f"[CC] epoch={epoch + 1}/{cfg.cc_epochs} ce={totals[0] / count:.4f} kd={totals[1] / count:.4f} align={totals[2] / count:.4f}")


def train_ra(
    student: CollectiveStudent,
    expert: Expert,
    grown_head: LogitHead,
    data_loader: DataLoader,
    cfg: Settings,
    device: torch.device,
    log,
    agent: int,
) -> None:
    student.to(device).eval().requires_grad_(False)
    expert.to(device).eval().requires_grad_(False)
    grown_head.to(device)
    optimizer = torch.optim.SGD(grown_head.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    for epoch in range(cfg.ra_epochs):
        grown_head.train()
        totals = [0.0, 0]
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                features = student.embed(images)
                teacher_logits = expert(images)
            logits = grown_head(features)
            ce = F.cross_entropy(logits, labels)
            kd = kd_loss(logits, teacher_logits, cfg.temperature)
            align = energy_alignment(logits, cfg.energy_anchor)
            loss = ce + cfg.lambda_kd * kd + cfg.lambda_align * align
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals[0] += float(loss.detach()) * labels.numel()
            totals[1] += labels.numel()
        if epoch == 0 or (epoch + 1) % 25 == 0 or epoch + 1 == cfg.ra_epochs:
            log(f"[RA/a{agent}] epoch={epoch + 1}/{cfg.ra_epochs} loss={totals[0] / max(1, totals[1]):.4f}")


@torch.no_grad()
def accuracy(model_fn, dataset: Dataset, indices: list[int], batch_size: int, device: torch.device) -> float:
    data_loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=2)
    correct = total = 0
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model_fn(images)
        correct += int((logits.argmax(1) == labels).sum())
        total += labels.numel()
    return 100.0 * correct / max(1, total)


@torch.no_grad()
def hfe_logits(student: CollectiveStudent, grown_head: LogitHead, images: torch.Tensor) -> torch.Tensor:
    features, general_logits = student(images)
    expert_logits = grown_head(features)
    use_expert = torch.logsumexp(expert_logits, dim=1) > torch.logsumexp(general_logits, dim=1)
    return torch.where(use_expert[:, None], expert_logits, general_logits)


def save_checkpoint(path: Path, state: dict, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def matching_checkpoint(path: Path, metadata: dict, force: bool) -> bool:
    meta_path = path.with_suffix(".json")
    if force or not path.exists() or not meta_path.exists():
        return False
    return json.loads(meta_path.read_text(encoding="utf-8")) == metadata


def train_or_load_experts(cfg: Settings, train, device, run_root: Path, log) -> dict[int, Expert]:
    experts = {}
    expert_root = (
        Path(cfg.output_root) / f"experts_seed{cfg.seed}"
        if cfg.config is None
        else Path(cfg.output_root) / "expert_sets" / cfg.source_run_name
    )
    for agent, classes in cfg.class_split.items():
        path = expert_root / f"agent_{agent}.pt"
        meta = {
            "stage": "expert",
            "seed": cfg.seed,
            "epochs": cfg.teacher_epochs,
            "classes": classes,
            "lr": cfg.lr,
            "momentum": cfg.momentum,
            "weight_decay": cfg.weight_decay,
            "lambda_align": cfg.lambda_align,
            "energy_anchor": cfg.energy_anchor,
            "paper_recipe": True,
        }
        model = Expert()
        if matching_checkpoint(path, meta, cfg.force):
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            log(f"[teacher/a{agent}] reuse {path}")
        else:
            train_expert(model, loader(train, indices_for_classes(train.targets, classes), cfg), cfg, device, log)
            save_checkpoint(path, model.state_dict(), meta)
        experts[agent] = model
    return experts


def parameter_bytes(model: nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_full_result(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        average = next(row for row in csv.DictReader(handle) if row["agent_id"] == "avg")
    failures = []
    if float(average["expert_before"]) < 50.0:
        failures.append(f"expert_before={float(average['expert_before']):.2f} < 50")
    if float(average["acc_global"]) < 40.0:
        failures.append(f"acc_global={float(average['acc_global']):.2f} < 40")
    if float(average["acc_new"]) < 35.0:
        failures.append(f"acc_new={float(average['acc_new']):.2f} < 35")
    if failures:
        raise RuntimeError("MASC full-data sanity validation failed; do not run IPC experiments: " + "; ".join(failures))
    marker = path.parent / "FULL_VALIDATED"
    marker.write_text(json.dumps(average, indent=2), encoding="utf-8")
    print(f"[validation] full-data MASC passed; marker={marker}", flush=True)


def run(cfg: Settings) -> Path:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    legacy = cfg.config is None
    run_name = (
        f"seed{cfg.seed}_{cfg.budget}"
        if legacy
        else f"{cfg.source_run_name}_masc_homogeneous_{cfg.budget}"
    )
    run_root = Path(cfg.output_root) / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    log_path = run_root / "run.log"

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    log(f"[start] MASC-complete seed={cfg.seed} budget={cfg.budget} ipc={cfg.ipc} device={device}")
    resolved_config = asdict(cfg)
    resolved_config["ipc"] = cfg.ipc
    resolved_config["protocol"] = "homogeneous_masc_complete_current_class_split_v1"
    (run_root / "resolved_config.json").write_text(
        json.dumps(resolved_config, indent=2, sort_keys=True), encoding="utf-8"
    )
    train, test = load_data(cfg.data_dir)
    experts = train_or_load_experts(cfg, train, device, run_root, log)

    selected = communication_indices(train.targets, cfg.ipc, cfg.seed, cfg.class_split)
    (run_root / "communication_indices.json").write_text(json.dumps(selected), encoding="utf-8")
    cc_indices = [index for agent in sorted(selected) for index in selected[agent]]
    sender_counts = {
        agent: len(classes) * cfg.ipc for agent, classes in cfg.class_split.items()
    }
    log(f"[communication] {len(cc_indices)} real images total; per_sender={sender_counts}")

    student = CollectiveStudent(len(cfg.class_split))
    cc_path = run_root / "cc_student.pt"
    cc_meta = {
        "stage": "CC",
        "seed": cfg.seed,
        "budget": cfg.budget,
        "ipc": cfg.ipc,
        "epochs": cfg.cc_epochs,
        "lr": cfg.lr,
        "lambda_kd": cfg.lambda_kd,
        "lambda_align": cfg.lambda_align,
        "energy_anchor": cfg.energy_anchor,
        "temperature": cfg.temperature,
        "class_split": {str(agent): classes for agent, classes in cfg.class_split.items()},
    }
    if matching_checkpoint(cc_path, cc_meta, cfg.force):
        student.load_state_dict(torch.load(cc_path, map_location="cpu", weights_only=True))
        log(f"[CC] reuse {cc_path}")
    else:
        train_cc(student, experts, cfg.class_split, loader(train, cc_indices, cfg), cfg, device, log)
        save_checkpoint(cc_path, student.state_dict(), cc_meta)

    all_test = list(range(len(test)))
    before = {}
    for agent, expert in experts.items():
        expert.to(device).eval()
        own = indices_for_classes(test.targets, cfg.class_split[agent])
        if cfg.smoke:
            own = own[: cfg.eval_batch_size]
        before[agent] = accuracy(expert, test, own, cfg.eval_batch_size, device)

    grown_heads = {}
    for agent, classes in cfg.class_split.items():
        path = run_root / f"ra_agent_{agent}.pt"
        meta = {
            "stage": "RA",
            "agent": agent,
            "seed": cfg.seed,
            "budget": cfg.budget,
            "epochs": cfg.ra_epochs,
            "lr": cfg.lr,
            "lambda_kd": cfg.lambda_kd,
            "lambda_align": cfg.lambda_align,
            "energy_anchor": cfg.energy_anchor,
            "temperature": cfg.temperature,
        }
        head = LogitHead()
        if matching_checkpoint(path, meta, cfg.force):
            head.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            log(f"[RA/a{agent}] reuse {path}")
        else:
            train_ra(student, experts[agent], head, loader(train, indices_for_classes(train.targets, classes), cfg), cfg, device, log, agent)
            save_checkpoint(path, head.state_dict(), meta)
        grown_heads[agent] = head.to(device).eval()

    student.to(device).eval()
    rows = []
    expert_upload_bytes = sum(parameter_bytes(model) for model in experts.values())
    student_downlink_bytes = len(cfg.class_split) * parameter_bytes(student)
    model_interaction_bytes = expert_upload_bytes + student_downlink_bytes
    for agent, classes in cfg.class_split.items():
        own = indices_for_classes(test.targets, classes)
        novel = [index for index in all_test if int(test.targets[index]) not in set(classes)]
        global_indices = all_test
        if cfg.smoke:
            own = own[: cfg.eval_batch_size]
            novel = novel[: cfg.eval_batch_size]
            global_indices = all_test[: cfg.eval_batch_size]
        predict = lambda images, a=agent: hfe_logits(student, grown_heads[a], images)
        expert_acc = accuracy(predict, test, own, cfg.eval_batch_size, device)
        new_acc = accuracy(predict, test, novel, cfg.eval_batch_size, device)
        global_acc = accuracy(predict, test, global_indices, cfg.eval_batch_size, device)
        rows.append(
            {
                "method": "MASC-complete",
                "budget": cfg.budget,
                "ipc": cfg.ipc,
                "seed": cfg.seed,
                "agent_id": agent,
                "backbone": "official-Netwider13-homogeneous",
                "acc_global": global_acc,
                "acc_new": new_acc,
                "acc_expert": expert_acc,
                "forgetting": before[agent] - expert_acc,
                "balanced_avg": (expert_acc + new_acc) / 2,
                "expert_before": before[agent],
                "comm_images_per_sender": sender_counts[agent],
                "total_cc_images": len(cc_indices),
                "expert_upload_bytes_all_agents": expert_upload_bytes,
                "cc_student_downlink_bytes_all_agents": student_downlink_bytes,
                "model_parameter_bytes": model_interaction_bytes,
            }
        )
        log(f"[result/a{agent}] global={global_acc:.2f} new={new_acc:.2f} expert={expert_acc:.2f} forgetting={before[agent] - expert_acc:.2f}")
    average = dict(rows[0])
    average["agent_id"] = "avg"
    for key in ["acc_global", "acc_new", "acc_expert", "forgetting", "balanced_avg", "expert_before"]:
        average[key] = sum(float(row[key]) for row in rows) / len(rows)
    rows.append(average)
    output = run_root / "social_results.csv"
    write_rows(output, rows)
    log(f"[done] {output}")
    return output


def main() -> None:
    args = parse_args()
    if args.config is None:
        class_split = LEGACY_CLASS_SPLIT
        source_run_name = f"legacy_4agent25cls_seed{args.seed}"
        config_sha256 = None
    else:
        config_path = Path(args.config).resolve()
        project_cfg = load_config(config_path)
        class_split = get_agent_class_split(project_cfg)
        configured_seed = int(project_cfg.get("runtime", {}).get("seed", args.seed))
        if configured_seed != int(args.seed):
            raise ValueError(
                f"MASC --seed={args.seed} differs from config runtime.seed={configured_seed}"
            )
        source_run_name = str(project_cfg["project"]["run_name"])
        config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    validate_class_split(class_split)
    full_result = None
    for budget in args.budget:
        cfg = Settings(
            config=str(Path(args.config).resolve()) if args.config else None,
            config_sha256=config_sha256,
            source_run_name=source_run_name,
            class_split=class_split,
            budget=budget,
            seed=args.seed,
            data_dir=args.data_dir,
            output_root=args.output_root,
            teacher_epochs=1 if args.smoke else args.teacher_epochs,
            cc_epochs=1 if args.smoke else args.cc_epochs,
            ra_epochs=1 if args.smoke else args.ra_epochs,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            workers=args.workers,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            lambda_kd=args.lambda_kd,
            lambda_align=args.lambda_align,
            energy_anchor=args.energy_anchor,
            temperature=args.temperature,
            device=args.device,
            smoke=args.smoke,
            force=args.force,
        )
        result = run(cfg)
        if budget == "full":
            validate_full_result(result)
            full_result = result
        elif "full" in args.budget and full_result is None:
            raise RuntimeError("IPC MASC was requested before the full-data validation completed")


if __name__ == "__main__":
    main()
