#!/usr/bin/env python3
"""复现 MASC CIFAR-100 流程，仅将通信图像限制为每类 IPC 张。"""

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


REPO_ROOT = Path(__file__).resolve().parents[1]
MASC_ROOT = REPO_ROOT / "external_baselines" / "repos" / "MASC_SL" / "MASC_SL_inference_CIFAR100_raw_4_25"
if str(MASC_ROOT) not in sys.path:
    sys.path.insert(0, str(MASC_ROOT))

from utils.network_wider_cifar100 import Netwider, Netwider_multi  # noqa: E402


AGENT_CLASS_SPLIT = {
    0: list(range(0, 25)),
    1: list(range(25, 50)),
    2: list(range(50, 75)),
    3: list(range(75, 100)),
}

# 官方 MASC CIFAR100 dataloader 内部使用 ImageNet 归一化参数。
MASC_MEAN = (0.485, 0.456, 0.406)
MASC_STD = (0.229, 0.224, 0.225)

CSV_FIELDS = [
    "method",
    "ipc",
    "agent_id",
    "model_setting",
    "agent_model",
    "comm_type",
    "comm_units_per_sender",
    "external_comm_units",
    "global_acc",
    "expert_acc",
    "new_acc",
    "balanced_avg",
    "directionality",
    "forgetting",
    "teacher_expert_acc_before",
    "generalist_global_acc",
    "seed",
    "run_name",
    "notes",
]


class ActiveClassDataset(Dataset):
    """只暴露指定 CIFAR-100 全局类别的数据。"""

    def __init__(self, dataset, classes):
        self.dataset = dataset
        self.classes = [int(c) for c in classes]
        class_set = set(self.classes)
        self.indices = [i for i, y in enumerate(dataset.targets) if int(y) in class_set]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


class IndicesDataset(Dataset):
    """用固定样本索引构造 IPC-limited 通信训练集。"""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


@dataclass
class RunConfig:
    ipc: int
    seed: int
    data_dir: str
    output_root: str
    teacher_epochs: int
    generalist_epochs: int
    batch_size: int
    eval_batch_size: int
    lr: float
    generalist_lr: float
    kd_weight: float
    momentum: float
    weight_decay: float
    device: str
    smoke: bool
    force: bool
    stage: str
    only_agent: int | None


def set_seed(seed):
    """固定随机种子，保证 IPC 样本选择和训练可复现。"""

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cfg):
    """根据配置返回本次训练使用的设备。"""

    if cfg.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_cifar100(data_dir):
    """加载 CIFAR-100，并复刻 MASC 官方训练预处理。"""

    train_transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MASC_MEAN, MASC_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(MASC_MEAN, MASC_STD),
        ]
    )
    train_aug = datasets.CIFAR100(data_dir, train=True, transform=train_transform, download=False)
    train_eval = datasets.CIFAR100(data_dir, train=True, transform=eval_transform, download=False)
    test_eval = datasets.CIFAR100(data_dir, train=False, transform=eval_transform, download=False)
    return train_aug, train_eval, test_eval


def subset_by_classes(dataset, classes):
    """按 CIFAR-100 全局标签筛出指定类别子集。"""

    class_set = set(int(c) for c in classes)
    return Subset(dataset, [i for i, y in enumerate(dataset.targets) if int(y) in class_set])


def select_ipc_indices(dataset, ipc, seed):
    """从每个 agent 的 expert 类中选择每类 IPC 张通信图片。"""

    rng = random.Random(seed)
    by_agent = {}
    for agent_id, classes in AGENT_CLASS_SPLIT.items():
        agent_indices = []
        for cls in classes:
            indices = [i for i, y in enumerate(dataset.targets) if int(y) == int(cls)]
            rng.shuffle(indices)
            agent_indices.extend(indices[:ipc])
        by_agent[agent_id] = agent_indices
    return by_agent


def train_model(model, loader, epochs, lr, cfg, log, tag):
    """使用官方 SGD+CE 训练 teacher 或 generalist。"""

    device = get_device(cfg)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * labels.numel()
            correct += int((outputs.argmax(1) == labels).sum().item())
            total += int(labels.numel())
        log(f"[{tag}] epoch={epoch + 1}/{epochs} loss={total_loss / max(1, total):.4f} acc={100.0 * correct / max(1, total):.2f}")


def train_generalist_with_kd(model, teachers, loader, epochs, lr, cfg, log):
    """复刻 MASC train_kd：generalist 使用 CE + 四个 teacher KL 蒸馏训练。"""

    device = get_device(cfg)
    model.to(device)
    for teacher in teachers.values():
        teacher.to(device).eval()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_kd = 0.0
        total = 0
        correct = 0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            ce_loss = criterion(outputs, labels)
            kd_loss = 0.0
            with torch.no_grad():
                teacher_outputs = [teacher(images) for teacher in teachers.values()]
            for teacher_output in teacher_outputs:
                kd_loss = kd_loss + F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(teacher_output, dim=1))
            loss = ce_loss + cfg.kd_weight * kd_loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * labels.numel()
            total_ce += float(ce_loss.detach().cpu()) * labels.numel()
            total_kd += float(kd_loss.detach().cpu()) * labels.numel()
            correct += int((outputs.argmax(1) == labels).sum().item())
            total += int(labels.numel())
        log(
            "[generalist] epoch={epoch}/{epochs} loss={loss:.4f} ce={ce:.4f} kd={kd:.4f} acc={acc:.2f}".format(
                epoch=epoch + 1,
                epochs=epochs,
                loss=total_loss / max(1, total),
                ce=total_ce / max(1, total),
                kd=total_kd / max(1, total),
                acc=100.0 * correct / max(1, total),
            )
        )


@torch.no_grad()
def logits_before_softmax(model, images):
    """复刻官方做法：临时移除 classifier 最后的 Softmax 得到 energy logits。"""

    classifier_save = model.classifier
    model.classifier = nn.ModuleList(classifier_save[:-1])
    outputs = model(images)
    model.classifier = classifier_save
    return outputs


@torch.no_grad()
def evaluate_plain(model, dataset, cfg):
    device = get_device(cfg)
    model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        correct += int((outputs.argmax(1) == labels).sum().item())
        total += int(labels.numel())
        if cfg.smoke and total >= cfg.eval_batch_size * 2:
            break
    return 100.0 * correct / max(1, total)


@torch.no_grad()
def evaluate_masc_gate(teacher, generalist, dataset, cfg):
    """复刻官方 test_ewc_ts：按 logsumexp energy 在 teacher/student 间逐样本选择输出。"""

    device = get_device(cfg)
    teacher.to(device).eval()
    generalist.to(device).eval()
    loader = DataLoader(dataset, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs_s = generalist(images)
        outputs_t = teacher(images)
        logits_s = logits_before_softmax(generalist, images)
        logits_t = logits_before_softmax(teacher, images)
        logit_stack = torch.stack((logits_s, logits_t))
        outputs_stack = torch.stack((outputs_s, outputs_t))
        energy = torch.logsumexp(logit_stack, dim=-1)
        _, selected_model = torch.max(energy, dim=0)
        output_all = outputs_stack.permute(1, 0, 2).contiguous().view(-1, 100)
        row_offsets = torch.arange(selected_model.size(0), device=selected_model.device) * 2
        outputs = output_all[selected_model + row_offsets]
        correct += int((outputs.argmax(1) == labels).sum().item())
        total += int(labels.numel())
        if cfg.smoke and total >= cfg.eval_batch_size * 2:
            break
    return 100.0 * correct / max(1, total)


def train_or_load_teacher(cfg, agent_id, train_aug, teacher_root, log):
    """训练或复用单个 agent 的本地 MASC teacher。"""

    ckpt_path = teacher_root / "checkpoints" / f"teacher_agent_{agent_id}.pt"
    meta_path = ckpt_path.with_suffix(".json")
    expected_meta = {
        "stage": "masc_teacher",
        "agent_id": agent_id,
        "classes": AGENT_CLASS_SPLIT[agent_id],
        "seed": cfg.seed,
        "teacher_epochs": cfg.teacher_epochs,
        "smoke": cfg.smoke,
    }
    model = Netwider(13)
    if ckpt_path.exists() and meta_path.exists() and not cfg.force:
        with open(meta_path, encoding="utf-8") as f:
            if json.load(f) == expected_meta:
                log(f"[teacher] reuse agent={agent_id} {ckpt_path}")
                model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                return model
    ds = ActiveClassDataset(train_aug, AGENT_CLASS_SPLIT[agent_id])
    if cfg.smoke:
        ds = Subset(ds, list(range(min(len(ds), cfg.batch_size * 2))))
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=(get_device(cfg).type == "cuda"))
    log(f"[teacher] train agent={agent_id} classes={AGENT_CLASS_SPLIT[agent_id][0]}-{AGENT_CLASS_SPLIT[agent_id][-1]} samples={len(ds)}")
    train_model(model, loader, cfg.teacher_epochs, cfg.lr, cfg, log, f"teacher/a{agent_id}")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(expected_meta, f, indent=2, ensure_ascii=False)
    return model


def train_or_load_generalist(cfg, train_aug, ipc_indices_by_agent, teachers, run_root, log):
    """用聚合后的 IPC 通信图训练 MASC generalist/student。"""

    ckpt_path = run_root / "checkpoints" / "generalist_netwider_multi.pt"
    meta_path = ckpt_path.with_suffix(".json")
    expected_meta = {
        "stage": "masc_generalist",
        "seed": cfg.seed,
        "ipc": cfg.ipc,
        "generalist_epochs": cfg.generalist_epochs,
        "kd_weight": cfg.kd_weight,
        "smoke": cfg.smoke,
        "comm_images_per_sender": 25 * cfg.ipc,
        "total_comm_images": 4 * 25 * cfg.ipc,
    }
    model = Netwider_multi(13)
    if ckpt_path.exists() and meta_path.exists() and not cfg.force:
        with open(meta_path, encoding="utf-8") as f:
            if json.load(f) == expected_meta:
                log(f"[generalist] reuse {ckpt_path}")
                model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                return model
    indices = []
    for agent_id in sorted(ipc_indices_by_agent):
        indices.extend(ipc_indices_by_agent[agent_id])
    if cfg.smoke:
        indices = indices[: min(len(indices), cfg.batch_size * 2)]
    ds = IndicesDataset(train_aug, indices)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=(get_device(cfg).type == "cuda"))
    log(f"[generalist] train aggregated IPC data samples={len(ds)} ipc={cfg.ipc}")
    train_generalist_with_kd(model, teachers, loader, cfg.generalist_epochs, cfg.generalist_lr, cfg, log)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(expected_meta, f, indent=2, ensure_ascii=False)
    return model


def load_teachers(cfg, train_aug, teacher_root, log):
    """训练或加载四个本地 teacher，并返回 expert-before 指标。"""

    teachers = {}
    for agent_id in sorted(AGENT_CLASS_SPLIT):
        teachers[agent_id] = train_or_load_teacher(cfg, agent_id, train_aug, teacher_root, log)
    return teachers


def make_row(cfg, agent_id, teacher_before, generalist_global, scores):
    """整理单个 receiver 的 MASC 指标行。"""

    expert = scores["expert"]
    new = scores["new"]
    return {
        "method": "MASC-official-IPC",
        "ipc": cfg.ipc,
        "agent_id": agent_id,
        "model_setting": "homo_netwider_official",
        "agent_model": "Netwider(13)+Netwider_multi(13)",
        "comm_type": "ipc_limited_real_images_for_generalist",
        "comm_units_per_sender": 25 * cfg.ipc,
        "external_comm_units": 3 * 25 * cfg.ipc,
        "global_acc": scores["global"],
        "expert_acc": expert,
        "new_acc": new,
        "balanced_avg": (expert + new) / 2.0,
        "directionality": expert - new,
        "forgetting": teacher_before - expert,
        "teacher_expert_acc_before": teacher_before,
        "generalist_global_acc": generalist_global,
        "seed": cfg.seed,
        "run_name": f"masc_official_ipc{cfg.ipc}_seed{cfg.seed}",
        "notes": f"Official MASC Netwider flow reconstructed from released CIFAR100 code; only communication images are capped to IPC/class; generalist uses CE+{cfg.kd_weight}*4-teacher-KL.",
    }


def average_row(rows):
    """计算四个 receiver 的平均结果。"""

    avg = dict(rows[0])
    avg["agent_id"] = "avg"
    numeric_keys = [
        "global_acc",
        "expert_acc",
        "new_acc",
        "balanced_avg",
        "directionality",
        "forgetting",
        "teacher_expert_acc_before",
        "generalist_global_acc",
    ]
    for key in numeric_keys:
        avg[key] = sum(float(row[key]) for row in rows) / len(rows)
    return avg


def write_csv(path, rows):
    """写出实验指标 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_one(cfg):
    """执行一个 IPC/seed 的完整 MASC 复现实验。"""

    set_seed(cfg.seed)
    output_root = Path(cfg.output_root) / "masc_official_ipc"
    run_root = output_root / f"ipc{cfg.ipc}_seed{cfg.seed}"
    teacher_root = output_root / f"teachers_seed{cfg.seed}"
    if cfg.stage == "teachers" and cfg.only_agent is not None:
        log_path = teacher_root / "logs" / f"teacher_agent_{cfg.only_agent}.log"
    else:
        log_path = run_root / "logs" / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(message):
        print(message, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    log(f"[start] MASC official IPC reproduction ipc={cfg.ipc} seed={cfg.seed} smoke={cfg.smoke}")
    log(f"[env] python={sys.executable}")
    log(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()} device={get_device(cfg)}")
    train_aug, train_eval, test_eval = load_cifar100(cfg.data_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    with open(run_root / "resolved_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    if cfg.stage == "teachers" and cfg.only_agent is not None:
        teacher = train_or_load_teacher(cfg, cfg.only_agent, train_aug, teacher_root, log)
        expert_acc = evaluate_plain(teacher, subset_by_classes(test_eval, AGENT_CLASS_SPLIT[cfg.only_agent]), cfg)
        log(f"[teacher-eval] agent={cfg.only_agent} expert_acc={expert_acc:.2f}")
        log(f"[done] teacher={teacher_root / 'checkpoints' / f'teacher_agent_{cfg.only_agent}.pt'}")
        return teacher_root / "checkpoints" / f"teacher_agent_{cfg.only_agent}.pt"

    ipc_indices_by_agent = select_ipc_indices(train_eval, cfg.ipc, cfg.seed)
    with open(run_root / "communication_indices.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in ipc_indices_by_agent.items()}, f)
    for agent_id, indices in ipc_indices_by_agent.items():
        log(f"[comm] sender={agent_id} transmitted_images={len(indices)} expected={25 * cfg.ipc}")

    teachers = load_teachers(cfg, train_aug, teacher_root, log)
    teacher_before = {}
    for agent_id, teacher in teachers.items():
        teacher_before[agent_id] = evaluate_plain(teacher, subset_by_classes(test_eval, AGENT_CLASS_SPLIT[agent_id]), cfg)
        log(f"[teacher-eval] agent={agent_id} expert_acc={teacher_before[agent_id]:.2f}")
    if cfg.stage == "teachers":
        log(f"[done] teachers={teacher_root / 'checkpoints'}")
        return teacher_root / "checkpoints"

    generalist = train_or_load_generalist(cfg, train_aug, ipc_indices_by_agent, teachers, run_root, log)
    generalist_global = evaluate_plain(generalist, test_eval, cfg)
    log(f"[generalist-eval] global_acc={generalist_global:.2f}")

    rows = []
    for receiver_id in sorted(AGENT_CLASS_SPLIT):
        expert_classes = AGENT_CLASS_SPLIT[receiver_id]
        new_classes = [c for c in range(100) if c not in expert_classes]
        scores = {
            "global": evaluate_masc_gate(teachers[receiver_id], generalist, test_eval, cfg),
            "expert": evaluate_masc_gate(teachers[receiver_id], generalist, subset_by_classes(test_eval, expert_classes), cfg),
            "new": evaluate_masc_gate(teachers[receiver_id], generalist, subset_by_classes(test_eval, new_classes), cfg),
        }
        row = make_row(cfg, receiver_id, teacher_before[receiver_id], generalist_global, scores)
        rows.append(row)
        write_csv(run_root / "metrics" / "social_results.csv", rows + [average_row(rows)])
        log(
            "[masc-eval] receiver={rid} global={global_acc:.2f} expert={expert_acc:.2f} "
            "new={new_acc:.2f} forgetting={forgetting:.2f}".format(rid=receiver_id, **row)
        )

    rows.append(average_row(rows))
    write_csv(run_root / "metrics" / "social_results.csv", rows)
    log(f"[done] metrics={run_root / 'metrics' / 'social_results.csv'}")
    return run_root / "metrics" / "social_results.csv"


def parse_args():
    """解析命令行参数。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--ipc", type=int, nargs="+", default=[10])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--output-root", default=str(REPO_ROOT / "outputs"))
    parser.add_argument("--teacher-epochs", type=int, default=100)
    parser.add_argument("--generalist-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--generalist-lr", type=float, default=0.005)
    parser.add_argument("--kd-weight", type=float, default=1.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stage", choices=["all", "teachers"], default="all")
    parser.add_argument("--only-agent", type=int, choices=[0, 1, 2, 3], default=None)
    return parser.parse_args()


def main():
    """脚本入口，支持一次运行多个 IPC。"""

    args = parse_args()
    if not args.smoke and args.stage == "all":
        # Compatibility for the already-running one-ResNet shell queue.  The
        # old adapter omitted RA and energy alignment; redirect formal calls
        # to the complete paper reproduction and require the full-data gate.
        command = [
            sys.executable,
            str(REPO_ROOT / "baseline_adapters/run_masc_complete.py"),
            "--seed",
            str(args.seed),
            "--budget",
            "full",
            *[str(ipc) for ipc in args.ipc],
        ]
        os.execv(sys.executable, command)
    for ipc in args.ipc:
        cfg = RunConfig(
            ipc=int(ipc),
            seed=args.seed,
            data_dir=args.data_dir,
            output_root=args.output_root,
            teacher_epochs=1 if args.smoke else args.teacher_epochs,
            generalist_epochs=1 if args.smoke else args.generalist_epochs,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            lr=args.lr,
            generalist_lr=args.generalist_lr,
            kd_weight=args.kd_weight,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            device=args.device,
            smoke=args.smoke,
            force=args.force,
            stage=args.stage,
            only_agent=args.only_agent,
        )
        run_one(cfg)


if __name__ == "__main__":
    main()
