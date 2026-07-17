import argparse
import csv
import json
import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset, WeightedRandomSampler
from torchvision import datasets, transforms


AGENT_CLASS_SPLIT = {
    0: list(range(0, 25)),
    1: list(range(25, 50)),
    2: list(range(50, 75)),
    3: list(range(75, 100)),
}

HETERO_MODEL_SPLIT = {
    0: "convnet3w05",
    1: "convnet3w1",
    2: "convnet4w1",
    3: "convnet4w15",
}

MASC_MODEL_SPLIT = {
    0: "conv3",
    1: "conv3",
    2: "conv3",
    3: "conv3",
}

MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
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
    "seed",
    "run_name",
    "notes",
]


class ActiveClassDataset(Dataset):
    def __init__(self, dataset, classes):
        self.dataset = dataset
        self.classes = [int(c) for c in classes]
        class_set = set(self.classes)
        self.indices = [i for i, y in enumerate(dataset.targets) if int(y) in class_set]
        self.targets = [int(dataset.targets[i]) for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


class ConvNet(nn.Module):
    def __init__(self, num_classes=100, depth=3, width=1.0, norm="instance"):
        super().__init__()
        channels = max(1, int(round(128 * float(width))))
        layers = []
        in_channels = 3
        size = 32
        for _ in range(depth):
            layers.append(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1))
            if norm == "batch":
                layers.append(nn.BatchNorm2d(channels))
            elif norm == "instance":
                layers.append(nn.GroupNorm(channels, channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            in_channels = channels
            size //= 2
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(channels * size * size, num_classes)

    def forward(self, x, return_features=False):
        feat = self.features(x)
        flat = feat.reshape(feat.shape[0], -1)
        logits = self.classifier(flat)
        if return_features:
            return logits, flat
        return logits


def model_spec(model_name):
    if model_name in {"conv3", "convnet3w1"}:
        return 3, 1.0
    if model_name == "convnet3w05":
        return 3, 0.5
    if model_name == "convnet4w1":
        return 4, 1.0
    if model_name == "convnet4w15":
        return 4, 1.5
    raise ValueError(f"Unsupported model: {model_name}")


def build_model(model_name):
    depth, width = model_spec(model_name)
    return ConvNet(num_classes=100, depth=depth, width=width)


@dataclass
class RunConfig:
    method: str
    ipc: int
    seed: int
    data_dir: str
    output_root: str
    expert_epochs: int
    social_epochs: int
    anchor_steps: int
    batch_size: int
    lr: float
    social_lr: float
    anchor_lr: float
    kd_weight: float
    device: str
    smoke: bool


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_from_cfg(cfg):
    if cfg.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def repo_commit(path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def load_datasets(data_dir):
    train_raw = datasets.CIFAR100(data_dir, train=True, transform=transforms.ToTensor(), download=False)
    train_norm = datasets.CIFAR100(
        data_dir,
        train=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
        download=False,
    )
    test_norm = datasets.CIFAR100(
        data_dir,
        train=False,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
        download=False,
    )
    return train_raw, train_norm, test_norm


def subset_by_classes(dataset, classes):
    class_set = set(int(c) for c in classes)
    return Subset(dataset, [i for i, y in enumerate(dataset.targets) if int(y) in class_set])


def balanced_loader(images, labels, batch_size):
    labels = labels.long()
    counts = torch.bincount(labels, minlength=100).float()
    weights = torch.tensor([1.0 / max(1.0, counts[int(y)].item()) for y in labels], dtype=torch.float)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return DataLoader(TensorDataset(images.float(), labels), batch_size=batch_size, sampler=sampler, num_workers=0)


def normalize_images(images, device):
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    return (images.to(device).float() - mean) / std


def train_expert(cfg, agent_id, model_name, train_norm, out_dir, log):
    ckpt = out_dir / "experts" / f"agent_{agent_id}_{model_name}.pt"
    meta_path = ckpt.with_suffix(".json")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    expected_meta = {
        "seed": cfg.seed,
        "model_name": model_name,
        "expert_epochs": cfg.expert_epochs,
        "smoke": cfg.smoke,
        "classes": AGENT_CLASS_SPLIT[agent_id],
    }
    if ckpt.exists() and meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            existing_meta = json.load(f)
        if existing_meta == expected_meta:
            log(f"[expert] reuse {ckpt}")
            return ckpt
    if ckpt.exists():
        log(f"[expert] existing checkpoint metadata mismatch; retrain {ckpt}")
    device = device_from_cfg(cfg)
    model = build_model(model_name).to(device)
    ds = ActiveClassDataset(train_norm, AGENT_CLASS_SPLIT[agent_id])
    if cfg.smoke:
        ds = Subset(ds, list(range(min(len(ds), cfg.batch_size * 2))))
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    opt = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4)
    ce = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(cfg.expert_epochs):
        total_loss = 0.0
        total = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad()
            loss = ce(model(images), labels)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu()) * labels.numel()
            total += labels.numel()
        log(f"[expert] agent={agent_id} epoch={epoch + 1}/{cfg.expert_epochs} loss={total_loss / max(1, total):.4f}")
    torch.save(model.state_dict(), ckpt)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(expected_meta, f, indent=2, ensure_ascii=False)
    return ckpt


@torch.no_grad()
def evaluate(model, dataset, agent_id, cfg):
    device = device_from_cfg(cfg)
    model.eval()
    classes = AGENT_CLASS_SPLIT[agent_id]
    new_classes = [c for c in range(100) if c not in classes]
    loaders = {
        "global": DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=2),
        "expert": DataLoader(subset_by_classes(dataset, classes), batch_size=cfg.batch_size, shuffle=False, num_workers=2),
        "new": DataLoader(subset_by_classes(dataset, new_classes), batch_size=cfg.batch_size, shuffle=False, num_workers=2),
    }
    scores = {}
    for name, loader in loaders.items():
        correct = 0
        total = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            pred = model(images).argmax(1)
            correct += int((pred == labels).sum().item())
            total += int(labels.numel())
            if cfg.smoke and total >= cfg.batch_size * 2:
                break
        scores[name] = 100.0 * correct / max(1, total)
    return scores


def select_real_units(cfg, train_raw, agent_id):
    images = []
    labels = []
    by_class = {c: [] for c in AGENT_CLASS_SPLIT[agent_id]}
    for i, y in enumerate(train_raw.targets):
        y = int(y)
        if y in by_class and len(by_class[y]) < cfg.ipc:
            by_class[y].append(i)
    for c in AGENT_CLASS_SPLIT[agent_id]:
        for idx in by_class[c]:
            image, label = train_raw[idx]
            images.append(image)
            labels.append(int(label))
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


def make_desa_anchors(cfg, sender_model, train_raw, agent_id, packet_dir, log):
    packet_dir.mkdir(parents=True, exist_ok=True)
    path = packet_dir / f"agent_{agent_id}_desa_anchors.pt"
    expected_meta = {"method": "desa", "ipc": cfg.ipc, "seed": cfg.seed, "anchor_steps": cfg.anchor_steps, "smoke": cfg.smoke}
    if path.exists():
        packet = torch.load(path, map_location="cpu")
        if packet.get("adapter_meta") == expected_meta:
            log(f"[comm] reuse {path}")
            return packet
        log(f"[comm] existing packet metadata mismatch; rebuild {path}")
    device = device_from_cfg(cfg)
    images, labels = select_real_units(cfg, train_raw, agent_id)
    anchors = images.clone().to(device).requires_grad_(True)
    labels_dev = labels.to(device)
    opt = optim.SGD([anchors], lr=cfg.anchor_lr, momentum=0.5)
    ce = nn.CrossEntropyLoss()
    sender_model.eval()
    for step in range(cfg.anchor_steps):
        opt.zero_grad()
        logits = sender_model(normalize_images(anchors.clamp(0.0, 1.0), device))
        loss = ce(logits, labels_dev)
        loss.backward()
        opt.step()
        with torch.no_grad():
            anchors.clamp_(0.0, 1.0)
        if step == 0 or step == cfg.anchor_steps - 1:
            log(f"[desa-anchor] agent={agent_id} step={step + 1}/{cfg.anchor_steps} loss={float(loss.detach().cpu()):.4f}")
    packet = {"images": anchors.detach().cpu(), "labels": labels, "class_ids": AGENT_CLASS_SPLIT[agent_id], "adapter_meta": expected_meta}
    torch.save(packet, path)
    return packet


@torch.no_grad()
def make_fedre_units(cfg, sender_model, train_raw, agent_id, packet_dir, log):
    packet_dir.mkdir(parents=True, exist_ok=True)
    path = packet_dir / f"agent_{agent_id}_fedre_repr.pt"
    expected_meta = {"method": "fedre", "ipc": cfg.ipc, "seed": cfg.seed, "smoke": cfg.smoke}
    if path.exists():
        packet = torch.load(path, map_location="cpu")
        if packet.get("adapter_meta") == expected_meta:
            log(f"[comm] reuse {path}")
            return packet
        log(f"[comm] existing packet metadata mismatch; rebuild {path}")
    device = device_from_cfg(cfg)
    images, labels = select_real_units(cfg, train_raw, agent_id)
    sender_model.eval()
    logits_chunks = []
    feat_chunks = []
    for start in range(0, images.shape[0], cfg.batch_size):
        batch = normalize_images(images[start : start + cfg.batch_size], device)
        logits, features = sender_model(batch, return_features=True)
        logits_chunks.append(logits.cpu())
        feat_chunks.append(features.cpu())
    packet = {
        "images_ref_for_adaptation": images,
        "labels": labels,
        "class_ids": AGENT_CLASS_SPLIT[agent_id],
        "sender_logits": torch.cat(logits_chunks).half(),
        "sender_features": torch.cat(feat_chunks).half(),
        "adapter_meta": expected_meta,
        "note": "Representation units are stored; receiver training uses logits on the same IPC-limited unit identities.",
    }
    torch.save(packet, path)
    log(f"[fedre] agent={agent_id} representation_shape={tuple(packet['sender_features'].shape)}")
    return packet


def make_masc_units(cfg, train_raw, agent_id, packet_dir, log):
    packet_dir.mkdir(parents=True, exist_ok=True)
    path = packet_dir / f"agent_{agent_id}_masc_real_ipc.pt"
    expected_meta = {"method": "masc_conv3", "ipc": cfg.ipc, "seed": cfg.seed, "smoke": cfg.smoke}
    if path.exists():
        packet = torch.load(path, map_location="cpu")
        if packet.get("adapter_meta") == expected_meta:
            log(f"[comm] reuse {path}")
            return packet
        log(f"[comm] existing packet metadata mismatch; rebuild {path}")
    images, labels = select_real_units(cfg, train_raw, agent_id)
    packet = {"images": images, "labels": labels, "class_ids": AGENT_CLASS_SPLIT[agent_id], "adapter_meta": expected_meta}
    torch.save(packet, path)
    log(f"[masc] agent={agent_id} real_units={len(labels)}")
    return packet


def build_packets(cfg, method_name, model_split, train_raw, expert_paths, packet_dir, log):
    device = device_from_cfg(cfg)
    packets = {}
    for agent_id, model_name in model_split.items():
        sender = build_model(model_name).to(device)
        sender.load_state_dict(torch.load(expert_paths[agent_id], map_location=device))
        if method_name == "desa":
            packets[agent_id] = make_desa_anchors(cfg, sender, train_raw, agent_id, packet_dir, log)
        elif method_name == "fedre":
            packets[agent_id] = make_fedre_units(cfg, sender, train_raw, agent_id, packet_dir, log)
        elif method_name == "masc_conv3":
            packets[agent_id] = make_masc_units(cfg, train_raw, agent_id, packet_dir, log)
        else:
            raise ValueError(method_name)
    return packets


def train_receiver(cfg, method_name, receiver_id, model_name, expert_path, packets, out_dir, test_norm, log):
    device = device_from_cfg(cfg)
    model_old = build_model(model_name).to(device)
    model_new = build_model(model_name).to(device)
    state = torch.load(expert_path, map_location=device)
    model_old.load_state_dict(state)
    model_new.load_state_dict(state)
    before = evaluate(model_old, test_norm, receiver_id, cfg)

    train_images = []
    train_labels = []
    teacher_logits = []
    has_kd = method_name == "fedre"
    for sender_id, packet in packets.items():
        if method_name == "fedre":
            train_images.append(packet["images_ref_for_adaptation"])
            train_labels.append(packet["labels"])
            teacher_logits.append(packet["sender_logits"].float())
        else:
            train_images.append(packet["images"])
            train_labels.append(packet["labels"])
    images = torch.cat(train_images)
    labels = torch.cat(train_labels)
    loader = balanced_loader(normalize_images(images, torch.device("cpu")), labels, cfg.batch_size)
    if has_kd:
        all_teacher_logits = torch.cat(teacher_logits)
        kd_dataset = TensorDataset(normalize_images(images, torch.device("cpu")).float(), labels.long(), all_teacher_logits.float())
        loader = DataLoader(kd_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    opt = optim.SGD(model_new.parameters(), lr=cfg.social_lr, momentum=0.9, weight_decay=5e-4)
    ce = nn.CrossEntropyLoss()
    kl = nn.KLDivLoss(reduction="batchmean")
    model_new.train()
    for epoch in range(cfg.social_epochs):
        total_loss = 0.0
        total = 0
        for batch in loader:
            if has_kd:
                batch_images, batch_labels, batch_teacher = batch
                batch_teacher = batch_teacher.to(device)
            else:
                batch_images, batch_labels = batch
                batch_teacher = None
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            opt.zero_grad()
            logits = model_new(batch_images)
            loss = ce(logits, batch_labels)
            if has_kd:
                loss = loss + cfg.kd_weight * kl(torch.log_softmax(logits, dim=1), torch.softmax(batch_teacher, dim=1))
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu()) * batch_labels.numel()
            total += batch_labels.numel()
        log(f"[social] method={method_name} receiver={receiver_id} epoch={epoch + 1}/{cfg.social_epochs} loss={total_loss / max(1, total):.4f}")

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model_old.state_dict(), ckpt_dir / f"receiver_{receiver_id}_before_social.pt")
    torch.save(model_new.state_dict(), ckpt_dir / f"receiver_{receiver_id}_after_social.pt")
    after = evaluate(model_new, test_norm, receiver_id, cfg)
    return before, after


def row_for(cfg, method_name, receiver_id, model_name, before, after):
    method_labels = {
        "desa": "DeSA-adapted",
        "fedre": "FedRE",
        "masc_conv3": "MASC-adapted-Conv3",
    }
    comm_types = {
        "desa": "synthetic_anchors",
        "fedre": "representation_units",
        "masc_conv3": "ipc_limited_real_collaboration",
    }
    model_setting = "homo_conv3" if method_name == "masc_conv3" else "heterogeneous_conv_family"
    external_units = 3 * 25 * cfg.ipc
    expert = after["expert"]
    new = after["new"]
    return {
        "method": method_labels[method_name],
        "ipc": cfg.ipc,
        "agent_id": receiver_id,
        "model_setting": model_setting,
        "agent_model": model_name,
        "comm_type": comm_types[method_name],
        "comm_units_per_sender": 25 * cfg.ipc,
        "external_comm_units": external_units,
        "global_acc": after["global"],
        "expert_acc": expert,
        "new_acc": new,
        "balanced_avg": (expert + new) / 2.0,
        "directionality": expert - new,
        "forgetting": before["expert"] - after["expert"],
        "seed": cfg.seed,
        "run_name": f"{method_name}_ipc{cfg.ipc}_seed{cfg.seed}",
        "notes": "global CIFAR-100 labels; receiver excludes no labels; communication capped at 25*IPC units per sender",
    }


def average_row(rows):
    avg = dict(rows[0])
    avg["agent_id"] = "avg"
    avg["agent_model"] = "avg"
    numeric = ["global_acc", "expert_acc", "new_acc", "balanced_avg", "directionality", "forgetting"]
    for key in numeric:
        avg[key] = sum(float(r[key]) for r in rows) / len(rows)
    return avg


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_reproduction_log(root, text):
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "reproduction_log.md", "a", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")


def run_one(cfg):
    set_seed(cfg.seed)
    method_name = cfg.method
    model_split = MASC_MODEL_SPLIT if method_name == "masc_conv3" else HETERO_MODEL_SPLIT
    base_root = Path(cfg.output_root) / "baselines_cifar100_conv_family"
    run_root = base_root / method_name / f"ipc{cfg.ipc}"
    metrics_path = run_root / "metrics" / "social_results.csv"
    log_path = run_root / "logs" / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(f"[start] {method_name} ipc={cfg.ipc} seed={cfg.seed} smoke={cfg.smoke}")
    log(f"[env] python={sys.executable}")
    log(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()}")
    train_raw, train_norm, test_norm = load_datasets(cfg.data_dir)
    config_dir = run_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    with open(config_dir / "resolved_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)
    commits = {
        "DESA": repo_commit(Path.cwd() / "DESA"),
        "FedRE": repo_commit(Path.cwd() / "FedRE"),
        "SL": repo_commit(Path.cwd() / "SL"),
        "adapter_worktree": repo_commit(Path.cwd()),
    }
    with open(config_dir / "commits.json", "w", encoding="utf-8") as f:
        json.dump(commits, f, indent=2, ensure_ascii=False)

    expert_paths = {}
    for agent_id, model_name in model_split.items():
        expert_paths[agent_id] = train_expert(cfg, agent_id, model_name, train_norm, run_root, log)
    packets = build_packets(cfg, method_name, model_split, train_raw, expert_paths, run_root / "packets", log)
    rows = []
    for receiver_id, model_name in model_split.items():
        before, after = train_receiver(cfg, method_name, receiver_id, model_name, expert_paths[receiver_id], packets, run_root, test_norm, log)
        rows.append(row_for(cfg, method_name, receiver_id, model_name, before, after))
        write_csv(metrics_path, rows + [average_row(rows)])
    rows.append(average_row(rows))
    write_csv(metrics_path, rows)
    append_reproduction_log(
        base_root,
        (
            f"## {method_name} ipc={cfg.ipc} seed={cfg.seed}\n"
            f"1. Method adapted: {rows[0]['method']}\n"
            f"2. Source repository and commit: {commits}\n"
            f"3. Task setting confirmed: CIFAR-100, 4 agents, 25 class-disjoint classes, global labels 0-99\n"
            f"4. Communication budget confirmed: 25*IPC units per sender, external units per receiver={3 * 25 * cfg.ipc}\n"
            f"5. Command used: {' '.join(sys.argv)}\n"
            f"6. Output path: {metrics_path}\n"
            f"7. Current result or blocking error: completed\n"
        ),
    )
    log(f"[done] metrics={metrics_path}")
    return metrics_path


def consolidate(output_root):
    base_root = Path(output_root) / "baselines_cifar100_conv_family"
    rows = []
    for path in sorted(base_root.glob("*/ipc*/metrics/social_results.csv")):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)
    if rows:
        write_csv(base_root / "main_table_baselines.csv", rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["desa", "fedre", "masc_conv3", "all"], default="all")
    parser.add_argument("--ipc", type=int, nargs="+", default=[10, 50])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-root", default="./outputs")
    parser.add_argument("--expert-epochs", type=int, default=20)
    parser.add_argument("--social-epochs", type=int, default=20)
    parser.add_argument("--anchor-steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--social-lr", type=float, default=0.005)
    parser.add_argument("--anchor-lr", type=float, default=0.2)
    parser.add_argument("--kd-weight", type=float, default=0.5)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    methods = ["desa", "fedre", "masc_conv3"] if args.method == "all" else [args.method]
    for method in methods:
        for ipc in args.ipc:
            cfg = RunConfig(
                method=method,
                ipc=int(ipc),
                seed=args.seed,
                data_dir=args.data_dir,
                output_root=args.output_root,
                expert_epochs=1 if args.smoke else args.expert_epochs,
                social_epochs=1 if args.smoke else args.social_epochs,
                anchor_steps=2 if args.smoke else args.anchor_steps,
                batch_size=args.batch_size,
                lr=args.lr,
                social_lr=args.social_lr,
                anchor_lr=args.anchor_lr,
                kd_weight=args.kd_weight,
                device=args.device,
                smoke=args.smoke,
            )
            run_one(cfg)
    consolidate(args.output_root)


if __name__ == "__main__":
    main()
