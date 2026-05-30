"""
作用：
    本地 expert 训练阶段的入口脚本。当前实现最小真实训练版本：
    - 读取配置
    - 构建 CIFAR direct split
    - 逐个 agent 训练本地 anchor model
    - 保存 checkpoint

主要部分：
    1. parse_args(): 解析命令行参数
    2. train_one_agent(): 训练单个 agent 的本地 expert model
    3. main(): 读取配置、构建数据划分、启动训练

输入输出：
    - 输入：--config 指定的实验配置文件
    - 输出：每个 agent 的 checkpoint 与终端训练日志
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.cifar import (
    build_cifar_train_dataset,
    make_direct_class_splits,
    subset_by_classes,
)
from src.models.agent_model import AgentModel
from src.utils.config import load_yaml
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="实验配置文件路径")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        print("配置请求 cuda，但当前 PyTorch 无法使用 CUDA，自动切换到 CPU。")
        return torch.device("cpu")
    return torch.device(device_name)


def train_one_agent(agent_id: int, class_ids, cfg, train_dataset, device: torch.device, save_dir: Path):
    agent_subset = subset_by_classes(train_dataset, class_ids)
    loader = DataLoader(
        agent_subset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    model = AgentModel(
        model_name=cfg["model"]["name"],
        dataset=cfg["dataset"]["name"],
        num_classes=cfg["dataset"]["num_classes"],
        image_size=tuple(cfg["dataset"]["image_size"]),
        norm_type=cfg["model"]["norm_type"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg["train"]["lr"], momentum=0.9, weight_decay=5e-4)

    print(f"\n=== train agent_{agent_id} ===")
    print(f"class_ids: {class_ids}")
    print(f"samples: {len(agent_subset)}")

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        progress = tqdm(loader, desc=f"agent_{agent_id} epoch {epoch + 1}/{cfg['train']['epochs']}")
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_seen += batch_size

            progress.set_postfix(
                loss=f"{total_loss / total_seen:.4f}",
                acc=f"{total_correct / total_seen:.4f}",
            )

    save_path = save_dir / f"agent_{agent_id}_anchor.pt"
    torch.save(
        {
            "agent_id": agent_id,
            "class_ids": class_ids,
            "model_state_dict": model.state_dict(),
            "cfg": cfg,
        },
        save_path,
    )
    print(f"saved: {save_path}")


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    set_seed(cfg["seed"])
    device = resolve_device(cfg.get("device", "cpu"))

    if cfg["split"]["mode"] != "direct":
        raise NotImplementedError("当前 run_local_pretrain 先支持 direct split。")

    train_dataset = build_cifar_train_dataset(
        name=cfg["dataset"]["name"],
        root=cfg["dataset"]["root"],
        image_size=tuple(cfg["dataset"]["image_size"]),
        download=True,
    )
    class_splits = make_direct_class_splits(
        num_classes=cfg["dataset"]["num_classes"],
        num_agents=cfg["split"]["num_agents"],
        classes_per_agent=cfg["split"]["classes_per_agent"],
    )

    save_dir = (
        Path(cfg["output"]["root"])
        / "checkpoints"
        / "local_pretrain"
        / f"{cfg['dataset']['name']}_{cfg['split']['mode']}_{cfg['model']['name']}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=== run_local_pretrain ===")
    print(f"dataset: {cfg['dataset']['name']}")
    print(f"split mode: {cfg['split']['mode']}")
    print(f"model: {cfg['model']['name']}")
    print(f"num_classes: {cfg['dataset']['num_classes']}")
    print(f"device: {device}")
    print(f"save_dir: {save_dir}")

    for agent_id, class_ids in enumerate(class_splits):
        train_one_agent(
            agent_id=agent_id,
            class_ids=class_ids,
            cfg=cfg,
            train_dataset=train_dataset,
            device=device,
            save_dir=save_dir,
        )


if __name__ == "__main__":
    main()
