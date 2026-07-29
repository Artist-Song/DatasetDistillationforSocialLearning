import torch
from torch.utils.data import DataLoader, Subset

from agent_data import get_agent_class_split, get_num_classes, get_test_dataset


def _subset_by_classes(dataset, class_ids):
    """根据全局类别筛选测试集样本。"""
    targets = getattr(dataset, "targets")
    indices = [i for i, y in enumerate(targets) if int(y) in set(class_ids)]
    return Subset(dataset, indices)


def compute_accuracy(model, loader, device, allowed_class_ids=None):
    """计算模型在给定 loader 上的 top-1 准确率。"""
    model.eval()
    allowed_class_ids = None if allowed_class_ids is None else [int(c) for c in allowed_class_ids]
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            if allowed_class_ids is not None:
                allowed = torch.zeros(logits.shape[1], dtype=torch.bool, device=logits.device)
                allowed[allowed_class_ids] = True
                logits = logits.masked_fill(~allowed.view(1, -1), float("-inf"))
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.numel()
    return 100.0 * correct / max(1, total)


def evaluate_receiver_model(args, model, receiver_agent, device, expert_mask=False):
    """评估 receiver 的 global/expert/new 三类准确率。"""
    dataset = get_test_dataset(args)
    expert_classes = get_agent_class_split(args)[int(receiver_agent)]
    new_classes = [c for c in range(get_num_classes(args)) if c not in expert_classes]
    global_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    expert_loader = DataLoader(_subset_by_classes(dataset, expert_classes), batch_size=args.batch_size, shuffle=False)
    new_loader = DataLoader(_subset_by_classes(dataset, new_classes), batch_size=args.batch_size, shuffle=False)
    allowed_class_ids = expert_classes if expert_mask else None
    return {
        "acc_global": compute_accuracy(model, global_loader, device, allowed_class_ids),
        "acc_expert": compute_accuracy(model, expert_loader, device, allowed_class_ids),
        "acc_new": compute_accuracy(model, new_loader, device, allowed_class_ids),
    }
