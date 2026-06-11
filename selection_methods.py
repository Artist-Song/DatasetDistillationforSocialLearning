from collections import defaultdict

import torch


def _get_target(train_set, index):
    """读取数据集样本标签，兼容 targets 和 labels。"""
    if hasattr(train_set, "targets"):
        target = train_set.targets[index]
    elif hasattr(train_set, "labels"):
        target = train_set.labels[index]
    else:
        _, target = train_set[index]
    return int(target)


def _collect_indices_by_class(train_set):
    """按类别收集真实训练集样本索引。"""
    indices_by_class = defaultdict(list)
    for index in range(len(train_set)):
        indices_by_class[_get_target(train_set, index)].append(index)
    return indices_by_class


def _stack_samples(train_set, indices):
    """根据索引从真实训练集中堆叠图片和标签。"""
    images = []
    labels = []
    for index in indices:
        image, label = train_set[index]
        images.append(image.detach().cpu() if torch.is_tensor(image) else image)
        labels.append(int(label))
    return torch.stack(images).float(), torch.tensor(labels, dtype=torch.long)


def build_heuristic_packet(args, train_set):
    """每类随机选择 ipc 张真实样本，构建 Heuristic packet。"""
    generator = torch.Generator().manual_seed(int(args.seed))
    indices_by_class = _collect_indices_by_class(train_set)
    selected = []
    class_ids = sorted(indices_by_class.keys())
    for class_id in class_ids:
        candidates = indices_by_class[class_id]
        order = torch.randperm(len(candidates), generator=generator).tolist()
        selected.extend([candidates[i] for i in order[: args.ipc]])
    images, labels = _stack_samples(train_set, selected)
    return images, labels, class_ids


def _score_confidence(args, images, guide_models, device):
    """用 guide ensemble 计算每张图片的平均最大 softmax 置信度。"""
    batch = images.to(device)
    scores = torch.zeros(batch.shape[0], device=device)
    for model in guide_models:
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(batch), dim=1)
            scores += probs.max(dim=1).values
    scores = scores / max(1, len(guide_models))
    return scores.detach().cpu()


def build_importance_packet(args, train_set, guide_models):
    """使用 guide ensemble 选择低置信度样本，构建 Importance packet。"""
    if not guide_models:
        images, labels, class_ids = build_heuristic_packet(args, train_set)
        return images, labels, class_ids

    device = torch.device("cuda" if getattr(args, "device", "cuda") == "cuda" and torch.cuda.is_available() else "cpu")
    for model in guide_models:
        model.to(device)

    indices_by_class = _collect_indices_by_class(train_set)
    selected = []
    class_ids = sorted(indices_by_class.keys())
    for class_id in class_ids:
        candidates = indices_by_class[class_id]
        images, _ = _stack_samples(train_set, candidates)
        scores = _score_confidence(args, images, guide_models, device)
        order = torch.argsort(scores, descending=False).tolist()
        selected.extend([candidates[i] for i in order[: args.ipc]])
    images, labels = _stack_samples(train_set, selected)
    return images, labels, class_ids
