from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

from config_adapter import build_dsdm_args_from_config


AGENT_CLASS_SPLIT = {
    0: [0, 1],
    1: [2, 3],
    2: [4, 5],
    3: [6, 7],
    4: [8, 9],
}

AGENT_MODEL_SPLIT = {
    0: "convnet",
    1: "convnet",
    2: "resnet",
    3: "resnet",
    4: "resnet_ap",
}


class ActiveClassDataset(Dataset):
    """只暴露指定类别样本，同时保留 CIFAR-10 全局标签。"""

    def __init__(self, dataset, active_class_ids):
        """根据 active_class_ids 过滤数据集索引。"""
        self.dataset = dataset
        self.active_class_ids = [int(c) for c in active_class_ids]
        targets = getattr(dataset, "targets", getattr(dataset, "labels", None))
        if targets is None:
            raise ValueError("数据集缺少 targets/labels 字段")
        self.indices = [i for i, y in enumerate(targets) if int(y) in self.active_class_ids]
        self.targets = [int(targets[i]) for i in self.indices]
        self.nclass = 10

    def __len__(self):
        """返回过滤后的样本数。"""
        return len(self.indices)

    def __getitem__(self, index):
        """返回过滤后的样本和全局标签。"""
        return self.dataset[self.indices[index]]


def get_agent_ids(only_agent=None):
    """返回本次需要处理的 agent id 列表。"""
    if only_agent is None:
        return list(AGENT_CLASS_SPLIT.keys())
    return [int(only_agent)]


def get_receiver_ids(only_receiver=None):
    """返回本次需要处理的 receiver id 列表。"""
    if only_receiver is None:
        return list(AGENT_CLASS_SPLIT.keys())
    return [int(only_receiver)]


def build_agent_args(base_cfg, config_path, agent_id):
    """基于主配置构造单个 agent 的 DSDM args。"""
    args = build_dsdm_args_from_config(base_cfg, config_path=config_path)
    args.agent_id = int(agent_id)
    args.num_classes = 10
    args.nclass = 10
    args.active_class_ids = list(AGENT_CLASS_SPLIT[int(agent_id)])
    args.net_type = AGENT_MODEL_SPLIT[int(agent_id)]
    if args.net_type in {"resnet", "resnet_ap"}:
        args.depth = 10
    args.sender_model = args.net_type
    args.save_pretrain_dir = str(get_agent_dir(args, agent_id) / "checkpoints")
    args.save_dir = str(get_agent_dir(args, agent_id) / "checkpoints")
    return args


def get_run_dir(args):
    """返回当前 run 的输出根目录。"""
    return Path(args.output_root) / args.run_name


def get_agent_dir(args, agent_id):
    """返回单个 agent 的输出目录。"""
    return get_run_dir(args) / "agents" / f"agent_{int(agent_id)}"


def get_receiver_dir(args, receiver_id):
    """返回单个 receiver 的输出目录。"""
    return get_run_dir(args) / "social_learning" / f"receiver_agent_{int(receiver_id)}"


def get_cifar10_train_dataset(args, normalize=False):
    """加载 CIFAR-10 训练集，标签保持全局编号。"""
    transform_list = [transforms.ToTensor()]
    if normalize:
        from data import MEANS, STDS

        transform_list.append(transforms.Normalize(mean=MEANS["cifar10"], std=STDS["cifar10"]))
    return datasets.CIFAR10(args.data_dir, train=True, transform=transforms.Compose(transform_list))


def get_cifar10_test_dataset(args):
    """加载 CIFAR-10 测试集，使用 DSDM 标准归一化。"""
    from data import MEANS, STDS

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=MEANS["cifar10"], std=STDS["cifar10"]),
        ]
    )
    return datasets.CIFAR10(args.data_dir, train=False, transform=transform)


def get_agent_train_dataset(args, agent_id, normalize=False):
    """加载单个 agent 的 expert class 训练集。"""
    dataset = get_cifar10_train_dataset(args, normalize=normalize)
    return ActiveClassDataset(dataset, AGENT_CLASS_SPLIT[int(agent_id)])


def save_tensor_state(path, model):
    """保存模型 state_dict 到指定路径。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path
