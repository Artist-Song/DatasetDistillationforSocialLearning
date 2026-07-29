from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

from config_adapter import apply_classifier_overrides, apply_pcbn_overrides, build_dsdm_args_from_config
from tiny_imagenet_data import TinyImageNetDataset, load_tiny_imagenet_leaked_validation_paths


CIFAR10_AGENT_CLASS_SPLIT = {
    0: [0, 1],
    1: [2, 3],
    2: [4, 5],
    3: [6, 7],
    4: [8, 9],
}

CIFAR10_AGENT_MODEL_SPLIT = {
    0: "convnet",
    1: "convnet",
    2: "resnet",
    3: "resnet",
    4: "resnet_ap",
}

CIFAR100_AGENT_CLASS_SPLIT = {
    0: list(range(0, 25)),
    1: list(range(25, 50)),
    2: list(range(50, 75)),
    3: list(range(75, 100)),
}

CIFAR100_AGENT_MODEL_SPLIT = {
    0: "convnet",
    1: "convnet",
    2: "resnet",
    3: "resnet_ap",
}

AGENT_CLASS_SPLIT = CIFAR10_AGENT_CLASS_SPLIT
AGENT_MODEL_SPLIT = CIFAR10_AGENT_MODEL_SPLIT


def _get_dataset_name(args_or_cfg):
    """从 args 或配置字典中读取数据集名称。"""
    if isinstance(args_or_cfg, dict):
        return args_or_cfg.get("dataset", {}).get("name", "cifar10")
    return getattr(args_or_cfg, "dataset", "cifar10")


def _normalize_split(split):
    """把配置中的 agent_x 划分转换为 int key 字典。"""
    normalized = {}
    for key, value in split.items():
        agent_id = int(str(key).replace("agent_", ""))
        normalized[agent_id] = [int(c) for c in value]
    return normalized


def _normalize_model_split(split):
    """把配置中的 agent_x 模型划分转换为 int key 字典。"""
    normalized = {}
    for key, value in split.items():
        agent_id = int(str(key).replace("agent_", ""))
        normalized[agent_id] = str(value)
    return normalized


def get_num_classes(args_or_cfg):
    """根据 dataset 返回类别数，cifar10=10，cifar100=100。"""
    if isinstance(args_or_cfg, dict):
        dataset_cfg = args_or_cfg.get("dataset", {})
        if "num_classes" in dataset_cfg:
            return int(dataset_cfg["num_classes"])
    if hasattr(args_or_cfg, "num_classes"):
        return int(args_or_cfg.num_classes)
    if hasattr(args_or_cfg, "nclass"):
        return int(args_or_cfg.nclass)
    dataset_name = _get_dataset_name(args_or_cfg)
    if dataset_name == "cifar100":
        return 100
    if dataset_name == "cifar10":
        return 10
    raise ValueError(f"暂不支持的数据集类别数: {dataset_name}")


def get_agent_class_split(args_or_cfg):
    """根据 dataset 返回 agent 类划分。"""
    if isinstance(args_or_cfg, dict):
        cfg_split = args_or_cfg.get("agents", {}).get("class_split")
        if cfg_split:
            return _normalize_split(cfg_split)
    if hasattr(args_or_cfg, "agent_class_split"):
        return {int(k): [int(c) for c in v] for k, v in args_or_cfg.agent_class_split.items()}
    dataset_name = _get_dataset_name(args_or_cfg)
    if dataset_name == "cifar100":
        return CIFAR100_AGENT_CLASS_SPLIT
    if dataset_name == "cifar10":
        return CIFAR10_AGENT_CLASS_SPLIT
    raise ValueError(f"暂不支持的数据集 agent 划分: {dataset_name}")


def get_agent_model_split(args_or_cfg):
    """根据 dataset 返回 agent 模型划分。"""
    if isinstance(args_or_cfg, dict):
        cfg_split = args_or_cfg.get("agents", {}).get("model_split")
        if cfg_split:
            return _normalize_model_split(cfg_split)
    if hasattr(args_or_cfg, "agent_model_split"):
        return {int(k): str(v) for k, v in args_or_cfg.agent_model_split.items()}
    dataset_name = _get_dataset_name(args_or_cfg)
    if dataset_name == "cifar100":
        return CIFAR100_AGENT_MODEL_SPLIT
    if dataset_name == "cifar10":
        return CIFAR10_AGENT_MODEL_SPLIT
    raise ValueError(f"暂不支持的数据集模型划分: {dataset_name}")


class ActiveClassDataset(Dataset):
    """只暴露指定类别样本，同时保留数据集全局标签。"""

    def __init__(self, dataset, active_class_ids, num_classes=None):
        """根据 active_class_ids 过滤数据集索引。"""
        self.dataset = dataset
        self.active_class_ids = [int(c) for c in active_class_ids]
        targets = getattr(dataset, "targets", getattr(dataset, "labels", None))
        if targets is None:
            raise ValueError("数据集缺少 targets/labels 字段")
        self.indices = [i for i, y in enumerate(targets) if int(y) in self.active_class_ids]
        self.targets = [int(targets[i]) for i in self.indices]
        self.nclass = int(num_classes or getattr(dataset, "nclass", max(targets) + 1))

    def __len__(self):
        """返回过滤后的样本数。"""
        return len(self.indices)

    def __getitem__(self, index):
        """返回过滤后的样本和全局标签。"""
        return self.dataset[self.indices[index]]


def get_agent_ids(args_or_cfg=None, only_agent=None):
    """返回本次需要处理的 agent id 列表。"""
    if only_agent is None:
        return list(get_agent_class_split(args_or_cfg).keys())
    return [int(only_agent)]


def get_receiver_ids(args_or_cfg=None, only_receiver=None):
    """返回本次需要处理的 receiver id 列表。"""
    if only_receiver is None:
        return list(get_agent_class_split(args_or_cfg).keys())
    return [int(only_receiver)]




def _refresh_model_metadata(args):
    """根据实际 net_type/depth/width 刷新 DSDM 模型标签和特征层。"""
    if args.net_type == "convnet":
        args.f_idx = str(args.depth - 1)
    elif args.net_type in {"resnet", "resnet_cifar_standard", "resnet_ap"}:
        # ResNet logits前一层 = avgpool+flatten输出，idx=5（idx4是layer4空间图，idx6是logits）
        args.f_idx = "5"
    elif args.net_type == "alexnet":
        # AlexNetCIFAR last_feature = idx7，对应logits前一层[B,512]
        args.f_idx = "7"
    elif args.net_type == "vgg":
        # VGG11-CIFAR last_feature = idx10，对应logits前一层[B,512]
        args.f_idx = "10"
    args.datatag = f"{args.dataset}"
    if args.net_type == "resnet_ap":
        args.modeltag = f"resnet{args.depth}ap"
    elif args.net_type == "convnet":
        args.modeltag = f"conv{args.depth}"
    elif args.net_type in {"alexnet", "vgg"}:
        # alexnet/vgg 不依赖depth，modeltag只用架构名
        args.modeltag = args.net_type
    elif args.net_type == "resnet_cifar_standard":
        args.modeltag = f"resnet{args.depth}_cifar_w1"
    else:
        args.modeltag = f"{args.net_type}{args.depth}"
    if args.norm_type == "instance":
        args.modeltag += "in"
    if args.width != 1.0:
        args.modeltag += f"_w{args.width}"


def _refresh_feature_indices(args):
    """同步 DSDM 特征层索引，避免 per-agent f_idx 更新后 idx_from 仍沿用旧值。"""
    if getattr(args, "ipc", 0) > 0 and (
        getattr(args, "match", "") == "semantic"
        or bool(getattr(args, "official_dsdm_protocol", False))
    ):
        f_list = [int(s) for s in str(args.f_idx).split(",")]
        if len(f_list) == 1:
            f_list.append(-1)
        args.idx_from, args.idx_to = f_list
        args.metric = "mse"
    else:
        args.idx_from, args.idx_to = 0, -1


def build_agent_args(base_cfg, config_path, agent_id):
    """基于主配置构造单个 agent 的 DSDM args。"""
    args = build_dsdm_args_from_config(base_cfg, config_path=config_path)
    class_split = get_agent_class_split(base_cfg)
    model_split = get_agent_model_split(base_cfg)
    model_cfg = base_cfg.get("model_pool", {}).get("models", {}).get(model_split[int(agent_id)], {})
    num_classes = get_num_classes(base_cfg)
    args.agent_id = int(agent_id)
    args.num_classes = num_classes
    args.nclass = num_classes
    args.active_class_ids = list(class_split[int(agent_id)])
    args.model_name = model_split[int(agent_id)]
    args.net_type = str(model_cfg.get("family", args.model_name))
    args.depth = int(model_cfg.get("depth", args.depth))
    args.width = float(model_cfg.get("width", args.width))
    args.norm_type = str(model_cfg.get("norm_type", args.norm_type))
    args.sender_model = args.model_name
    _refresh_model_metadata(args)
    _refresh_feature_indices(args)
    guide_cfg = model_cfg.get("guide_training", {})
    expert_cfg = model_cfg.get("expert_training", {})
    args.separate_expert = bool(expert_cfg.get("separate", False))
    args.classifier_type = "linear"
    args.cosine_scale_init = 10.0
    apply_classifier_overrides(args, model_cfg.get("classifier"))
    apply_classifier_overrides(args, expert_cfg.get("classifier"))
    if "classifier_type" in expert_cfg:
        args.classifier_type = str(expert_cfg["classifier_type"])
    if "cosine_scale_init" in expert_cfg:
        args.cosine_scale_init = float(expert_cfg["cosine_scale_init"])
    args.expert_classifier_type = str(args.classifier_type).lower()
    args.expert_cosine_scale_init = float(args.cosine_scale_init)
    args.expert_mask_nonlocal_classes = bool(
        expert_cfg.get(
            "masked_local_ce",
            expert_cfg.get("mask_nonlocal_classes", args.expert_classifier_type == "cosine"),
        )
    )
    guide_classifier = guide_cfg.get("classifier", "linear")
    args.guide_classifier_type = "linear"
    args.guide_cosine_scale_init = 10.0
    if isinstance(guide_classifier, str):
        args.guide_classifier_type = str(guide_classifier).lower()
    elif isinstance(guide_classifier, dict):
        args.guide_classifier_type = str(guide_classifier.get("type", "linear")).lower()
        args.guide_cosine_scale_init = float(
            guide_classifier.get("scale_init", guide_classifier.get("initial_scale", 10.0))
        )
    else:
        raise TypeError(f"guide classifier config must be a mapping or string: {guide_classifier!r}")

    if guide_cfg:
        args.pretrained_model_number = int(guide_cfg.get("num_models", args.pretrained_model_number))
        args.pretrained_epochs = int(guide_cfg.get("max_epochs", guide_cfg.get("epochs", args.pretrained_epochs)))
        args.guide_model_number = args.pretrained_model_number
        args.guide_max_epochs = args.pretrained_epochs
        args.guide_snapshot_epochs = sorted(
            {int(v) for v in guide_cfg.get("snapshot_epochs", [args.guide_max_epochs])}
        )
        args.guide_epoch = int(guide_cfg.get("selected_epoch", args.guide_max_epochs))
        if args.guide_epoch not in args.guide_snapshot_epochs:
            args.guide_snapshot_epochs.append(args.guide_epoch)
            args.guide_snapshot_epochs.sort()
        args.guide_lr = float(guide_cfg.get("lr", args.lr))
        args.guide_batch_size = int(guide_cfg.get("batch_size", args.batch_size))
        args.guide_augment = bool(guide_cfg.get("augment", False))
        args.guide_scheduler = str(guide_cfg.get("scheduler", "none"))
        args.guide_scheduler_milestones = [int(v) for v in guide_cfg.get("scheduler_milestones", [])]
        args.guide_scheduler_gamma = float(guide_cfg.get("scheduler_gamma", 0.1))
        args.guide_training_style = str(guide_cfg.get("training_style", "plain"))
        args.guide_pool_design = str(guide_cfg.get("pool_design", "independent_final_models"))
        args.guide_trajectory_count = int(guide_cfg.get("trajectory_count", args.guide_model_number))
        args.guide_trajectory_checkpoint_epochs = [
            int(v) for v in guide_cfg.get("trajectory_checkpoint_epochs", [])
        ]
        args.guide_source_root = guide_cfg.get("source_root")
    else:
        # Legacy runs trained one pool and then selected an expert from that pool.
        if expert_cfg and not args.separate_expert:
            args.pretrained_model_number = int(expert_cfg.get("num_models", args.pretrained_model_number))
            args.pretrained_epochs = int(expert_cfg.get("epochs", args.pretrained_epochs))
        args.guide_model_number = int(args.pretrained_model_number)
        args.guide_max_epochs = int(args.pretrained_epochs)
        args.guide_snapshot_epochs = [int(args.pretrained_epochs)]
        args.guide_epoch = int(args.pretrained_epochs)
        args.guide_lr = float(expert_cfg.get("lr", args.lr))
        args.guide_batch_size = int(expert_cfg.get("batch_size", args.batch_size))
        args.guide_augment = bool(expert_cfg.get("augment", False))
        args.guide_scheduler = str(expert_cfg.get("scheduler", "none"))
        args.guide_scheduler_milestones = [int(v) for v in expert_cfg.get("scheduler_milestones", [])]
        args.guide_scheduler_gamma = float(expert_cfg.get("scheduler_gamma", 0.1))
        args.guide_training_style = "plain"
        args.guide_pool_design = "independent_final_models"
        args.guide_trajectory_count = int(args.guide_model_number)
        args.guide_trajectory_checkpoint_epochs = []
        args.guide_source_root = None

    if expert_cfg:
        args.expert_epochs = int(expert_cfg.get("epochs", args.pretrained_epochs))
        args.expert_lr = float(expert_cfg.get("lr", args.lr))
        args.expert_batch_size = int(expert_cfg.get("batch_size", args.batch_size))
        args.expert_augment = bool(expert_cfg.get("augment", False))
        args.expert_scheduler = str(expert_cfg.get("scheduler", "none"))
        args.expert_scheduler_milestones = [int(v) for v in expert_cfg.get("scheduler_milestones", [])]
        args.expert_scheduler_gamma = float(expert_cfg.get("scheduler_gamma", 0.1))
        args.expert_validation_fraction = float(expert_cfg.get("validation_fraction", 0.1))
        args.expert_eval_interval = int(expert_cfg.get("eval_interval", 5))
        args.expert_retrain_full = bool(expert_cfg.get("retrain_full", True))
        args.expert_use_dsdm_train = bool(expert_cfg.get("use_dsdm_train", False))
        args.expert_source_root = expert_cfg.get("source_root")
    else:
        args.expert_epochs = int(args.pretrained_epochs)
        args.expert_lr = float(args.lr)
        args.expert_batch_size = int(args.batch_size)
        args.expert_augment = False
        args.expert_scheduler = "none"
        args.expert_scheduler_milestones = []
        args.expert_scheduler_gamma = 0.1
        args.expert_validation_fraction = 0.1
        args.expert_eval_interval = 5
        args.expert_retrain_full = True
        args.expert_use_dsdm_train = False
        args.expert_source_root = None
    # 应用 per-model 蒸馏参数覆盖，并在 f_idx 覆盖后同步实际 DSDM 特征层。
    distill_override = model_cfg.get("distillation", {})
    if distill_override.get("f_idx") is not None:
        args.f_idx = str(distill_override["f_idx"])
        _refresh_feature_indices(args)
    if distill_override.get("lr_img") is not None:
        args.lr_img = float(distill_override["lr_img"])
    if distill_override.get("grad_clip_norm") is not None:
        args.grad_clip_norm = float(distill_override["grad_clip_norm"])
    if distill_override.get("niter") is not None:
        args.niter = int(distill_override["niter"])
    if "evaluate_iter" in distill_override:
        args.evaluate_iter = int(distill_override["evaluate_iter"])
    if "evaluate_iterations" in distill_override:
        args.evaluate_iterations = distill_override["evaluate_iterations"]
    if isinstance(distill_override.get("pcbn"), dict):
        apply_pcbn_overrides(args, distill_override["pcbn"])
    if distill_override.get("pretrain_dir") is not None:
        args.pretrain_dir = str(distill_override["pretrain_dir"])
    args.save_pretrain_dir = str(get_agent_dir(args, agent_id) / "checkpoints")
    args.save_dir = str(get_agent_dir(args, agent_id) / "checkpoints")
    args.guide_only = bool(base_cfg.get("fullclass_pool", {}).get("guide_only", False))
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


def get_train_dataset(args, normalize=False, augment=False):
    """根据 args.dataset 加载训练集，标签保持全局编号。"""
    transform_list = []
    if augment and args.dataset in {"cifar10", "cifar100"}:
        transform_list.extend([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])
    elif augment and args.dataset == "tinyimagenet":
        transform_list.extend(
            [transforms.RandomCrop(64, padding=4, padding_mode="reflect"), transforms.RandomHorizontalFlip()]
        )
    transform_list.append(transforms.ToTensor())
    if normalize:
        from data import MEANS, STDS

        transform_list.append(transforms.Normalize(mean=MEANS[args.dataset], std=STDS[args.dataset]))
    transform = transforms.Compose(transform_list)
    if args.dataset == "cifar10":
        return datasets.CIFAR10(args.data_dir, train=True, transform=transform, download=False)
    if args.dataset == "cifar100":
        return datasets.CIFAR100(args.data_dir, train=True, transform=transform, download=False)
    if args.dataset == "tinyimagenet":
        return TinyImageNetDataset(args.data_dir, "train", transform=transform)
    raise ValueError(f"暂不支持的数据集: {args.dataset}")


def get_test_dataset(args):
    """根据 args.dataset 加载测试集，使用 DSDM 标准归一化。"""
    from data import MEANS, STDS

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=MEANS[args.dataset], std=STDS[args.dataset]),
        ]
    )
    if args.dataset == "cifar10":
        return datasets.CIFAR10(args.data_dir, train=False, transform=transform, download=False)
    if args.dataset == "cifar100":
        return datasets.CIFAR100(args.data_dir, train=False, transform=transform, download=False)
    if args.dataset == "tinyimagenet":
        excluded_paths = set()
        integrity_report = getattr(args, "tiny_integrity_report", None)
        if integrity_report:
            excluded_paths = load_tiny_imagenet_leaked_validation_paths(integrity_report)
        return TinyImageNetDataset(
            args.data_dir,
            "val",
            transform=transform,
            excluded_paths=excluded_paths,
        )
    raise ValueError(f"暂不支持的数据集: {args.dataset}")


def get_cifar10_train_dataset(args, normalize=False):
    """兼容旧接口，加载当前配置的数据集训练集。"""
    return get_train_dataset(args, normalize=normalize)


def get_cifar10_test_dataset(args):
    """兼容旧接口，加载当前配置的数据集测试集。"""
    return get_test_dataset(args)


def get_agent_train_dataset(args, agent_id, normalize=False, augment=False):
    """加载单个 agent 的 expert class 训练集。"""
    dataset = get_train_dataset(args, normalize=normalize, augment=augment)
    return ActiveClassDataset(dataset, get_agent_class_split(args)[int(agent_id)], num_classes=get_num_classes(args))


def save_tensor_state(path, model):
    """保存模型 state_dict 到指定路径。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path
