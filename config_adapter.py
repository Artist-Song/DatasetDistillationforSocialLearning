import argparse
import copy
import json
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


DSDM_DEFAULT_ARGS = {
    "dataset": "cifar10",
    "data_dir": "./data",
    "imagenet_dir": "/ssd_data/imagenet/",
    "nclass": 10,
    "dseed": 0,
    "size": 32,
    "phase": -1,
    "nclass_sub": -1,
    "load_memory": True,
    "net_type": "convnet",
    "norm_type": "instance",
    "depth": 3,
    "width": 1.0,
    "pretrained_model_number": 10,
    "pretrained_epochs": 20,
    "batch_size": 64,
    "lr": 0.01,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "seed": 0,
    "pretrained": False,
    "save_pretrain_dir": "./pre_trained_model",
    "mixup": "cut",
    "mixup_net": "cut",
    "beta": 1.0,
    "mix_p": 0.5,
    "print_freq": 10,
    "verbose": False,
    "workers": 8,
    "save_ckpt": False,
    "tag": "",
    "test": False,
    "time": False,
    "cov_weight": 50.0,
    "h_p_weight": 0.2,
    "smooth_factor": 0.99,
    "epochs": 1500,
    "ipc": 10,
    "factor": 2,
    "decode_type": "single",
    "init": "mix",
    "aug_type": "color_crop_cutout",
    "match": "semantic",
    "metric": "mse",
    "bias": False,
    "fc": False,
    "f_idx": "2",
    "niter": 10000,
    "smooth_iter": 2000,
    "evaluate_iter": 100,
    "batch_real": 256,
    "batch_syn_max": 256,
    "lr_img": 0.1,
    "mom_img": 0.5,
    "reproduce": False,
    "slct_type": "DSDM",
    "repeat": 1,
    "dsa": True,
    "dsa_strategy": "color_crop_cutout_flip_scale_rotate",
    "rrc": True,
    "same_compute": False,
    "name": "",
    "bottleneck": True,
    "nch": 3,
    "datatag": "cifar10",
    "modeltag": "conv3in",
    "epoch_print_freq": 1500,
    "augment": False,
    "gpu_id": 0,
    "device": "cuda",
    "output_root": "./outputs",
    "run_name": "cifar10_dsdm_ipc10",
    "config_path": None,
    "save_dir": "./results/cifar10/conv3in_semantic_mse_cut_factor2_lr0.1_mix_ipc10",
}


def load_config(path):
    """读取 YAML 主配置文件，返回普通字典。"""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if yaml is not None:
        cfg = yaml.safe_load(text) or {}
    else:
        cfg = _load_simple_yaml(text)
    return cfg


def _parse_scalar(value):
    """解析轻量 YAML 标量值。"""
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load_simple_yaml(text):
    """解析当前主配置使用的简单 YAML 子集。"""
    raw_lines = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        raw_lines.append((indent, raw.strip()))

    def parse_block(index, indent):
        is_list = index < len(raw_lines) and raw_lines[index][1].startswith("- ")
        container = [] if is_list else {}
        while index < len(raw_lines):
            cur_indent, content = raw_lines[index]
            if cur_indent < indent:
                break
            if cur_indent > indent:
                break
            if is_list:
                container.append(_parse_scalar(content[2:]))
                index += 1
                continue
            key, _, value = content.partition(":")
            key = key.strip()
            value = value.strip()
            if value:
                container[key] = _parse_scalar(value)
                index += 1
            else:
                child_indent = raw_lines[index + 1][0] if index + 1 < len(raw_lines) else indent + 2
                child, index = parse_block(index + 1, child_indent)
                container[key] = child
        return container, index

    cfg, _ = parse_block(0, 0)
    return cfg


def _remove_aug(augtype, remove_aug):
    """从增强字符串中移除指定增强项。"""
    aug_list = []
    for aug in augtype.split("_"):
        if aug not in remove_aug.split("_"):
            aug_list.append(aug)
    return "_".join(aug_list)


def _apply_dataset_rules(args):
    """补齐 DSDM 原参数脚本中的数据集派生字段。"""
    args.nch = 3
    if args.dataset[:5] == "cifar":
        args.size = 32
        args.mix_p = 0.5
        args.dsa = True
        args.nclass = 10 if args.dataset == "cifar10" else 100
    elif args.dataset == "svhn":
        args.size = 32
        args.nclass = 10
        args.mix_p = 0.5
        args.dsa = True
        args.dsa_strategy = _remove_aug(args.dsa_strategy, "flip")
    elif args.dataset[:5] == "mnist":
        args.nclass = 10
        args.size = 28
        args.nch = 1
        args.mix_p = 0.5
        args.dsa = True
        args.dsa_strategy = _remove_aug(args.dsa_strategy, "flip")
    elif args.dataset == "fashion":
        args.nclass = 10
        args.size = 28
        args.nch = 1
        args.mix_p = 0.5
        args.dsa = True
    elif args.dataset == "speech":
        args.nch = 1
        args.size = 64
        args.nclass = 8
        args.mixup = "vanilla"
        args.mixup_net = "vanilla"
        args.dsa = False


def _apply_model_rules(args):
    """补齐 DSDM 原参数脚本中的模型标签和特征层字段。"""
    if args.net_type == "convnet" and args.depth > 4:
        args.depth = 3
    if args.net_type == "convnet":
        args.f_idx = str(args.depth - 1)

    args.datatag = f"{args.dataset}"
    if args.net_type == "resnet_ap":
        args.modeltag = f"resnet{args.depth}ap"
    elif args.net_type == "convnet":
        args.modeltag = f"conv{args.depth}"
    else:
        args.modeltag = f"{args.net_type}{args.depth}"
    if args.norm_type == "instance":
        args.modeltag += "in"
    if args.width != 1.0:
        args.modeltag += f"_w{args.width}"


def _apply_runtime_rules(args):
    """补齐 DSDM 原参数脚本中的运行和增强派生字段。"""
    if args.factor > 1:
        args.init = "mix"
    if args.ipc > 0 and args.match == "semantic":
        f_list = [int(s) for s in args.f_idx.split(",")]
        if len(f_list) == 1:
            f_list.append(-1)
        args.idx_from, args.idx_to = f_list
        args.metric = "mse"
    else:
        args.idx_from, args.idx_to = 0, -1

    args.epoch_print_freq = args.epochs if args.ipc > 0 else 1
    if args.mixup == "cut":
        args.dsa_strategy = _remove_aug(args.dsa_strategy, "cutout")
    args.augment = not args.dsa


def _apply_config_overrides(args, cfg):
    """把主配置中的字段覆盖到 DSDM Namespace。"""
    project = cfg.get("project", {})
    dataset = cfg.get("dataset", {})
    model_pool = cfg.get("model_pool", {})
    model_name = model_pool.get("default", args.net_type)
    model_cfg = model_pool.get("models", {}).get(model_name, {})
    distill = cfg.get("distillation", {})
    evaluation = cfg.get("evaluation", {})
    runtime = cfg.get("runtime", {})

    args.output_root = project.get("output_root", args.output_root)
    args.run_name = project.get("run_name", args.run_name)
    args.dataset = dataset.get("name", args.dataset)
    args.data_dir = dataset.get("data_dir", args.data_dir)
    args.nclass = dataset.get("num_classes", args.nclass)
    args.size = dataset.get("image_size", args.size)

    args.net_type = model_name
    args.depth = model_cfg.get("depth", args.depth)
    args.width = model_cfg.get("width", args.width)
    args.norm_type = model_cfg.get("norm_type", args.norm_type)

    for key in [
        "ipc",
        "factor",
        "init",
        "decode_type",
        "aug_type",
        "match",
        "metric",
        "f_idx",
        "niter",
        "lr_img",
        "mom_img",
        "batch_real",
        "batch_syn_max",
        "smooth_iter",
        "cov_weight",
        "h_p_weight",
        "smooth_factor",
    ]:
        if key in distill:
            setattr(args, key, distill[key])

    args.epochs = evaluation.get("epochs", args.epochs)
    args.batch_size = evaluation.get("batch_size", args.batch_size)
    args.repeat = evaluation.get("repeat", args.repeat)
    args.seed = runtime.get("seed", args.seed)
    args.workers = runtime.get("workers", args.workers)
    args.device = runtime.get("device", args.device)
    args.gpu_id = runtime.get("gpu_id", args.gpu_id)


def build_dsdm_args_from_config(cfg, config_path=None):
    """将主配置转换为 DSDM 可直接使用的 argparse.Namespace。"""
    values = copy.deepcopy(DSDM_DEFAULT_ARGS)
    args = argparse.Namespace(**values)
    _apply_config_overrides(args, cfg)
    args.config_path = str(config_path) if config_path is not None else None
    _apply_dataset_rules(args)
    _apply_model_rules(args)
    _apply_runtime_rules(args)
    args.save_dir = str(Path(args.output_root) / args.run_name / "checkpoints")
    return args


def args_to_pretty_json(args):
    """把解析后的 args 转为便于 dry-run 打印的 JSON 字符串。"""
    return json.dumps(vars(args), indent=2, ensure_ascii=False, sort_keys=True)
