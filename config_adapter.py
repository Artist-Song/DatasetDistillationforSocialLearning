import argparse
import copy
import json
import math
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


DSDM_DEFAULT_ARGS = {
    "dataset": "cifar10",
    "data_dir": "./data",
    "imagenet_dir": "/ssd_data/imagenet/",
    "tiny_integrity_report": None,
    "nclass": 10,
    "num_classes": 10,
    "active_class_ids": None,
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
    "classifier_type": "linear",
    "cosine_scale_init": 10.0,
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
    "pcbn_enabled": False,
    "pcbn_weight": 0.0,
    "pcbn_layers": "all",
    "pcbn_normalize_layers": True,
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
    "evaluate_iterations": [100, 500, 1000, 2000, 3000, 5000, 7500, 10000],
    "batch_real": 256,
    "batch_syn_max": 256,
    "lr_img": 0.1,
    "mom_img": 0.5,
    "grad_clip_norm": 0.0,
    "guide_model_mode": "eval",
    "freeze_guide_parameters": True,
    "official_dsdm_protocol": False,
    "official_dsdm_commit": None,
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
    "fast_repo_path": "external_baselines/repos/FAST",
    "fast_cache_root": "external_baselines/outputs/fast_cache",
    "fast_commit": "6a218fcfdc93838634921399b0de6a36cdd29756",
    "fast_seed": 0,
    "receiver_local_ce_source": "real",
    "receiver_local_ce_real_fraction": None,
    "receiver_optimizer_steps": None,
    "receiver_self_packet_batch_size": 64,
    "receiver_scheduler_unit": "epoch",
    "receiver_scheduler_step_milestones": [],
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


def _as_bool(value):
    """把配置里的布尔值安全转成 bool。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def apply_classifier_overrides(args, classifier_cfg):
    """Apply an opt-in classifier block while keeping linear as the default."""
    if classifier_cfg is None:
        return
    if isinstance(classifier_cfg, str):
        args.classifier_type = classifier_cfg
        return
    if not isinstance(classifier_cfg, dict):
        raise TypeError(f"classifier config must be a mapping or string: {classifier_cfg!r}")
    if "type" in classifier_cfg:
        args.classifier_type = str(classifier_cfg["type"])
    if "scale_init" in classifier_cfg:
        args.cosine_scale_init = float(classifier_cfg["scale_init"])
    elif "initial_scale" in classifier_cfg:
        args.cosine_scale_init = float(classifier_cfg["initial_scale"])
    if str(args.classifier_type).lower() == "cosine":
        positive = classifier_cfg.get("positive", classifier_cfg.get("positive_parameterization"))
        if positive is not None and str(positive).lower() != "softplus":
            raise ValueError(f"cosine scale positive parameterization must be softplus: {positive}")
        if "bias" in classifier_cfg and _as_bool(classifier_cfg["bias"]):
            raise ValueError("cosine classifier must be bias-free")
        if "scale_weight_decay" in classifier_cfg and float(classifier_cfg["scale_weight_decay"]) != 0.0:
            raise ValueError("cosine classifier scale_weight_decay must be 0")


def normalize_receiver_checkpoint_retention(value):
    """Validate the opt-in receiver checkpoint retention policy."""
    resolved = str(value).strip().lower()
    if resolved not in {"all", "final_only"}:
        raise ValueError(
            "receiver checkpoint_retention must be 'all' or 'final_only': "
            f"{value!r}"
        )
    return resolved


def normalize_receiver_local_ce_source(value):
    """Validate the opt-in source used for receiver-local classification CE."""
    resolved = str(value).strip().lower()
    allowed = {"real", "packet", "real_packet_50_50", "real_packet_mix"}
    if resolved not in allowed:
        raise ValueError(
            "receiver.local_ce_source must be one of "
            f"{sorted(allowed)}: {value!r}"
        )
    return resolved


def normalize_receiver_scheduler_unit(value):
    """Validate whether the receiver scheduler advances per epoch or optimizer step."""
    resolved = str(value).strip().lower()
    if resolved not in {"epoch", "optimizer_step"}:
        raise ValueError(
            "receiver.scheduler_unit must be 'epoch' or 'optimizer_step': "
            f"{value!r}"
        )
    return resolved


def _positive_config_int(value, field_name):
    """Require a positive YAML integer without accepting bools or lossy casts."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer: {value!r}")
    return int(value)


def apply_receiver_training_overrides(args, receiver):
    """Parse the opt-in fixed-step/local-CE receiver controls."""
    if not isinstance(receiver, dict):
        raise TypeError(f"social_learning.receiver must be a mapping: {receiver!r}")

    if "local_ce_source" in receiver:
        args.receiver_local_ce_source = normalize_receiver_local_ce_source(
            receiver["local_ce_source"]
        )
    has_real_fraction = "local_ce_real_fraction" in receiver
    local_ce_source = str(getattr(args, "receiver_local_ce_source", "real"))
    if local_ce_source == "real_packet_mix":
        if not has_real_fraction:
            raise ValueError(
                "receiver.local_ce_source='real_packet_mix' requires "
                "receiver.local_ce_real_fraction"
            )
        value = receiver["local_ce_real_fraction"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("receiver.local_ce_real_fraction must be numeric")
        value = float(value)
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError(
                "receiver.local_ce_real_fraction must be finite and strictly between 0 and 1"
            )
        args.receiver_local_ce_real_fraction = value
    elif has_real_fraction:
        raise ValueError(
            "receiver.local_ce_real_fraction is only valid when "
            "receiver.local_ce_source='real_packet_mix'"
        )
    if "optimizer_steps" in receiver:
        args.receiver_optimizer_steps = _positive_config_int(
            receiver["optimizer_steps"],
            "receiver.optimizer_steps",
        )
    if "self_packet_batch_size" in receiver:
        args.receiver_self_packet_batch_size = _positive_config_int(
            receiver["self_packet_batch_size"],
            "receiver.self_packet_batch_size",
        )
    if "packet_raw_per_class" in receiver:
        args.receiver_packet_raw_per_class = _positive_config_int(
            receiver["packet_raw_per_class"],
            "receiver.packet_raw_per_class",
        )
    if "scheduler_unit" in receiver:
        args.receiver_scheduler_unit = normalize_receiver_scheduler_unit(
            receiver["scheduler_unit"]
        )

    has_step_milestones = "scheduler_step_milestones" in receiver
    scheduler_unit = str(getattr(args, "receiver_scheduler_unit", "epoch"))
    if scheduler_unit == "optimizer_step":
        optimizer_steps = getattr(args, "receiver_optimizer_steps", None)
        if optimizer_steps is None:
            raise ValueError(
                "receiver.scheduler_unit='optimizer_step' requires "
                "receiver.optimizer_steps"
            )
        if not has_step_milestones:
            raise ValueError(
                "receiver.scheduler_unit='optimizer_step' requires explicit "
                "receiver.scheduler_step_milestones"
            )
        raw_milestones = receiver["scheduler_step_milestones"]
        if not isinstance(raw_milestones, (list, tuple)) or not raw_milestones:
            raise ValueError(
                "receiver.scheduler_step_milestones must be a non-empty sequence"
            )
        milestones = [
            _positive_config_int(value, "receiver.scheduler_step_milestones entry")
            for value in raw_milestones
        ]
        if any(current <= previous for previous, current in zip(milestones, milestones[1:])):
            raise ValueError(
                "receiver.scheduler_step_milestones must be strictly increasing"
            )
        if any(value >= optimizer_steps for value in milestones):
            raise ValueError(
                "receiver.scheduler_step_milestones entries must be smaller than "
                "receiver.optimizer_steps"
            )
        args.receiver_scheduler_step_milestones = milestones
    elif has_step_milestones:
        raise ValueError(
            "receiver.scheduler_step_milestones is only valid when "
            "receiver.scheduler_unit='optimizer_step'"
        )


def apply_pcbn_overrides(args, pcbn_cfg):
    """将 PCBN 可选正则配置写入 DSDM args。"""
    if not isinstance(pcbn_cfg, dict):
        return
    if "enabled" in pcbn_cfg:
        args.pcbn_enabled = _as_bool(pcbn_cfg["enabled"])
    if "weight" in pcbn_cfg:
        args.pcbn_weight = float(pcbn_cfg["weight"])
    if "layers" in pcbn_cfg:
        args.pcbn_layers = pcbn_cfg["layers"]
    if "normalize_layers" in pcbn_cfg:
        args.pcbn_normalize_layers = _as_bool(pcbn_cfg["normalize_layers"])
    if "enabled" not in pcbn_cfg and float(getattr(args, "pcbn_weight", 0.0)) > 0:
        args.pcbn_enabled = True


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
    elif args.dataset == "tinyimagenet":
        args.size = 64
        args.nclass = int(args.num_classes)
        args.mix_p = 0.5
        args.dsa = True
    elif args.dataset == "speech":
        args.nch = 1
        args.size = 64
        args.nclass = 8
        args.mixup = "vanilla"
        args.mixup_net = "vanilla"
        args.dsa = False
    args.num_classes = args.nclass


def _apply_model_rules(args):
    """补齐 DSDM 原参数脚本中的模型标签和特征层字段。"""
    if args.net_type == "convnet" and args.depth > 4:
        args.depth = 3
    if args.net_type == "convnet":
        args.f_idx = str(args.depth - 1)
    elif args.net_type == "alexnet":
        # AlexNetCIFAR last_feature = idx7，对应logits前一层[B,512]
        args.f_idx = "7"
    elif args.net_type == "vgg":
        # VGG11-CIFAR last_feature = idx10，对应logits前一层[B,512]
        args.f_idx = "10"
    elif args.net_type in {"resnet", "resnet_cifar_standard", "resnet_ap"}:
        # ResNet特征索引：0=layer0,1=layer1,2=layer2,3=layer3,4=layer4(空间),
        # 5=avgpool+flatten(penultimate向量,logits前一层),6=logits
        # 用户要求取logits前一层，即idx=5
        args.f_idx = "5"

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


def _apply_runtime_rules(args):
    """补齐 DSDM 原参数脚本中的运行和增强派生字段。"""
    if args.factor > 1:
        args.init = "mix"
    if args.ipc > 0 and (
        args.match == "semantic" or bool(getattr(args, "official_dsdm_protocol", False))
    ):
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
    communication = cfg.get("communication", {})
    receiver = cfg.get("social_learning", {}).get("receiver", {})
    fast_cfg = cfg.get("selection", {}).get("fast", {})
    pcbn_cfg = {}
    if isinstance(cfg.get("pcbn"), dict):
        pcbn_cfg.update(cfg.get("pcbn"))
    if isinstance(distill.get("pcbn"), dict):
        pcbn_cfg.update(distill.get("pcbn"))

    args.output_root = project.get("output_root", args.output_root)
    args.run_name = project.get("run_name", args.run_name)
    args.dataset = dataset.get("name", args.dataset)
    args.data_dir = dataset.get("data_dir", args.data_dir)
    args.tiny_integrity_report = dataset.get("integrity_report", args.tiny_integrity_report)
    args.nclass = dataset.get("num_classes", args.nclass)
    args.num_classes = dataset.get("num_classes", args.nclass)
    args.size = dataset.get("image_size", args.size)
    agents_cfg = cfg.get("agents", {})
    if agents_cfg.get("class_split"):
        args.agent_class_split = {
            int(str(key).replace("agent_", "")): [int(c) for c in value]
            for key, value in agents_cfg.get("class_split", {}).items()
        }
    if agents_cfg.get("model_split"):
        args.agent_model_split = {
            int(str(key).replace("agent_", "")): str(value)
            for key, value in agents_cfg.get("model_split", {}).items()
        }

    args.model_name = model_name
    args.net_type = model_cfg.get("family", model_name)
    args.depth = model_cfg.get("depth", args.depth)
    args.width = model_cfg.get("width", args.width)
    args.norm_type = model_cfg.get("norm_type", args.norm_type)
    apply_classifier_overrides(args, model_cfg.get("classifier"))

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
        "evaluate_iter",
        "evaluate_iterations",
        "lr_img",
        "mom_img",
        "grad_clip_norm",
        "guide_model_mode",
        "freeze_guide_parameters",
        "official_dsdm_protocol",
        "official_dsdm_commit",
        "batch_real",
        "batch_syn_max",
        "smooth_iter",
        "cov_weight",
        "h_p_weight",
        "smooth_factor",
        "load_memory",
        "mixup",
        "mixup_net",
        "beta",
        "mix_p",
        "dsa",
        "dsa_strategy",
        "bias",
        "fc",
        "reproduce",
    ]:
        if key in distill:
            setattr(args, key, distill[key])

    apply_pcbn_overrides(args, pcbn_cfg)

    args.epochs = evaluation.get("epochs", args.epochs)
    args.batch_size = evaluation.get("batch_size", args.batch_size)
    args.repeat = evaluation.get("repeat", args.repeat)
    args.seed = runtime.get("seed", args.seed)
    args.workers = runtime.get("workers", args.workers)
    args.device = runtime.get("device", args.device)
    args.gpu_id = runtime.get("gpu_id", args.gpu_id)
    args.communication_protocol = str(communication.get("protocol", "none"))
    args.receiver_protocol = str(receiver.get("protocol", communication.get("receiver_protocol", "legacy")))
    args.strict_packet_validation = _as_bool(communication.get("strict_packet_validation", False))
    args.packet_source = str(communication.get("packet_source", "sender_local"))
    args.use_sender_logits = _as_bool(communication.get("use_sender_logits", False))
    args.dkp_variant = str(receiver.get("dkp_variant", communication.get("dkp_variant", "legacy")))
    args.receiver_local_batch_size = int(receiver.get("local_batch_size", 64))
    args.receiver_external_batch_size = int(receiver.get("external_batch_size", 64))
    args.lambda_sc = float(receiver.get("lambda_sc", 0.0))
    args.supcon_temperature = float(receiver.get("supcon_temperature", 0.07))
    args.prototype_decoded_per_class = int(receiver.get("prototype_decoded_per_class", 40))
    apply_receiver_training_overrides(args, receiver)
    if "loss_switches" in receiver:
        loss_switches = receiver["loss_switches"]
        if not isinstance(loss_switches, dict) or set(loss_switches) != {"fr", "kd", "supcon"}:
            raise ValueError("receiver.loss_switches must contain exactly fr/kd/supcon")
        if any(not isinstance(value, bool) for value in loss_switches.values()):
            raise ValueError("receiver.loss_switches values must be booleans")
        args.dkp_loss_switches = dict(loss_switches)
    if "checkpoint_retention" in receiver:
        args.receiver_checkpoint_retention = normalize_receiver_checkpoint_retention(
            receiver["checkpoint_retention"]
        )
    args.fast_repo_path = fast_cfg.get("repo_path", args.fast_repo_path)
    args.fast_cache_root = fast_cfg.get("cache_root", args.fast_cache_root)
    args.fast_commit = fast_cfg.get("commit", args.fast_commit)
    args.fast_seed = int(fast_cfg.get("seed", args.fast_seed))


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
