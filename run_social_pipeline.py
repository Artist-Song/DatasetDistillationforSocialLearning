import argparse
import os
import sys
from pathlib import Path

from agent_data import (
    build_agent_args,
    get_agent_dir,
    get_agent_class_split,
    get_agent_ids,
    get_agent_model_split,
    get_num_classes,
    get_receiver_ids,
)
from agent_trainer import prepare_agent_pretrained_dir, train_agent_experts
from config_adapter import args_to_pretty_json, build_dsdm_args_from_config, load_config
from output_manager import finalize_dsdm_packet, save_packet
from packet_logits import attach_sender_logits_to_packet
from packet_generalist import (
    attach_generalist_logits_to_packets,
    ensure_reused_artifacts,
    train_packet_generalist,
)
from progress_timer import ProgressTimer
from selection_methods import (
    build_fast_packet,
    build_full_real_packet,
    build_heuristic_packet,
    build_importance_packet,
)
from social_output_manager import (
    append_social_result,
    prepare_social_output_dirs,
    read_packet_manifest,
    register_agent_packet,
    save_social_config,
    write_packet_manifest,
)
from social_trainer import SocialTrainer, resolve_dkp_loss_switches


ROOT = Path(__file__).resolve().parent
DSDM_ROOT = ROOT / "DSDM"
if str(DSDM_ROOT) not in sys.path:
    sys.path.insert(0, str(DSDM_ROOT))


def _stage_banner(stage_name, detail=""):
    """打印阶段开始提示，便于在长日志中定位进度。"""
    print("")
    print("=" * 72)
    print(f"[stage] {stage_name}")
    if detail:
        print(f"[detail] {detail}")
    print("=" * 72)


def _stage_done(stage_name):
    """打印阶段完成提示。"""
    print(f"[done] {stage_name}")
    print("")


def parse_cli():
    """解析第二阶段社会化学习主入口参数。"""
    parser = argparse.ArgumentParser(description="第二阶段社会化 packet 学习闭环")
    parser.add_argument("--config", default="configs/main.yaml", help="主配置文件路径")
    parser.add_argument(
        "--stage",
        default="all",
        choices=[
            "train_experts",
            "distill_packets",
            "build_selection_packets",
            "attach_logits",
            "train_packet_generalist",
            "attach_generalist_logits",
            "build_communication",
            "train_receivers",
            "all",
        ],
        help="运行阶段",
    )
    parser.add_argument(
        "--packet-method",
        default="dsdm",
        choices=["dsdm", "heuristic", "importance", "fast", "full_real"],
        help="packet 方法",
    )
    parser.add_argument("--init-mode", default="expert", choices=["expert", "scratch"], help="receiver 初始化方式")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不启动训练")
    parser.add_argument("--resume", action="store_true", help="已存在输出时尽量跳过")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有输出")
    parser.add_argument("--only-agent", type=int, default=None, help="只处理指定 sender agent")
    parser.add_argument("--only-receiver", type=int, default=None, help="只处理指定 receiver agent")
    return parser.parse_args()


def _print_dry_run(args, cli):
    """打印第二阶段 dry-run 执行计划。"""
    class_split = get_agent_class_split(args)
    model_split = get_agent_model_split(args)
    cfg = load_config(cli.config)
    print("social pipeline dry-run")
    print(args_to_pretty_json(args))
    print(f"stage: {cli.stage}")
    print(f"stages: {_build_stages(cfg, cli)}")
    print(f"packet_method: {cli.packet_method}")
    print(f"init_mode: {cli.init_mode}")
    print(f"dataset: {args.dataset}")
    print(f"num_classes: {args.num_classes}")
    print(f"num_agents: {len(class_split)}")
    print(f"agents: {get_agent_ids(args, cli.only_agent)}")
    print(f"receivers: {get_receiver_ids(args, cli.only_receiver)}")
    if cli.stage in {"train_receivers", "all"}:
        receiver_cfg = cfg.get("social_learning", {}).get("receiver", {})
        print(f"receiver self_data_mode: {receiver_cfg.get('self_data_mode', 'packet')}")
        print(f"receiver self_real_per_class: {receiver_cfg.get('self_real_per_class', 0)}")
        print(f"receiver self_class_weight: {receiver_cfg.get('self_class_weight', 1.0)}")
    for agent_id, classes in class_split.items():
        agent_args = build_agent_args(cfg, cli.config, agent_id)
        print(
            f"agent {agent_id}: model={model_split[agent_id]} "
            f"net_type={agent_args.net_type} depth={agent_args.depth} "
            f"width={agent_args.width} norm={agent_args.norm_type} classes={classes}"
        )
    if cli.stage in {"attach_logits", "attach_generalist_logits"}:
        for agent_id in get_agent_ids(args, cli.only_agent):
            packet_path = get_agent_dir(args, agent_id) / "packets" / f"{cli.packet_method}_packet.pt"
            print(f"{cli.stage} packet: agent={agent_id} path={packet_path}")
    if cli.stage == "train_packet_generalist":
        generalist_cfg = cfg.get("generalist", {})
        reuse_cfg = cfg.get("reuse", {})
        print(f"generalist model: {generalist_cfg.get('model', '<default>')}")
        print(f"reuse source_run_name: {reuse_cfg.get('source_run_name', '')}")
    print(f"run_dir: {Path(args.output_root) / args.run_name}")


def _build_stages(cfg, cli):
    """根据命令行和 logits 配置构建实际阶段列表。"""
    logits_enabled = bool(cfg.get("logits", {}).get("enabled", False))
    stages = []
    if cli.stage in {"train_experts", "all"}:
        stages.append("train_experts")
    if cli.stage in {"distill_packets", "all"}:
        stages.append("distill_packets")
    if cli.stage in {"build_selection_packets"}:
        stages.append("build_selection_packets")
    if cli.stage in {"attach_logits"} or (cli.stage == "all" and logits_enabled):
        stages.append("attach_logits")
    generalist_enabled = bool(cfg.get("generalist", {}).get("enabled", False))
    if cli.stage in {"train_packet_generalist"} or (cli.stage == "all" and generalist_enabled):
        stages.append("train_packet_generalist")
    if cli.stage in {"attach_generalist_logits"} or (cli.stage == "all" and generalist_enabled):
        stages.append("attach_generalist_logits")
    if cli.stage in {"build_communication", "all"}:
        stages.append("build_communication")
    if cli.stage in {"train_receivers", "all"}:
        stages.append("train_receivers")
    return stages


def _stage_train_experts(cfg, config_path, base_args, cli):
    """训练每个 agent 的 expert guide model pool。"""
    class_split = get_agent_class_split(base_args)
    agent_ids = get_agent_ids(base_args, cli.only_agent)
    _stage_banner("train_experts", f"agents={agent_ids}")
    progress = ProgressTimer(len(agent_ids), name="train_experts")
    for index, agent_id in enumerate(agent_ids, start=1):
        agent_args = build_agent_args(cfg, config_path, agent_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(agent_args.gpu_id)
        print(f"[train_experts] agent={agent_id} classes={class_split[agent_id]}")
        train_agent_experts(agent_args, agent_id, resume=cli.resume, overwrite=cli.overwrite)
        progress.update(index, extra=f"agent={agent_id}")
    _stage_done("train_experts")


def _stage_distill_packets(cfg, config_path, base_args, cli):
    """为每个 agent 运行 DSDM 蒸馏并生成自己的 packet。"""
    from DSDM import run_dsdm

    agent_ids = get_agent_ids(base_args, cli.only_agent)
    _stage_banner("distill_packets", f"agents={agent_ids}")
    progress = ProgressTimer(len(agent_ids), name="distill_packets")
    for index, agent_id in enumerate(agent_ids, start=1):
        agent_args = build_agent_args(cfg, config_path, agent_id)
        agent_dir = get_agent_dir(agent_args, agent_id)
        agent_args.save_pretrain_dir = str(prepare_agent_pretrained_dir(agent_args, agent_id))
        agent_args.save_dir = str(agent_dir / "checkpoints")
        agent_args.output_root = str(agent_dir)
        agent_args.run_name = ""
        os.environ["CUDA_VISIBLE_DEVICES"] = str(agent_args.gpu_id)
        print(f"[distill_packets] agent={agent_id} model={agent_args.net_type} classes={agent_args.active_class_ids}")
        numerical_summary = run_dsdm(agent_args)
        packet_path = agent_dir / "packets" / "dsdm_packet.pt"
        if not packet_path.exists():
            raise FileNotFoundError(f"DSDM 完成后缺少 packet: {packet_path}")
        import torch

        packet = torch.load(packet_path, map_location="cpu")
        images = packet.get("images")
        if not torch.is_tensor(images) or not bool(torch.isfinite(images).all().item()):
            raise FloatingPointError(f"DSDM packet images 非有限: {packet_path}")
        finalize_dsdm_packet(packet_path, numerical_summary)
        progress.update(index, extra=f"agent={agent_id}")
    _stage_done("distill_packets")


def _load_agent_guide_models(args, agent_id):
    """加载单个 agent 的 guide model pool，用于 importance 选样。"""
    import torch
    from train import define_model

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    models = []
    ckpt_dir = get_agent_dir(args, agent_id) / "checkpoints"
    for model_idx in range(int(args.pretrained_model_number)):
        path = ckpt_dir / f"guide_model_{model_idx}.pt"
        if not path.exists():
            continue
        model = define_model(args, int(args.num_classes)).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        models.append(model)
    return models


def _stage_build_selection_packets(cfg, config_path, base_args, cli):
    """为每个 agent 构建真实图选择方法的 packet。"""
    from agent_data import get_agent_train_dataset

    if cli.packet_method not in {"heuristic", "importance", "fast", "full_real"}:
        print("[build_selection_packets] dsdm packet 由 distill_packets 生成，此阶段跳过。")
        return
    agent_ids = get_agent_ids(base_args, cli.only_agent)
    _stage_banner("build_selection_packets", f"method={cli.packet_method} agents={agent_ids}")
    progress = ProgressTimer(len(agent_ids), name=f"build_{cli.packet_method}")
    for index, agent_id in enumerate(agent_ids, start=1):
        agent_args = build_agent_args(cfg, config_path, agent_id)
        train_set = get_agent_train_dataset(agent_args, agent_id, normalize=False)
        packet_meta = None
        if cli.packet_method == "heuristic":
            images, labels, class_ids = build_heuristic_packet(agent_args, train_set)
        elif cli.packet_method == "fast":
            images, labels, class_ids, packet_meta = build_fast_packet(agent_args, train_set)
        elif cli.packet_method == "full_real":
            images, labels, class_ids = build_full_real_packet(agent_args, train_set)
        else:
            guide_models = _load_agent_guide_models(agent_args, agent_id)
            images, labels, class_ids = build_importance_packet(agent_args, train_set, guide_models)
        agent_args.output_root = str(get_agent_dir(agent_args, agent_id))
        agent_args.run_name = ""
        save_packet(
            agent_args,
            images,
            labels,
            class_ids,
            source=cli.packet_method,
            method=cli.packet_method.upper(),
            meta=packet_meta,
        )
        progress.update(index, extra=f"agent={agent_id}")
    _stage_done("build_selection_packets")


def _stage_attach_logits(cfg, config_path, base_args, cli):
    """为指定 packet method 的 agent packets 附加 sender logits。"""
    if cli.packet_method == "fast":
        raise ValueError("FAST baseline 只使用选中的真实图和 hard labels，不附加 logits。")
    agent_ids = get_agent_ids(base_args, cli.only_agent)
    _stage_banner("attach_logits", f"method={cli.packet_method} agents={agent_ids}")
    progress = ProgressTimer(len(agent_ids), name="attach_logits")
    for index, agent_id in enumerate(agent_ids, start=1):
        agent_args = build_agent_args(cfg, config_path, agent_id)
        attach_sender_logits_to_packet(
            agent_args,
            agent_id,
            cli.packet_method,
            resume=cli.resume,
            overwrite=cli.overwrite,
        )
        progress.update(index, extra=f"agent={agent_id}")
    _stage_done("attach_logits")



def _stage_train_packet_generalist(cfg, base_args, cli):
    """用复用的 communicated packets 训练 packet generalist。"""
    _stage_banner("train_packet_generalist", f"method={cli.packet_method}")
    copied = ensure_reused_artifacts(cfg, base_args, cli.packet_method)
    if copied:
        print(f"[reuse] copied artifacts: {len(copied)}")
    train_packet_generalist(cfg, base_args, cli.packet_method)
    _stage_done("train_packet_generalist")


def _stage_attach_generalist_logits(cfg, base_args, cli):
    """用 packet generalist 为 packets 附加 full-class logits。"""
    agent_ids = get_agent_ids(base_args, cli.only_agent)
    _stage_banner("attach_generalist_logits", f"method={cli.packet_method} agents={agent_ids}")
    ensure_reused_artifacts(cfg, base_args, cli.packet_method)
    attach_generalist_logits_to_packets(cfg, base_args, cli.packet_method, only_agent=cli.only_agent)
    _stage_done("attach_generalist_logits")

def _stage_build_communication(base_args, cli):
    """把 agent packet 注册到 packet_hub 并写 manifest。"""
    agent_ids = get_agent_ids(base_args, cli.only_agent)
    _stage_banner("build_communication", f"method={cli.packet_method} agents={agent_ids}")
    rows = []
    if cli.only_agent is not None:
        print("[warning] --only-agent build_communication 只更新指定 agent，避免覆盖完整 manifest。")
        try:
            rows = read_packet_manifest(base_args, cli.packet_method)
            rows = [row for row in rows if int(row["sender_agent"]) != int(cli.only_agent)]
        except FileNotFoundError:
            print("[warning] 当前没有已有 manifest，将写入只包含指定 agent 的临时 manifest。")
    progress = ProgressTimer(len(agent_ids), name="build_communication")
    for index, agent_id in enumerate(agent_ids, start=1):
        agent_dir = get_agent_dir(base_args, agent_id)
        packet_path = agent_dir / "packets" / f"{cli.packet_method}_packet.pt"
        if not packet_path.exists():
            raise FileNotFoundError(f"缺少 agent packet: {packet_path}")
        rows.append(register_agent_packet(base_args, agent_id, packet_path, cli.packet_method))
        progress.update(index, extra=f"agent={agent_id}")
    manifest_path = write_packet_manifest(base_args, rows, cli.packet_method)
    print(f"[build_communication] manifest: {manifest_path}")
    _stage_done("build_communication")


def build_receiver_args(cfg, config_path, receiver_id, packet_method="dsdm", init_mode="expert"):
    """Resolve exactly the receiver arguments used by training and preflight."""
    receiver_args = build_agent_args(cfg, config_path, receiver_id)
    receiver_cfg = cfg.get("social_learning", {}).get("receiver", {})
    receiver_args.receiver_epochs = receiver_cfg.get("epochs", receiver_args.epochs)
    receiver_args.receiver_lr = receiver_cfg.get("lr", receiver_args.lr)
    receiver_args.lambda_fr = receiver_cfg.get("lambda_fr", 0.05)
    receiver_args.self_data_mode = str(receiver_cfg.get("self_data_mode", "packet"))
    receiver_args.self_real_per_class = int(receiver_cfg.get("self_real_per_class", 0) or 0)
    receiver_args.self_class_weight = float(receiver_cfg.get("self_class_weight", 1.0))
    receiver_args.receiver_scheduler = str(receiver_cfg.get("scheduler", "none"))
    receiver_args.receiver_scheduler_milestones = receiver_cfg.get("scheduler_milestones", [])
    receiver_args.receiver_scheduler_gamma = float(receiver_cfg.get("scheduler_gamma", 0.2))
    receiver_args.receiver_augment = bool(receiver_cfg.get("augment", False))
    receiver_args.freeze_bn_stats = bool(receiver_cfg.get("freeze_bn_stats", False))
    receiver_args.lambda_schedule = str(receiver_cfg.get("lambda_schedule", "none"))
    receiver_args.lambda_schedule_switch = float(receiver_cfg.get("lambda_schedule_switch", 0.7))
    receiver_args.lambda_fr_late_multiplier = float(receiver_cfg.get("lambda_fr_late_multiplier", 1.5))
    receiver_args.lambda_kd_late_multiplier = float(receiver_cfg.get("lambda_kd_late_multiplier", 0.7))
    logits_cfg = cfg.get("logits", {})
    communication_cfg = cfg.get("communication", {})
    receiver_args.receiver_protocol = str(
        receiver_cfg.get("protocol", communication_cfg.get("receiver_protocol", "legacy"))
    )
    receiver_args.dkp_variant = str(
        receiver_cfg.get("dkp_variant", communication_cfg.get("dkp_variant", "legacy"))
    )
    receiver_args.receiver_local_batch_size = int(receiver_cfg.get("local_batch_size", 64))
    receiver_args.receiver_external_batch_size = int(receiver_cfg.get("external_batch_size", 64))
    receiver_args.lambda_sc = float(receiver_cfg.get("lambda_sc", 0.0))
    receiver_args.supcon_temperature = float(receiver_cfg.get("supcon_temperature", 0.07))
    receiver_args.prototype_decoded_per_class = int(receiver_cfg.get("prototype_decoded_per_class", 40))
    if "checkpoint_retention" in receiver_cfg:
        from config_adapter import normalize_receiver_checkpoint_retention

        receiver_args.receiver_checkpoint_retention = normalize_receiver_checkpoint_retention(
            receiver_cfg["checkpoint_retention"]
        )
    receiver_args.communication_mode = communication_cfg.get("mode", "direct")
    receiver_args.use_logits = bool(
        communication_cfg.get("use_sender_logits", logits_cfg.get("enabled", False))
    )
    if packet_method == "fast":
        receiver_args.use_logits = False
    dkp_loss_switches = resolve_dkp_loss_switches(
        receiver_args.dkp_variant,
        getattr(receiver_args, "dkp_loss_switches", None),
    ) if receiver_args.receiver_protocol == "dkp_sl_v1" else None
    if bool(getattr(receiver_args, "strict_packet_validation", False)):
        if receiver_args.receiver_protocol != "dkp_sl_v1":
            raise ValueError("Strict DKP receiver requires receiver protocol dkp_sl_v1")
        if receiver_args.use_logits is not dkp_loss_switches["kd"]:
            raise ValueError(
                "communication.use_sender_logits must exactly match the DKP KD switch"
            )
    if dkp_loss_switches is not None:
        configured_weights = {
            "fr": float(receiver_args.lambda_fr),
            "kd": float(logits_cfg.get("lambda_kd", 0.0)),
            "supcon": float(receiver_args.lambda_sc),
        }
        for loss_name, enabled in dkp_loss_switches.items():
            value = configured_weights[loss_name]
            if enabled and value <= 0.0:
                raise ValueError(f"enabled DKP loss {loss_name} requires a positive weight")
            if not enabled and value != 0.0:
                raise ValueError(f"disabled DKP loss {loss_name} requires a zero weight")
    receiver_args.use_generalist_logits = bool(communication_cfg.get("use_generalist_logits", False))
    receiver_args.kd_mix_beta = float(communication_cfg.get("kd_mix_beta", 0.5))
    receiver_args.lambda_kd = (
        float(logits_cfg.get("lambda_kd", 0.5))
        if (receiver_args.use_logits or receiver_args.use_generalist_logits)
        else 0.0
    )
    receiver_args.kd_temperature = float(logits_cfg.get("temperature", 2.0))
    receiver_args.packet_method = packet_method
    receiver_args.init_mode = init_mode
    receiver_args.use_fr = init_mode == "expert"
    if dkp_loss_switches is not None:
        receiver_args.use_fr = dkp_loss_switches["fr"]
    if init_mode == "scratch":
        receiver_args.lambda_fr = 0.0
    return receiver_args


def _stage_train_receivers(base_args, cli):
    """读取 packet_hub 并训练每个 receiver。"""
    cfg = load_config(cli.config)
    rows = read_packet_manifest(base_args, cli.packet_method)
    receiver_ids = get_receiver_ids(base_args, cli.only_receiver)
    _stage_banner("train_receivers", f"receivers={receiver_ids}")
    progress = ProgressTimer(len(receiver_ids), name="train_receivers")
    for index, receiver_id in enumerate(receiver_ids, start=1):
        receiver_args = build_receiver_args(
            cfg,
            cli.config,
            receiver_id,
            packet_method=cli.packet_method,
            init_mode=cli.init_mode,
        )
        print(
            f"[train_receivers] receiver={receiver_id} classes={receiver_args.active_class_ids} "
            f"self_data_mode={receiver_args.self_data_mode} self_real_per_class={receiver_args.self_real_per_class}"
        )
        result = SocialTrainer(receiver_args, receiver_id, rows).train()
        append_social_result(base_args, result)
        progress.update(index, extra=f"receiver={receiver_id}")
    _stage_done("train_receivers")


def main():
    """运行第二阶段社会化学习闭环。"""
    cli = parse_cli()
    cfg = load_config(cli.config)
    base_args = build_dsdm_args_from_config(cfg, config_path=cli.config)
    base_args.num_classes = get_num_classes(base_args)
    base_args.nclass = base_args.num_classes
    prepare_social_output_dirs(base_args)
    save_social_config(cli.config, base_args)

    if cli.dry_run:
        _print_dry_run(base_args, cli)
        return

    stages = _build_stages(cfg, cli)

    _stage_banner("social_pipeline", f"stages={stages}")
    pipeline_progress = ProgressTimer(len(stages), name="social_pipeline")
    finished = 0

    if "train_experts" in stages:
        _stage_train_experts(cfg, cli.config, base_args, cli)
        finished += 1
        pipeline_progress.update(finished, extra="train_experts")
    if "distill_packets" in stages:
        _stage_distill_packets(cfg, cli.config, base_args, cli)
        finished += 1
        pipeline_progress.update(finished, extra="distill_packets")
    if "build_selection_packets" in stages:
        _stage_build_selection_packets(cfg, cli.config, base_args, cli)
        finished += 1
        pipeline_progress.update(finished, extra="build_selection_packets")
    if "attach_logits" in stages:
        _stage_attach_logits(cfg, cli.config, base_args, cli)
        finished += 1
        pipeline_progress.update(finished, extra="attach_logits")
    if "train_packet_generalist" in stages:
        _stage_train_packet_generalist(cfg, base_args, cli)
        finished += 1
        pipeline_progress.update(finished, extra="train_packet_generalist")
    if "attach_generalist_logits" in stages:
        _stage_attach_generalist_logits(cfg, base_args, cli)
        finished += 1
        pipeline_progress.update(finished, extra="attach_generalist_logits")
    if "build_communication" in stages:
        _stage_build_communication(base_args, cli)
        finished += 1
        pipeline_progress.update(finished, extra="build_communication")
    if "train_receivers" in stages:
        _stage_train_receivers(base_args, cli)
        finished += 1
        pipeline_progress.update(finished, extra="train_receivers")
    _stage_done("social_pipeline")


if __name__ == "__main__":
    main()
