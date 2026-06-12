import argparse
import os
import sys
from pathlib import Path

from agent_data import (
    AGENT_CLASS_SPLIT,
    AGENT_MODEL_SPLIT,
    build_agent_args,
    get_agent_dir,
    get_agent_ids,
    get_receiver_ids,
)
from agent_trainer import prepare_agent_pretrained_dir, train_agent_experts
from config_adapter import args_to_pretty_json, build_dsdm_args_from_config, load_config
from social_output_manager import (
    append_social_result,
    prepare_social_output_dirs,
    read_packet_manifest,
    register_agent_packet,
    save_social_config,
    write_packet_manifest,
)
from social_trainer import SocialTrainer


ROOT = Path(__file__).resolve().parent
DSDM_ROOT = ROOT / "DSDM"
if str(DSDM_ROOT) not in sys.path:
    sys.path.insert(0, str(DSDM_ROOT))


def parse_cli():
    """解析第二阶段社会化学习主入口参数。"""
    parser = argparse.ArgumentParser(description="第二阶段社会化 packet 学习闭环")
    parser.add_argument("--config", default="configs/main.yaml", help="主配置文件路径")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["train_experts", "distill_packets", "build_communication", "train_receivers", "all"],
        help="运行阶段",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不启动训练")
    parser.add_argument("--resume", action="store_true", help="已存在输出时尽量跳过")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有输出")
    parser.add_argument("--only-agent", type=int, default=None, help="只处理指定 sender agent")
    parser.add_argument("--only-receiver", type=int, default=None, help="只处理指定 receiver agent")
    return parser.parse_args()


def _print_dry_run(args, cli):
    """打印第二阶段 dry-run 执行计划。"""
    print("social pipeline dry-run")
    print(args_to_pretty_json(args))
    print(f"stage: {cli.stage}")
    print(f"agents: {get_agent_ids(cli.only_agent)}")
    print(f"receivers: {get_receiver_ids(cli.only_receiver)}")
    for agent_id, classes in AGENT_CLASS_SPLIT.items():
        print(f"agent {agent_id}: model={AGENT_MODEL_SPLIT[agent_id]} classes={classes}")
    print(f"run_dir: {Path(args.output_root) / args.run_name}")


def _stage_train_experts(cfg, config_path, base_args, cli):
    """训练每个 agent 的 expert guide model pool。"""
    for agent_id in get_agent_ids(cli.only_agent):
        agent_args = build_agent_args(cfg, config_path, agent_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(agent_args.gpu_id)
        print(f"[train_experts] agent={agent_id} classes={AGENT_CLASS_SPLIT[agent_id]}")
        train_agent_experts(agent_args, agent_id, resume=cli.resume, overwrite=cli.overwrite)


def _stage_distill_packets(cfg, config_path, base_args, cli):
    """为每个 agent 运行 DSDM 蒸馏并生成自己的 packet。"""
    from DSDM import run_dsdm

    for agent_id in get_agent_ids(cli.only_agent):
        agent_args = build_agent_args(cfg, config_path, agent_id)
        agent_args.save_pretrain_dir = str(prepare_agent_pretrained_dir(agent_args, agent_id))
        agent_args.save_dir = str(get_agent_dir(agent_args, agent_id) / "checkpoints")
        agent_args.output_root = str(get_agent_dir(agent_args, agent_id))
        agent_args.run_name = ""
        os.environ["CUDA_VISIBLE_DEVICES"] = str(agent_args.gpu_id)
        print(f"[distill_packets] agent={agent_id} model={agent_args.net_type} classes={agent_args.active_class_ids}")
        run_dsdm(agent_args)


def _stage_build_communication(base_args, cli):
    """把 agent packet 注册到 packet_hub 并写 manifest。"""
    rows = []
    if cli.only_agent is not None:
        print("[warning] --only-agent build_communication 只更新指定 agent，避免覆盖完整 manifest。")
        try:
            rows = read_packet_manifest(base_args)
            rows = [row for row in rows if int(row["sender_agent"]) != int(cli.only_agent)]
        except FileNotFoundError:
            print("[warning] 当前没有已有 manifest，将写入只包含指定 agent 的临时 manifest。")
    for agent_id in get_agent_ids(cli.only_agent):
        agent_dir = get_agent_dir(base_args, agent_id)
        packet_path = agent_dir / "packets" / "dsdm_packet.pt"
        if not packet_path.exists():
            raise FileNotFoundError(f"缺少 agent packet: {packet_path}")
        rows.append(register_agent_packet(base_args, agent_id, packet_path))
    manifest_path = write_packet_manifest(base_args, rows)
    print(f"[build_communication] manifest: {manifest_path}")


def _stage_train_receivers(base_args, cli):
    """读取 packet_hub 并训练每个 receiver。"""
    cfg = load_config(cli.config)
    rows = read_packet_manifest(base_args)
    for receiver_id in get_receiver_ids(cli.only_receiver):
        receiver_args = build_agent_args(cfg, cli.config, receiver_id)
        receiver_cfg = cfg.get("social_learning", {}).get("receiver", {})
        receiver_args.receiver_epochs = receiver_cfg.get("epochs", receiver_args.epochs)
        receiver_args.receiver_lr = receiver_cfg.get("lr", receiver_args.lr)
        receiver_args.lambda_fr = receiver_cfg.get("lambda_fr", 0.05)
        print(f"[train_receivers] receiver={receiver_id} classes={receiver_args.active_class_ids}")
        result = SocialTrainer(receiver_args, receiver_id, rows).train()
        append_social_result(base_args, result)


def main():
    """运行第二阶段社会化学习闭环。"""
    cli = parse_cli()
    cfg = load_config(cli.config)
    base_args = build_dsdm_args_from_config(cfg, config_path=cli.config)
    base_args.num_classes = 10
    prepare_social_output_dirs(base_args)
    save_social_config(cli.config, base_args)

    if cli.dry_run:
        _print_dry_run(base_args, cli)
        return

    if cli.stage in {"train_experts", "all"}:
        _stage_train_experts(cfg, cli.config, base_args, cli)
    if cli.stage in {"distill_packets", "all"}:
        _stage_distill_packets(cfg, cli.config, base_args, cli)
    if cli.stage in {"build_communication", "all"}:
        _stage_build_communication(base_args, cli)
    if cli.stage in {"train_receivers", "all"}:
        _stage_train_receivers(base_args, cli)


if __name__ == "__main__":
    main()
