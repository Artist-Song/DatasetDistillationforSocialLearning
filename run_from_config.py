import argparse
import os
import sys
from pathlib import Path

import torch

from config_adapter import args_to_pretty_json, build_dsdm_args_from_config, load_config
from output_manager import (
    get_metrics_path,
    get_packet_path,
    get_run_dir,
    prepare_output_dirs,
    save_config_file_snapshot,
    save_packet,
    save_packet_visual,
    save_resolved_args,
)
from selection_methods import build_heuristic_packet, build_importance_packet


ROOT = Path(__file__).resolve().parent
DSDM_ROOT = ROOT / "DSDM"
if str(DSDM_ROOT) not in sys.path:
    sys.path.insert(0, str(DSDM_ROOT))


def parse_cli():
    """解析配置入口命令行参数。"""
    parser = argparse.ArgumentParser(description="从主配置运行 DSDM Stage 1")
    parser.add_argument("--config", default="configs/main.yaml", help="主配置文件路径")
    parser.add_argument("--dry-run", action="store_true", help="只解析配置并打印输出路径")
    return parser.parse_args()


def print_dry_run(args):
    """打印 dry-run 需要确认的参数和输出路径。"""
    print("resolved args:")
    print(args_to_pretty_json(args))
    print(f"run_dir: {get_run_dir(args)}")
    print(f"synthetic path: {get_run_dir(args) / 'synthetic' / 'data_best.pt'}")
    print(f"dsdm packet path: {get_packet_path(args, 'dsdm')}")
    print(f"heuristic packet path: {get_packet_path(args, 'heuristic')}")
    print(f"importance packet path: {get_packet_path(args, 'importance')}")
    print(f"metrics path: {get_metrics_path(args, 'packet_eval_dsdm')}")


def _load_guide_models(args, nclass):
    """从 DSDM 预训练目录加载 guide 模型池。"""
    from train import define_model

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    models = []
    for index in range(int(args.pretrained_model_number)):
        path = Path(args.save_pretrain_dir) / f"{args.dataset}_model_{index}.pth"
        if not path.exists():
            continue
        model = define_model(args, nclass)
        model.load_state_dict(torch.load(path, map_location=device))
        models.append(model.to(device))
    return models


def build_and_save_selection_packets(args):
    """基于真实训练集构建并保存 Heuristic 和 Importance packet。"""
    from DSDM import load_resized_data

    train_set, _ = load_resized_data(args)
    heuristic = build_heuristic_packet(args, train_set)
    save_packet(args, *heuristic, source="heuristic", method="Heuristic")
    save_packet_visual(args, "heuristic", heuristic[0])

    guide_models = _load_guide_models(args, len(heuristic[2]))
    importance = build_importance_packet(args, train_set, guide_models)
    meta = None if guide_models else {"fallback": "heuristic_when_no_guide_model"}
    save_packet(args, *importance, source="importance", method="Importance", meta=meta)
    save_packet_visual(args, "importance", importance[0])


def main():
    """读取主配置，准备输出目录，并按需启动 DSDM。"""
    cli = parse_cli()
    cfg = load_config(cli.config)
    args = build_dsdm_args_from_config(cfg, config_path=cli.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    prepare_output_dirs(args)

    if cli.dry_run:
        print_dry_run(args)
        return

    save_config_file_snapshot(cli.config, args)
    save_resolved_args(args)

    from DSDM import run_dsdm

    run_dsdm(args)
    build_and_save_selection_packets(args)


if __name__ == "__main__":
    main()
