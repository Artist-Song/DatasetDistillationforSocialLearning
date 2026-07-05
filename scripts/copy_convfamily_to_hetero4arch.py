"""
scripts/copy_convfamily_to_hetero4arch.py

把 conv_family_ipc10 中 conv3w1（agent_1）和 conv4w1.5（agent_3）的已有结果
复制到 hetero4arch_ipc10 的 agent_0 / agent_1 目录，避免重新蒸馏。

复制内容：
  - checkpoints/expert_model.pt
  - checkpoints/dsdm_pretrained/（teacher bank，10个pth + manifest）
  - checkpoints/args.txt
  - packets/dsdm_packet.pt
  - packets/heuristic_packet.pt
  - packets/full_real_packet.pt
  - synthetic/data_best.pt

用法：
    conda run -n sp python scripts/copy_convfamily_to_hetero4arch.py [--dry-run]
"""

import argparse
import shutil
import os
from pathlib import Path

# 源 run（conv_family_ipc10）
SRC_ROOT = Path("outputs/cifar100_4agent_25cls_conv_family_ipc10/agents")

# 目标 run（hetero4arch_ipc10）
DST_ROOT = Path("outputs/cifar100_4agent_25cls_hetero4arch_ipc10/agents")

# 对应关系：
#   src agent_1 (conv3w1,  classes 25-49) → dst agent_0
#   src agent_3 (conv4w15, classes 75-99) → dst agent_1
COPY_MAP = {
    "agent_1": "agent_0",  # conv3w1.0
    "agent_3": "agent_1",  # conv4w1.5
}

# 需要复制的子路径（相对于 agent 目录）
COPY_ITEMS = [
    "checkpoints/expert_model.pt",
    "checkpoints/dsdm_pretrained",        # 整个目录
    "checkpoints/args.txt",
    "packets/dsdm_packet.pt",
    "packets/heuristic_packet.pt",
    "packets/full_real_packet.pt",
    "synthetic/data_best.pt",
]


def copy_item(src: Path, dst: Path, dry_run: bool):
    """复制单个文件或目录，目标已存在时跳过并提示。"""
    if not src.exists():
        print(f"  ⚠  源不存在，跳过: {src}")
        return

    if dst.exists():
        print(f"  ⏭  目标已存在，跳过: {dst}")
        return

    if dry_run:
        kind = "目录" if src.is_dir() else "文件"
        print(f"  [dry-run] 将复制{kind}: {src} → {dst}")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst)
        print(f"  ✅  复制目录: {src} → {dst}")
    else:
        shutil.copy2(src, dst)
        print(f"  ✅  复制文件: {src} → {dst}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印，不实际复制")
    args = parser.parse_args()

    # 切换到项目根目录（脚本在 scripts/ 子目录下）
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    mode = "[dry-run]" if args.dry_run else "[实际执行]"
    print(f"\ncopy_convfamily_to_hetero4arch.py  {mode}")
    print(f"源 run  : {SRC_ROOT}")
    print(f"目标 run: {DST_ROOT}\n")

    for src_agent, dst_agent in COPY_MAP.items():
        print(f"{'='*55}")
        print(f"  {src_agent} → {dst_agent}")
        print(f"{'='*55}")

        for item in COPY_ITEMS:
            src = SRC_ROOT / src_agent / item
            dst = DST_ROOT / dst_agent / item
            copy_item(src, dst, args.dry_run)

        print()

    if args.dry_run:
        print("dry-run 完成，以上操作均未实际执行。去掉 --dry-run 后重新运行以实际复制。\n")
    else:
        print("复制完成。\n")


if __name__ == "__main__":
    main()
