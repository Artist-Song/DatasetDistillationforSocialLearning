from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Link exported social CIFAR-100 split into FedRE repo.")
    parser.add_argument("--exported-cifar100", required=True, help="导出的 fedre/Cifar100 目录")
    parser.add_argument("--fedre-repo", default="external_baselines/repos/FedRE", help="FedRE 仓库路径")
    parser.add_argument("--copy", action="store_true", help="复制数据而不是创建软链接")
    parser.add_argument("--replace-symlink", action="store_true", help="允许替换已有软链接")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.exported_cifar100).resolve()
    target = Path(args.fedre_repo).resolve() / "HtFLlib" / "dataset" / "Cifar100"
    if not source.exists():
        raise FileNotFoundError(f"source not found: {source}")

    if target.exists() or target.is_symlink():
        if target.is_symlink() and args.replace_symlink:
            target.unlink()
        else:
            raise FileExistsError(
                f"target already exists: {target}. "
                "Use --replace-symlink only for an existing symlink, or remove it manually after checking."
            )

    target.parent.mkdir(parents=True, exist_ok=True)
    if args.copy:
        shutil.copytree(source, target)
        action = "copied"
    else:
        target.symlink_to(source, target_is_directory=True)
        action = "linked"

    print(f"[fedre] {action} {source} -> {target}")
    print("[fedre] run from external_baselines/repos/FedRE/HtFLlib/system with dataset name Cifar100")


if __name__ == "__main__":
    main()
