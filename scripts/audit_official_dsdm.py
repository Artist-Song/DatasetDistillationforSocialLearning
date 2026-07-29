#!/usr/bin/env python3
"""Fail closed unless the local DSDM primitives match the pinned official repo."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "DSDM"
OFFICIAL = ROOT / "external_baselines" / "repos" / "DSDM_official"
EXPECTED_COMMIT = "cb12851831e39da6b0169da84598166ad7706e01"
EXACT_FILES = (
    "misc/augment.py",
    "misc/utils.py",
    "models/convnet.py",
)
EXACT_FUNCTIONS = {
    "pre_train_model.py": ("load_resized_data", "remove_aug", "diffaug"),
    "train.py": ("train", "train_epoch", "validate"),
    "DSDM.py": ("remove_aug", "diffaug", "dist", "add_loss"),
    "data.py": ("transform_cifar",),
}
RUNTIME_KEYS = (
    "aug_type",
    "augment",
    "batch_real",
    "batch_size",
    "batch_syn_max",
    "beta",
    "bias",
    "bottleneck",
    "cov_weight",
    "dataset",
    "decode_type",
    "depth",
    "dsa",
    "dsa_strategy",
    "dseed",
    "epoch_print_freq",
    "epochs",
    "evaluate_iter",
    "f_idx",
    "factor",
    "fc",
    "h_p_weight",
    "idx_from",
    "idx_to",
    "init",
    "ipc",
    "load_memory",
    "lr",
    "lr_img",
    "match",
    "metric",
    "mix_p",
    "mixup",
    "mixup_net",
    "mom_img",
    "momentum",
    "nch",
    "nclass",
    "nclass_sub",
    "net_type",
    "niter",
    "norm_type",
    "phase",
    "pretrained",
    "pretrained_epochs",
    "pretrained_model_number",
    "print_freq",
    "repeat",
    "reproduce",
    "rrc",
    "same_compute",
    "save_ckpt",
    "seed",
    "size",
    "slct_type",
    "smooth_factor",
    "smooth_iter",
    "test",
    "time",
    "verbose",
    "weight_decay",
    "width",
    "workers",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _functions(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name: ast.dump(node, annotate_fields=True, include_attributes=False)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def main() -> None:
    if not (OFFICIAL / ".git").is_dir():
        raise SystemExit(
            "missing official DSDM checkout; clone https://github.com/Li-Hongcheng/DSDM "
            f"to {OFFICIAL}"
        )
    commit = subprocess.check_output(
        ["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != EXPECTED_COMMIT:
        raise SystemExit(f"official DSDM commit mismatch: {commit} != {EXPECTED_COMMIT}")

    checked = []
    for relative in EXACT_FILES:
        local_hash = _sha256(LOCAL / relative)
        official_hash = _sha256(OFFICIAL / relative)
        if local_hash != official_hash:
            raise SystemExit(f"official DSDM file mismatch: {relative}")
        checked.append(f"file:{relative}")

    for relative, names in EXACT_FUNCTIONS.items():
        local_functions = _functions(LOCAL / relative)
        official_functions = _functions(OFFICIAL / relative)
        for name in names:
            if local_functions.get(name) != official_functions.get(name):
                raise SystemExit(f"official DSDM function mismatch: {relative}::{name}")
            checked.append(f"function:{relative}::{name}")

    official_command = (
        "import sys,json; "
        "sys.argv=['argument.py','--reproduce','-d','cifar100','-f','2','--ipc','10',"
        "'--pretrained_epochs','200','--evaluate_iter','500']; "
        "import argument; print(json.dumps(vars(argument.args),sort_keys=True))"
    )
    output = subprocess.check_output(
        [sys.executable, "-c", official_command], cwd=OFFICIAL, text=True
    )
    official_args = json.loads(output.strip().splitlines()[-1])

    sys.path.insert(0, str(ROOT))
    from agent_data import build_agent_args
    from scripts.prepare_fullclass_dsdm import build_config

    local_args = vars(build_agent_args(build_config("conv3"), "generated", 0))
    mismatches = {
        key: {"official": official_args.get(key), "local": local_args.get(key)}
        for key in RUNTIME_KEYS
        if official_args.get(key) != local_args.get(key)
    }
    if mismatches:
        raise SystemExit(f"official DSDM resolved-argument mismatch: {json.dumps(mismatches, sort_keys=True)}")
    checked.append("runtime:official_reproduce_cifar100_ipc10_vs_local_conv3")

    print(
        json.dumps(
            {
                "official_repo": "https://github.com/Li-Hongcheng/DSDM",
                "official_commit": commit,
                "checks": checked,
                "status": "passed",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
