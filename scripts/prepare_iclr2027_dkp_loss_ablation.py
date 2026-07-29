#!/usr/bin/env python3
"""Prepare the six missing seed0 DKP-SL loss-ablation configs."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_data import get_agent_class_split, get_agent_model_split  # noqa: E402


BASE_CONFIG = ROOT / "configs/iclr2027/cifar100_5agent20cls_dkp_sl_full_ipc10_seed0_v2.yaml"
TARGET_DIR = ROOT / "configs/iclr2027"
EXPERT_RUN = "cifar100_5agent20cls_dkp_cosine_experts_seed0_v1"
EXPECTED_MODELS = {
    0: "convnet3w1",
    1: "convnet4w15",
    2: "alexnet",
    3: "resnet10_standard",
    4: "resnet18_standard",
}
FIXED_HYPERPARAMETERS = {
    "lambda_fr": 0.2,
    "lambda_kd": 0.6,
    "lambda_sc": 0.1,
    "kd_temperature": 2.0,
    "supcon_temperature": 0.07,
}
ENDPOINTS = {(False, False, False), (True, True, True)}
MISSING_COMBINATIONS = tuple(
    (fr, kd, sc)
    for fr in (False, True)
    for kd in (False, True)
    for sc in (False, True)
    if (fr, kd, sc) not in ENDPOINTS
)


class AblationConfigError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AblationConfigError(message)


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    _require(isinstance(payload, dict), f"config is not a mapping: {path}")
    return payload


def variant_id(fr: bool, kd: bool, sc: bool) -> str:
    return f"fr{int(fr)}_kd{int(kd)}_sc{int(sc)}"


def run_name(fr: bool, kd: bool, sc: bool) -> str:
    return f"cifar100_5agent20cls_dkp_ablation_{variant_id(fr, kd, sc)}_ipc10_seed0_v1"


def config_filename(fr: bool, kd: bool, sc: bool) -> str:
    return f"{run_name(fr, kd, sc)}.yaml"


def _build_variant(base: dict, fr: bool, kd: bool, sc: bool) -> dict:
    _require((fr, kd, sc) in MISSING_COMBINATIONS, "only the six missing loss combinations may be generated")
    config = copy.deepcopy(base)
    name = run_name(fr, kd, sc)
    config["project"].update(
        {
            "run_name": name,
            "stage": "iclr2027_dkp_loss_ablation_diagnostic",
            "paper_eligible": False,
            "protocol_status": "planned_diagnostic",
        }
    )

    receiver = config["social_learning"]["receiver"]
    receiver.update(
        {
            "dkp_variant": f"ablation_{variant_id(fr, kd, sc)}",
            "loss_switches": {"fr": bool(fr), "kd": bool(kd), "supcon": bool(sc)},
            "lambda_fr": FIXED_HYPERPARAMETERS["lambda_fr"] if fr else 0.0,
            "lambda_sc": FIXED_HYPERPARAMETERS["lambda_sc"] if sc else 0.0,
            "supcon_temperature": FIXED_HYPERPARAMETERS["supcon_temperature"],
            "checkpoint_retention": "final_only",
        }
    )
    config["communication"]["use_sender_logits"] = bool(kd)
    config["logits"].update(
        {
            "enabled": bool(kd),
            "lambda_kd": FIXED_HYPERPARAMETERS["lambda_kd"] if kd else 0.0,
            "temperature": FIXED_HYPERPARAMETERS["kd_temperature"],
        }
    )
    config["loss_ablation"] = {
        "role": "fixed_seed0_diagnostic_not_hyperparameter_search",
        "variant_id": variant_id(fr, kd, sc),
        "switches": {"fr": bool(fr), "kd": bool(kd), "supcon": bool(sc)},
        "fixed_hyperparameters": dict(FIXED_HYPERPARAMETERS),
        "existing_endpoints": {
            "fr0_kd0_sc0": "cifar100_5agent20cls_dkp_ce_only_ipc10_seed0_v2",
            "fr1_kd1_sc1": "cifar100_5agent20cls_dkp_sl_full_ipc10_seed0_v2",
        },
    }
    config["expert_reuse"] = {
        "source_run": EXPERT_RUN,
        "require_checkpoint_sha_match": True,
    }
    return config


def build_variants(base_config: Path | str = BASE_CONFIG) -> dict[str, dict]:
    base_path = Path(base_config)
    base = _load(base_path)
    variants = {}
    for switches in MISSING_COMBINATIONS:
        fr, kd, sc = switches
        config = _build_variant(base, fr, kd, sc)
        validate_variant(config, switches, base)
        variants[config_filename(fr, kd, sc)] = config
    return variants


def validate_variant(config: dict, switches: tuple[bool, bool, bool], base: dict | None = None) -> None:
    fr, kd, sc = switches
    expected_id = variant_id(fr, kd, sc)
    expected_name = run_name(fr, kd, sc)
    project = config.get("project", {})
    receiver = config.get("social_learning", {}).get("receiver", {})
    communication = config.get("communication", {})
    logits = config.get("logits", {})
    ablation = config.get("loss_ablation", {})

    _require(switches in MISSING_COMBINATIONS, f"{expected_id}: not a missing combination")
    _require(project.get("run_name") == expected_name, f"{expected_id}: run_name mismatch")
    _require(project.get("paper_eligible") is False, f"{expected_id}: must be diagnostic-only")
    _require(project.get("protocol_status") == "planned_diagnostic", f"{expected_id}: status drifted")
    _require(receiver.get("dkp_variant") == f"ablation_{expected_id}", f"{expected_id}: variant mismatch")
    _require(receiver.get("loss_switches") == {"fr": fr, "kd": kd, "supcon": sc}, f"{expected_id}: receiver switches mismatch")
    _require(ablation.get("variant_id") == expected_id, f"{expected_id}: ablation id mismatch")
    _require(ablation.get("switches") == {"fr": fr, "kd": kd, "supcon": sc}, f"{expected_id}: ablation switches mismatch")
    _require(ablation.get("fixed_hyperparameters") == FIXED_HYPERPARAMETERS, f"{expected_id}: fixed hyperparameters drifted")
    _require(receiver.get("checkpoint_retention") == "final_only", f"{expected_id}: checkpoint retention is not final_only")
    _require(communication.get("use_sender_logits") is kd, f"{expected_id}: communication logits mismatch")
    _require(logits.get("enabled") is kd, f"{expected_id}: logits.enabled mismatch")
    _require(float(receiver.get("lambda_fr", -1)) == (0.2 if fr else 0.0), f"{expected_id}: lambda_FR mismatch")
    _require(float(logits.get("lambda_kd", -1)) == (0.6 if kd else 0.0), f"{expected_id}: lambda_KD mismatch")
    _require(float(receiver.get("lambda_sc", -1)) == (0.1 if sc else 0.0), f"{expected_id}: lambda_SC mismatch")
    _require(float(logits.get("temperature", -1)) == 2.0, f"{expected_id}: KD temperature mismatch")
    _require(float(receiver.get("supcon_temperature", -1)) == 0.07, f"{expected_id}: SupCon temperature mismatch")
    _require(config.get("expert_reuse", {}).get("source_run") == EXPERT_RUN, f"{expected_id}: expert source mismatch")
    _require(int(config.get("runtime", {}).get("seed", -1)) == 0, f"{expected_id}: seed is not zero")
    _require(int(config.get("agents", {}).get("num_agents", -1)) == 5, f"{expected_id}: agent count is not five")
    _require(int(config.get("distillation", {}).get("ipc", -1)) == 10, f"{expected_id}: IPC is not 10")
    _require(int(receiver.get("epochs", -1)) == 60, f"{expected_id}: epochs drifted")
    _require(int(receiver.get("local_batch_size", -1)) == 64, f"{expected_id}: local batch size drifted")
    _require(int(receiver.get("external_batch_size", -1)) == 64, f"{expected_id}: external batch size drifted")
    _require(receiver.get("scheduler_milestones") == [39, 51], f"{expected_id}: milestones drifted")
    _require(float(receiver.get("scheduler_gamma", -1)) == 0.2, f"{expected_id}: gamma drifted")

    class_split = get_agent_class_split(config)
    flat = [class_id for agent_id in range(5) for class_id in class_split[agent_id]]
    _require(all(len(class_split[agent_id]) == 20 for agent_id in range(5)), f"{expected_id}: class count drifted")
    _require(len(flat) == 100 and set(flat) == set(range(100)), f"{expected_id}: classes do not partition 0-99")
    _require(get_agent_model_split(config) == EXPECTED_MODELS, f"{expected_id}: model assignment drifted")
    for model_name in EXPECTED_MODELS.values():
        classifier = config["model_pool"]["models"][model_name].get("classifier", {})
        _require(classifier == {"type": "cosine", "scale_init": 10.0}, f"{expected_id}/{model_name}: cosine classifier drifted")

    if base is not None:
        _require(get_agent_class_split(config) == get_agent_class_split(base), f"{expected_id}: base class split differs")
        _require(get_agent_model_split(config) == get_agent_model_split(base), f"{expected_id}: base model split differs")
        _require(config.get("pool_reuse") == base.get("pool_reuse"), f"{expected_id}: pool provenance config drifted")
        _require(config.get("dataset") == base.get("dataset"), f"{expected_id}: dataset config drifted")
        _require(config.get("distillation") == base.get("distillation"), f"{expected_id}: distillation config drifted")


def write_variants(variants: dict[str, dict], target_dir: Path | str = TARGET_DIR) -> list[Path]:
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for filename, config in sorted(variants.items()):
        target = target_dir / filename
        if target.exists():
            if _load(target) != config:
                raise FileExistsError(f"refusing to replace a different loss-ablation config: {target}")
        else:
            target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        written.append(target)
    return written


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default=str(BASE_CONFIG))
    parser.add_argument("--target-dir", default=str(TARGET_DIR))
    parser.add_argument("--check", action="store_true", help="Validate expected files without writing")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    cli = parse_args(argv)
    try:
        variants = build_variants(cli.base_config)
        target_dir = Path(cli.target_dir)
        if cli.check:
            for filename, expected in sorted(variants.items()):
                path = target_dir / filename
                _require(path.is_file(), f"expected config is missing: {path}")
                _require(_load(path) == expected, f"existing config differs from generator: {path}")
            paths = [target_dir / filename for filename in sorted(variants)]
        else:
            paths = write_variants(variants, target_dir)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ICLR2027 DKP loss-ablation config generation FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "passed", "configs": [str(path) for path in paths]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
