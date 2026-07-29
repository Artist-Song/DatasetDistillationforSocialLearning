#!/usr/bin/env python3
"""Strict gate for five matched-linear ICLR 2027 CIFAR-100 experts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
DSDM_ROOT = ROOT / "DSDM"
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))
if str(DSDM_ROOT) not in sys.path:
    sys.path.append(str(DSDM_ROOT))

from agent_data import build_agent_args, get_agent_class_split, get_agent_model_split  # noqa: E402
from agent_trainer import _agent_test_subset, _evaluate_expert_accuracy  # noqa: E402
from config_adapter import load_config  # noqa: E402
from output_manager import atomic_write_json  # noqa: E402
from scripts.validate_iclr2027_cosine_experts import (  # noqa: E402
    PreflightError,
    _extract_feature_tensors,
    _finite_float,
    _load_state_dict,
    _require,
    _resolve_from_root,
    resolve_run_dir,
    sha256_file,
)


DEFAULT_CONFIG = ROOT / "configs/iclr2027/cifar100_5agent20cls_dkp_linear_experts_seed0_v1.yaml"
EXPECTED_MODELS = {
    0: {
        "name": "convnet3w1",
        "family": "convnet",
        "depth": 3,
        "width": 1.0,
        "feature_index": 2,
        "classifier_in_features": 2048,
        "parameters": 504_420,
        "expert_epochs": 500,
    },
    1: {
        "name": "convnet4w15",
        "family": "convnet",
        "depth": 4,
        "width": 1.5,
        "feature_index": 3,
        "classifier_in_features": 768,
        "parameters": 1_079_716,
        "expert_epochs": 500,
    },
    2: {
        "name": "alexnet",
        "family": "alexnet",
        "depth": 3,
        "width": 1.0,
        "feature_index": 7,
        "classifier_in_features": 512,
        "parameters": 7_025_316,
        "expert_epochs": 500,
    },
    3: {
        "name": "resnet10_standard",
        "family": "resnet_cifar_standard",
        "depth": 10,
        "width": 1.0,
        "feature_index": 5,
        "classifier_in_features": 512,
        "parameters": 4_949_412,
        "expert_epochs": 200,
        "blocks_per_stage": 1,
    },
    4: {
        "name": "resnet18_standard",
        "family": "resnet_cifar_standard",
        "depth": 18,
        "width": 1.0,
        "feature_index": 5,
        "classifier_in_features": 512,
        "parameters": 11_220_132,
        "expert_epochs": 200,
        "blocks_per_stage": 2,
    },
}


def validate_protocol_config(config, config_path):
    """Validate the fixed seed0 partition, backbones, recipes, and linear heads."""
    dataset = config.get("dataset", {})
    agents = config.get("agents", {})
    distillation = config.get("distillation", {})
    project = config.get("project", {})
    _require(dataset.get("name") == "cifar100", "linear expert config must use CIFAR-100")
    _require(int(dataset.get("num_classes", -1)) == 100, "linear classifier universe must be 100")
    _require(int(dataset.get("class_assignment_seed", -1)) == 0, "class assignment must use seed0")
    _require(int(agents.get("num_agents", -1)) == 5, "linear protocol requires five agents")
    _require(int(distillation.get("ipc", -1)) == 10, "linear diagnostic must use IPC=10")
    _require(project.get("paper_eligible") is False, "linear diagnostic must be paper-ineligible")
    _require(
        project.get("stage") == "iclr2027_linear_expert_training",
        "linear expert stage marker is missing",
    )

    class_split = get_agent_class_split(config)
    model_split = get_agent_model_split(config)
    _require(sorted(class_split) == list(EXPECTED_MODELS), "linear class split has incomplete agents")
    flattened = []
    permutation = np.random.RandomState(0).permutation(100).tolist()
    args_by_agent = {}
    for agent_id, expected in EXPECTED_MODELS.items():
        class_ids = [int(value) for value in class_split[agent_id]]
        expected_ids = permutation[agent_id * 20:(agent_id + 1) * 20]
        _require(class_ids == expected_ids, f"agent {agent_id} does not match seed0 class order")
        flattened.extend(class_ids)
        _require(model_split[agent_id] == expected["name"], f"agent {agent_id} model assignment drifted")
        args = build_agent_args(config, config_path, agent_id)
        _require(args.classifier_type == "linear", f"agent {agent_id} classifier is not linear")
        _require(bool(args.expert_mask_nonlocal_classes), f"agent {agent_id} masked CE is disabled")
        _require(bool(args.separate_expert), f"agent {agent_id} expert is not separate")
        _require(args.net_type == expected["family"], f"agent {agent_id} family drifted")
        _require(int(args.depth) == expected["depth"], f"agent {agent_id} depth drifted")
        _require(math.isclose(float(args.width), expected["width"]), f"agent {agent_id} width drifted")
        _require(
            (int(args.idx_from), int(args.idx_to)) == (expected["feature_index"], -1),
            f"agent {agent_id} feature index drifted",
        )
        _require(int(args.expert_epochs) == expected["expert_epochs"], f"agent {agent_id} epoch horizon drifted")
        args_by_agent[agent_id] = args
    _require(len(flattened) == len(set(flattened)), "linear expert class sets overlap")
    _require(set(flattened) == set(range(100)), "linear expert classes do not cover 0-99")
    return args_by_agent


def _build_model(args):
    from train import define_model

    return define_model(args, 100)


def _validate_model_definition(model, args, agent_id, classifier):
    expected = EXPECTED_MODELS[int(agent_id)]
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    _require(parameter_count == expected["parameters"], f"agent {agent_id} parameter count drifted")
    _require(isinstance(classifier, nn.Linear), f"agent {agent_id} output head is not nn.Linear")
    _require(int(classifier.in_features) == expected["classifier_in_features"], "linear feature width drifted")
    _require(int(classifier.out_features) == 100, "linear head does not output 100 classes")
    _require(classifier.bias is not None and tuple(classifier.bias.shape) == (100,), "linear bias is missing")
    if int(agent_id) in {3, 4}:
        _require(args.net_type == "resnet_cifar_standard", f"agent {agent_id} is not standard ResNet")
        _require(getattr(model, "cifar_base_width", None) == 64, f"agent {agent_id} ResNet is not width 64")
        lengths = [len(getattr(model, f"layer{stage}")) for stage in range(1, 5)]
        _require(lengths == [expected["blocks_per_stage"]] * 4, f"agent {agent_id} stage layout drifted")


def validate_expert_artifact(
    args,
    agent_id,
    run_dir,
    min_local_test_accuracy=20.0,
    device="cpu",
    model_builder=None,
    verify_dataset_accuracy=False,
):
    """Strictly load one expert and verify manifest, geometry, SHA, and accuracy."""
    expected = EXPECTED_MODELS[int(agent_id)]
    checkpoint_dir = Path(run_dir) / "agents" / f"agent_{int(agent_id)}" / "checkpoints"
    checkpoint_path = checkpoint_dir / "expert_model.pt"
    manifest_path = checkpoint_dir / "expert_manifest.json"
    _require(checkpoint_path.is_file(), f"agent {agent_id} checkpoint is missing")
    _require(manifest_path.is_file(), f"agent {agent_id} manifest is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"agent {agent_id} manifest cannot be read") from exc
    _require(isinstance(manifest, dict), f"agent {agent_id} manifest is not an object")
    _require(int(manifest.get("agent_id", -1)) == int(agent_id), f"agent {agent_id} manifest id mismatch")
    _require(
        manifest.get("role") == "fully_converged_agent_expert_and_logit_teacher",
        f"agent {agent_id} manifest role mismatch",
    )
    _require(manifest.get("test_used_for_selection") is False, f"agent {agent_id} used test selection")
    _require(manifest.get("retrained_on_full_local_train") is True, f"agent {agent_id} was not full retrained")
    _require(manifest.get("masked_local_ce") is True, f"agent {agent_id} manifest masked CE is false")
    _require(manifest.get("labels") == "global", f"agent {agent_id} labels are not global")
    _require(int(manifest.get("global_output_dim", -1)) == 100, f"agent {agent_id} output dim is not 100")
    _require(
        [int(value) for value in manifest.get("active_class_ids", [])] == list(args.active_class_ids),
        f"agent {agent_id} active classes differ",
    )
    _require(int(manifest.get("max_epochs", -1)) == expected["expert_epochs"], "expert epoch horizon changed")
    selected_epoch = int(manifest.get("selected_epoch", -1))
    _require(0 < selected_epoch <= expected["expert_epochs"], f"agent {agent_id} selected epoch is invalid")
    validation_accuracy = _finite_float(
        manifest.get("best_validation_accuracy"), f"agent {agent_id} validation accuracy"
    )
    local_accuracy = _finite_float(
        manifest.get("official_test_accuracy_report_only"), f"agent {agent_id} local test accuracy"
    )
    threshold = float(min_local_test_accuracy)
    _require(0.0 <= validation_accuracy <= 100.0, "validation accuracy is outside [0,100]")
    _require(local_accuracy > threshold, f"agent {agent_id} local accuracy is not above {threshold}")

    classifier_manifest = manifest.get("classifier", {})
    _require(classifier_manifest.get("type") == "linear", f"agent {agent_id} manifest head is not linear")
    _require(classifier_manifest.get("bias") is True, f"agent {agent_id} linear bias is disabled")
    _require(classifier_manifest.get("feature_normalization") is False, "linear features were normalized")
    _require(classifier_manifest.get("weight_normalization") is False, "linear weights were normalized")
    _require(classifier_manifest.get("scale_parameterization") is None, "linear head has scale parameterization")
    _require(classifier_manifest.get("final_scale") is None, "linear head unexpectedly has a final scale")

    actual_sha = sha256_file(checkpoint_path)
    _require(actual_sha == str(manifest.get("expert_sha256", "")).lower(), "checkpoint SHA-256 mismatch")
    recorded_path = _resolve_from_root(manifest.get("expert_path", ""))
    _require(recorded_path == checkpoint_path.resolve(), f"agent {agent_id} expert path provenance differs")

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA requested for linear expert preflight but unavailable")
    model = (model_builder or _build_model)(args)
    state = _load_state_dict(checkpoint_path)
    try:
        incompatible = model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise PreflightError(f"agent {agent_id} checkpoint failed strict loading: {exc}") from exc
    _require(not incompatible.missing_keys and not incompatible.unexpected_keys, "strict state load was incomplete")
    from models.cosine_classifier import get_output_classifier, get_output_classifier_type

    classifier = get_output_classifier(model)
    _require(get_output_classifier_type(model) == "linear", f"agent {agent_id} runtime head is not linear")
    _validate_model_definition(model, args, agent_id, classifier)
    _require(bool(torch.isfinite(classifier.weight).all()), "linear weights are non-finite")
    _require(bool(torch.isfinite(classifier.bias).all()), "linear biases are non-finite")
    local_index = torch.tensor(args.active_class_ids, dtype=torch.long)
    local_norms = classifier.weight.detach().index_select(0, local_index).norm(p=2, dim=1)
    alpha = float(local_norms.mean())
    beta = float(classifier.bias.detach().index_select(0, local_index).mean())
    _require(math.isfinite(alpha) and alpha > 0.0, "linear prototype alpha would be invalid")
    _require(math.isfinite(beta), "linear prototype beta would be invalid")

    model = model.to(resolved_device).eval()
    generator = torch.Generator(device="cpu").manual_seed(20_000 + int(agent_id))
    images = torch.randn(2, int(args.nch), int(args.size), int(args.size), generator=generator).to(resolved_device)
    with torch.no_grad():
        logits = model(images)
        feature_result = model.get_feature(images, int(args.idx_from), int(args.idx_to))
    _require(tuple(logits.shape) == (2, 100), f"agent {agent_id} runtime output shape is invalid")
    _require(bool(torch.isfinite(logits).all()), f"agent {agent_id} runtime logits are non-finite")
    feature_tensors = _extract_feature_tensors(feature_result)
    _require(all(bool(torch.isfinite(value).all()) for value in feature_tensors), "features are non-finite")

    recomputed_accuracy = None
    if verify_dataset_accuracy:
        loader = DataLoader(
            _agent_test_subset(args, agent_id),
            batch_size=128,
            shuffle=False,
            num_workers=0,
        )
        recomputed_accuracy = _evaluate_expert_accuracy(
            model,
            loader,
            resolved_device,
            active_class_ids=args.active_class_ids,
        )
        _require(recomputed_accuracy > threshold, f"agent {agent_id} recomputed accuracy is unreasonable")
        _require(
            math.isclose(recomputed_accuracy, local_accuracy, abs_tol=1e-6, rel_tol=0.0),
            f"agent {agent_id} recomputed accuracy differs from manifest",
        )

    return {
        "agent_id": int(agent_id),
        "model": expected["name"],
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": actual_sha,
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "parameters": expected["parameters"],
        "output_shape": [2, 100],
        "feature_shapes": [list(value.shape) for value in feature_tensors],
        "selected_epoch": selected_epoch,
        "best_validation_accuracy": validation_accuracy,
        "local_test_accuracy": local_accuracy,
        "recomputed_local_test_accuracy": recomputed_accuracy,
        "prototype_alpha": alpha,
        "prototype_beta": beta,
    }


def run_preflight(
    config_path=DEFAULT_CONFIG,
    run_dir=None,
    min_local_test_accuracy=20.0,
    device="cpu",
    model_builder=None,
    verify_dataset_accuracy=True,
):
    config_path = _resolve_from_root(config_path)
    _require(config_path.is_file(), f"linear expert config is missing: {config_path}")
    threshold = _finite_float(min_local_test_accuracy, "minimum local test accuracy")
    _require(0.0 <= threshold < 100.0, "minimum local accuracy must be in [0,100)")
    config = load_config(config_path)
    args_by_agent = validate_protocol_config(config, config_path)
    resolved_run_dir = resolve_run_dir(config, override=run_dir)
    reports = [
        validate_expert_artifact(
            args_by_agent[agent_id],
            agent_id,
            resolved_run_dir,
            min_local_test_accuracy=threshold,
            device=device,
            model_builder=model_builder,
            verify_dataset_accuracy=verify_dataset_accuracy,
        )
        for agent_id in EXPECTED_MODELS
    ]
    return {
        "status": "passed",
        "protocol": "iclr2027_cifar100_5agent20cls_matched_linear_experts_v1",
        "config": str(config_path),
        "run_dir": str(resolved_run_dir),
        "minimum_local_test_accuracy_exclusive": threshold,
        "dataset_accuracy_recomputed": bool(verify_dataset_accuracy),
        "agents": reports,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-dir")
    parser.add_argument("--min-local-test-accuracy", type=float, default=20.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--skip-dataset-accuracy", action="store_true")
    parser.add_argument("--report-json")
    return parser.parse_args(argv)


def main(argv=None):
    cli = parse_args(argv)
    try:
        report = run_preflight(
            config_path=cli.config,
            run_dir=cli.run_dir,
            min_local_test_accuracy=cli.min_local_test_accuracy,
            device=cli.device,
            verify_dataset_accuracy=not cli.skip_dataset_accuracy,
        )
        if cli.report_json:
            report_path = _resolve_from_root(cli.report_json)
            if report_path.exists():
                raise FileExistsError(f"refusing to overwrite linear preflight report: {report_path}")
            atomic_write_json(report, report_path)
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ICLR2027 linear expert preflight FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
