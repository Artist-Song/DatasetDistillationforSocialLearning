#!/usr/bin/env python3
"""Strict preflight gate for the five ICLR 2027 CIFAR-100 cosine experts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DSDM_ROOT = ROOT / "DSDM"
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))
if str(DSDM_ROOT) not in sys.path:
    sys.path.append(str(DSDM_ROOT))

from agent_data import build_agent_args, get_agent_class_split, get_agent_model_split  # noqa: E402
from config_adapter import load_config  # noqa: E402
from output_manager import atomic_write_json  # noqa: E402
from packet_integrity import resolve_strict_dkp_contract  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/iclr2027/cifar100_5agent20cls_dkp_cosine_experts_seed0_v1.yaml"
MODEL_SPECS = {
    "convnet3w1": {
        "family": "convnet",
        "depth": 3,
        "width": 1.0,
        "feature_index": 2,
        "classifier_in_features": 2048,
        "parameters": 504_321,
        "expert_epochs": 500,
    },
    "convnet4w15": {
        "family": "convnet",
        "depth": 4,
        "width": 1.5,
        "feature_index": 3,
        "classifier_in_features": 768,
        "parameters": 1_079_617,
        "expert_epochs": 500,
    },
    "alexnet": {
        "family": "alexnet",
        "depth": 3,
        "width": 1.0,
        "feature_index": 7,
        "classifier_in_features": 512,
        "parameters": 7_025_217,
        "expert_epochs": 500,
    },
    "resnet10_standard": {
        "family": "resnet_cifar_standard",
        "depth": 10,
        "width": 1.0,
        "feature_index": 5,
        "classifier_in_features": 512,
        "parameters": 4_949_313,
        "expert_epochs": 200,
        "blocks_per_stage": 1,
    },
    "resnet18_standard": {
        "family": "resnet_cifar_standard",
        "depth": 18,
        "width": 1.0,
        "feature_index": 5,
        "classifier_in_features": 512,
        "parameters": 11_220_033,
        "expert_epochs": 200,
        "blocks_per_stage": 2,
    },
}
BACKBONES = tuple(MODEL_SPECS)
EXPECTED_MODELS = {
    agent_id: {"name": model_name, **MODEL_SPECS[model_name]}
    for agent_id, model_name in enumerate(BACKBONES)
}


class PreflightError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise PreflightError(message)


def _finite_float(value, field):
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise PreflightError(f"{field} is not numeric: {value!r}") from exc
    _require(math.isfinite(resolved), f"{field} is not finite: {resolved}")
    return resolved


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_from_root(path):
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved.resolve()


def resolve_run_dir(cfg, override=None):
    if override is not None:
        return _resolve_from_root(override)
    project = cfg.get("project", {})
    output_root = _resolve_from_root(project.get("output_root", "./outputs"))
    run_name = str(project.get("run_name", "")).strip()
    _require(run_name, "project.run_name is missing")
    return (output_root / run_name).resolve()


def validate_protocol_config(cfg, config_path):
    dataset = cfg.get("dataset", {})
    agents_cfg = cfg.get("agents", {})
    distillation = cfg.get("distillation", {})
    _require(dataset.get("name") == "cifar100", "expert config must use CIFAR-100")
    _require(int(dataset.get("num_classes", -1)) == 100, "classifier universe must contain 100 classes")
    seed = int(dataset.get("class_assignment_seed", -1))
    _require(seed >= 0, "class assignment seed must be explicit and non-negative")
    _require(int(cfg.get("runtime", {}).get("seed", -1)) == seed, "runtime/class assignment seeds differ")
    _require(int(agents_cfg.get("num_classes", -1)) == 100, "agent config must preserve 100 global classes")
    _require(int(distillation.get("ipc", -1)) == 10, "scaling protocol must use IPC=10")

    try:
        contract = resolve_strict_dkp_contract(cfg)
    except (TypeError, ValueError) as error:
        raise PreflightError(f"strict DKP partition/overlap contract is invalid: {error}") from error
    agent_count = contract.agent_count
    classes_per_agent = contract.classes_per_agent
    _require(agent_count in {5, 10, 20}, f"unsupported agent count: {agent_count}")
    _require(int(agents_cfg.get("num_agents", -1)) == agent_count, "agents.num_agents drifted")

    class_split = get_agent_class_split(cfg)
    model_split = get_agent_model_split(cfg)
    expected_ids = list(range(agent_count))
    _require(sorted(class_split) == expected_ids, f"unexpected agent ids: {sorted(class_split)}")
    _require(sorted(model_split) == expected_ids, f"incomplete model split: {sorted(model_split)}")

    children_per_macro = agent_count // len(BACKBONES)
    permutation = np.random.RandomState(seed).permutation(100).tolist()
    expected_classes = {}
    expected_models = {}
    agent_id = 0
    for macro_id, model_name in enumerate(BACKBONES):
        macro = permutation[macro_id * 20 : (macro_id + 1) * 20]
        for child_id in range(children_per_macro):
            start = child_id * classes_per_agent
            expected_classes[agent_id] = macro[start : start + classes_per_agent]
            expected_models[agent_id] = model_name
            agent_id += 1

    args_by_agent = {}
    for agent_id in expected_ids:
        expected_name = expected_models[agent_id]
        expected = MODEL_SPECS[expected_name]
        _require(
            class_split[agent_id] == expected_classes[agent_id],
            f"agent {agent_id} does not match seed{seed} nested class order",
        )
        _require(
            model_split[agent_id] == expected_name,
            f"agent {agent_id} model mismatch: {model_split[agent_id]} != {expected_name}",
        )
        args = build_agent_args(cfg, config_path, agent_id)
        _require(args.model_name == expected_name, f"agent {agent_id} resolved the wrong model name")
        _require(args.net_type == expected["family"], f"agent {agent_id} resolved the wrong model family")
        _require(int(args.depth) == expected["depth"], f"agent {agent_id} resolved the wrong depth")
        _require(math.isclose(float(args.width), expected["width"]), f"agent {agent_id} resolved the wrong width")
        _require(
            (int(args.idx_from), int(args.idx_to)) == (expected["feature_index"], -1),
            f"agent {agent_id} get_feature index is not the validated penultimate layer",
        )
        _require(args.classifier_type == "cosine", f"agent {agent_id} classifier is not cosine")
        _require(math.isclose(float(args.cosine_scale_init), 10.0), f"agent {agent_id} scale init is not 10")
        _require(bool(args.expert_mask_nonlocal_classes), f"agent {agent_id} local CE mask is disabled")
        _require(bool(args.separate_expert), f"agent {agent_id} expert is not separated from the guide pool")
        _require(
            len(args.active_class_ids) == classes_per_agent,
            f"agent {agent_id} did not resolve {classes_per_agent} active classes",
        )
        _require(int(args.expert_epochs) == expected["expert_epochs"], f"agent {agent_id} expert epoch horizon changed")
        args_by_agent[agent_id] = args
    return args_by_agent


def _build_model(args):
    from train import define_model

    return define_model(args, 100)


def _load_state_dict(path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    _require(isinstance(state, dict) and state, f"checkpoint is not a non-empty state_dict: {path}")
    for key, value in state.items():
        _require(isinstance(key, str), f"checkpoint contains a non-string key: {key!r}")
        _require(torch.is_tensor(value), f"checkpoint entry is not a tensor: {key}")
        if value.is_floating_point() or value.is_complex():
            _require(bool(torch.isfinite(value).all()), f"checkpoint tensor is non-finite: {key}")
    return state


def _extract_feature_tensors(result):
    feature_result = result[0] if isinstance(result, tuple) else result
    if torch.is_tensor(feature_result):
        return [feature_result]
    _require(isinstance(feature_result, (list, tuple)), "get_feature returned an unsupported object")
    tensors = list(feature_result)
    _require(tensors, "get_feature returned no tensors")
    _require(all(torch.is_tensor(value) for value in tensors), "get_feature returned a non-tensor value")
    return tensors


def _validate_model_definition(model, args, agent_id, classifier):
    model_name = str(args.model_name)
    _require(model_name in MODEL_SPECS, f"agent {agent_id} has unsupported model {model_name}")
    expected = MODEL_SPECS[model_name]
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    _require(
        parameter_count == expected["parameters"],
        f"agent {agent_id} parameter count mismatch: {parameter_count} != {expected['parameters']}",
    )
    _require(
        int(classifier.in_features) == expected["classifier_in_features"],
        f"agent {agent_id} classifier feature dimension changed",
    )
    _require(int(classifier.out_features) == 100, f"agent {agent_id} classifier does not output 100 classes")
    if model_name in {"resnet10_standard", "resnet18_standard"}:
        _require(args.net_type == "resnet_cifar_standard", f"agent {agent_id} is not standard CIFAR ResNet")
        _require(getattr(model, "cifar_base_width", None) == 64, f"agent {agent_id} is not width-64 standard ResNet")
        stage_lengths = [len(getattr(model, f"layer{stage}")) for stage in range(1, 5)]
        expected_lengths = [expected["blocks_per_stage"]] * 4
        _require(stage_lengths == expected_lengths, f"agent {agent_id} ResNet stage layout changed: {stage_lengths}")


def validate_expert_artifact(
    args,
    agent_id,
    run_dir,
    min_local_test_accuracy=20.0,
    device="cpu",
    model_builder=None,
):
    model_name = str(args.model_name)
    _require(model_name in MODEL_SPECS, f"agent {agent_id} has unsupported model {model_name}")
    expected = MODEL_SPECS[model_name]
    checkpoint_dir = Path(run_dir) / "agents" / f"agent_{int(agent_id)}" / "checkpoints"
    checkpoint_path = checkpoint_dir / "expert_model.pt"
    manifest_path = checkpoint_dir / "expert_manifest.json"
    _require(checkpoint_path.is_file(), f"agent {agent_id} checkpoint is missing: {checkpoint_path}")
    _require(manifest_path.is_file(), f"agent {agent_id} manifest is missing: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"agent {agent_id} manifest cannot be read: {manifest_path}") from exc
    _require(isinstance(manifest, dict), f"agent {agent_id} manifest is not an object")
    _require(int(manifest.get("agent_id", -1)) == int(agent_id), f"agent {agent_id} manifest id mismatch")
    _require(
        manifest.get("role") == "fully_converged_agent_expert_and_logit_teacher",
        f"agent {agent_id} manifest role is not the final expert/logit teacher",
    )
    _require(manifest.get("test_used_for_selection") is False, f"agent {agent_id} used test data for selection")
    _require(manifest.get("retrained_on_full_local_train") is True, f"agent {agent_id} was not retrained on full local data")
    _require(int(manifest.get("global_output_dim", -1)) == 100, f"agent {agent_id} manifest output dimension is not 100")
    _require(manifest.get("labels") == "global", f"agent {agent_id} manifest does not preserve global labels")
    _require(manifest.get("masked_local_ce") is True, f"agent {agent_id} manifest local CE mask is disabled")
    _require(
        [int(value) for value in manifest.get("active_class_ids", [])]
        == [int(value) for value in args.active_class_ids],
        f"agent {agent_id} manifest active classes differ from config",
    )
    _require(int(manifest.get("max_epochs", -1)) == expected["expert_epochs"], f"agent {agent_id} max epochs changed")
    selected_epoch = int(manifest.get("selected_epoch", -1))
    _require(0 < selected_epoch <= expected["expert_epochs"], f"agent {agent_id} selected epoch is invalid")
    best_validation = _finite_float(manifest.get("best_validation_accuracy"), f"agent {agent_id} validation accuracy")
    _require(0.0 <= best_validation <= 100.0, f"agent {agent_id} validation accuracy is outside [0,100]")
    local_test_accuracy = _finite_float(
        manifest.get("official_test_accuracy_report_only"),
        f"agent {agent_id} local test accuracy",
    )
    _require(0.0 <= local_test_accuracy <= 100.0, f"agent {agent_id} local test accuracy is outside [0,100]")
    _require(
        local_test_accuracy > float(min_local_test_accuracy),
        f"agent {agent_id} local test accuracy {local_test_accuracy:.2f}% is not above "
        f"the {float(min_local_test_accuracy):.2f}% preflight floor",
    )

    classifier_manifest = manifest.get("classifier")
    _require(isinstance(classifier_manifest, dict), f"agent {agent_id} classifier provenance is missing")
    _require(classifier_manifest.get("type") == "cosine", f"agent {agent_id} manifest classifier is not cosine")
    _require(classifier_manifest.get("bias") is False, f"agent {agent_id} cosine classifier has bias")
    _require(classifier_manifest.get("feature_normalization") is True, f"agent {agent_id} feature normalization is disabled")
    _require(classifier_manifest.get("weight_normalization") is True, f"agent {agent_id} weight normalization is disabled")
    _require(
        classifier_manifest.get("scale_parameterization") == "softplus",
        f"agent {agent_id} scale is not softplus-parameterized",
    )
    _require(
        math.isclose(_finite_float(classifier_manifest.get("scale_init"), f"agent {agent_id} scale init"), 10.0),
        f"agent {agent_id} scale init is not 10",
    )
    _require(
        _finite_float(classifier_manifest.get("scale_weight_decay"), f"agent {agent_id} scale weight decay") == 0.0,
        f"agent {agent_id} scale is subject to weight decay",
    )
    manifest_final_scale = _finite_float(
        classifier_manifest.get("final_scale"), f"agent {agent_id} final scale"
    )
    _require(manifest_final_scale > 0.0, f"agent {agent_id} manifest final scale is not positive")

    expected_path = checkpoint_path.resolve()
    recorded_expert_path = manifest.get("expert_path")
    _require(
        isinstance(recorded_expert_path, str) and recorded_expert_path.strip(),
        f"agent {agent_id} manifest expert_path is missing",
    )
    recorded_path = _resolve_from_root(recorded_expert_path)
    _require(recorded_path == expected_path, f"agent {agent_id} manifest points to another checkpoint: {recorded_path}")
    actual_sha256 = sha256_file(checkpoint_path)
    recorded_sha256 = str(manifest.get("expert_sha256", "")).lower()
    _require(len(recorded_sha256) == 64, f"agent {agent_id} manifest SHA-256 is missing or malformed")
    _require(actual_sha256 == recorded_sha256, f"agent {agent_id} checkpoint SHA-256 does not match manifest")

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA was requested for preflight but is unavailable")
    builder = model_builder or _build_model
    model = builder(args)
    state = _load_state_dict(checkpoint_path)
    try:
        incompatible = model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise PreflightError(f"agent {agent_id} checkpoint failed strict model loading: {exc}") from exc
    _require(not incompatible.missing_keys and not incompatible.unexpected_keys, f"agent {agent_id} strict load was incomplete")

    from models.cosine_classifier import get_cosine_classifier

    try:
        classifier = get_cosine_classifier(model)
    except ValueError as exc:
        raise PreflightError(f"agent {agent_id} does not contain exactly one cosine classifier") from exc
    _validate_model_definition(model, args, int(agent_id), classifier)
    _require(classifier.bias is None, f"agent {agent_id} runtime cosine classifier has bias")
    _require(bool(torch.isfinite(classifier.weight).all()), f"agent {agent_id} classifier weight is non-finite")
    _require(bool((classifier.weight.norm(p=2, dim=1) > 0).all()), f"agent {agent_id} classifier contains zero rows")
    runtime_scale = float(classifier.scale.detach().cpu())
    _require(math.isfinite(runtime_scale) and runtime_scale > 0.0, f"agent {agent_id} runtime scale is invalid")
    _require(
        math.isclose(runtime_scale, manifest_final_scale, rel_tol=1e-6, abs_tol=1e-6),
        f"agent {agent_id} runtime scale differs from manifest",
    )

    model = model.to(resolved_device).eval()
    generator = torch.Generator(device="cpu").manual_seed(10_000 + int(agent_id))
    images = torch.randn(2, int(args.nch), int(args.size), int(args.size), generator=generator).to(resolved_device)
    with torch.no_grad():
        logits = model(images)
        feature_result = model.get_feature(images, int(args.idx_from), int(args.idx_to))
    _require(tuple(logits.shape) == (2, 100), f"agent {agent_id} runtime output shape is {tuple(logits.shape)}")
    _require(bool(torch.isfinite(logits).all()), f"agent {agent_id} runtime logits are non-finite")
    feature_tensors = _extract_feature_tensors(feature_result)
    for index, feature in enumerate(feature_tensors):
        _require(feature.shape[0] == 2, f"agent {agent_id} feature {index} has the wrong batch dimension")
        _require(bool(torch.isfinite(feature).all()), f"agent {agent_id} feature {index} is non-finite")

    return {
        "agent_id": int(agent_id),
        "model": model_name,
        "active_class_ids": [int(value) for value in args.active_class_ids],
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": actual_sha256,
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "parameters": expected["parameters"],
        "output_shape": [2, 100],
        "feature_shapes": [list(value.shape) for value in feature_tensors],
        "selected_epoch": selected_epoch,
        "best_validation_accuracy": best_validation,
        "local_test_accuracy": local_test_accuracy,
        "cosine_scale": runtime_scale,
    }


def run_preflight(
    config_path=DEFAULT_CONFIG,
    run_dir=None,
    min_local_test_accuracy=20.0,
    device="cpu",
    model_builder=None,
):
    config_path = _resolve_from_root(config_path)
    _require(config_path.is_file(), f"expert config is missing: {config_path}")
    threshold = _finite_float(min_local_test_accuracy, "minimum local test accuracy")
    _require(0.0 <= threshold < 100.0, "minimum local test accuracy must be in [0,100)")
    cfg = load_config(config_path)
    args_by_agent = validate_protocol_config(cfg, config_path)
    resolved_run_dir = resolve_run_dir(cfg, override=run_dir)
    agent_reports = [
        validate_expert_artifact(
            args_by_agent[agent_id],
            agent_id,
            resolved_run_dir,
            min_local_test_accuracy=threshold,
            device=device,
            model_builder=model_builder,
        )
        for agent_id in sorted(args_by_agent)
    ]
    return {
        "status": "passed",
        "protocol": str(cfg.get("communication", {}).get("protocol", "")),
        "config": str(config_path),
        "run_dir": str(resolved_run_dir),
        "minimum_local_test_accuracy_exclusive": threshold,
        "agents": agent_reports,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-dir", help="Override project.output_root/project.run_name for read-only validation")
    parser.add_argument("--min-local-test-accuracy", type=float, default=20.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--report-json", help="Optionally atomically write the successful preflight report")
    return parser.parse_args(argv)


def main(argv=None):
    cli = parse_args(argv)
    try:
        report = run_preflight(
            config_path=cli.config,
            run_dir=cli.run_dir,
            min_local_test_accuracy=cli.min_local_test_accuracy,
            device=cli.device,
        )
        if cli.report_json:
            report_path = _resolve_from_root(cli.report_json)
            atomic_write_json(report, report_path)
            report["report_json"] = str(report_path)
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    except (PreflightError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ICLR2027 cosine expert preflight FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
