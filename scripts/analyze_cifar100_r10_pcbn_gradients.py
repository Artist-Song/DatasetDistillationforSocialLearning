#!/usr/bin/env python3
"""Measure R10 DSDM/PCBN gradient scale and per-layer concentration."""

from __future__ import annotations

import copy
import json
import math
import random
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DSDM_ROOT = ROOT / "DSDM"
for path in (ROOT, DSDM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent_data import build_agent_args  # noqa: E402
from config_adapter import load_config  # noqa: E402
from DSDM import ClassMemDataLoader, Synthesizer, diffaug, load_resized_data, matchloss  # noqa: E402
from output_manager import atomic_write_json  # noqa: E402
from pcbn import PCBNRegularizer  # noqa: E402
from train import define_model  # noqa: E402


CONFIG = ROOT / (
    "configs/fullclass_dsdm/"
    "fullclass_resnet10_standard_modelbest_e0200_ipc10_seed0_recovery.yaml"
)
GUIDE_ROOT = ROOT / (
    "outputs/cifar100_fullclass_dsdm_resnet10_standard_modelbest_e0200_ipc10_seed0_recovery/"
    "agents/agent_0/checkpoints/dsdm_pretrained"
)
PCBN_BEST = ROOT / (
    "outputs/cifar100_fullclass_dsdm_resnet10_standard_modelbest_e0200_ipc10_seed0_recovery_pcbn_w960/"
    "agents/agent_0/synthetic/data_best.pt"
)
OUTPUT = ROOT / "experiments/diagnostics/cifar100_r10_pcbn_gradient_diagnostic_seed0.json"
GUIDES = (0, 5)
CLASSES = (0, 20, 60, 99)
WEIGHTS = (620.0, 960.0, 1300.0, 1700.0, 2100.0, 3400.0)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_best_images() -> torch.Tensor:
    payload = torch.load(PCBN_BEST, map_location="cpu", weights_only=False)
    images = payload.get("images") if isinstance(payload, dict) else payload[0]
    labels = payload.get("labels") if isinstance(payload, dict) else payload[1]
    if tuple(images.shape) != (1000, 3, 32, 32):
        raise ValueError(f"Unexpected PCBN best image shape: {tuple(images.shape)}")
    expected = torch.arange(100).repeat_interleave(10)
    if not torch.equal(labels.long().cpu(), expected):
        raise ValueError("PCBN best labels are not ordered global labels 0-99")
    if not torch.isfinite(images).all():
        raise FloatingPointError("PCBN best images are non-finite")
    return images.float()


def summarize(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This diagnostic requires CUDA")
    device = torch.device("cuda")
    cfg = load_config(CONFIG)
    args = build_agent_args(cfg, CONFIG, 0)
    if args.net_type != "resnet_cifar_standard" or args.depth != 10:
        raise ValueError("Expected standard CIFAR ResNet-10")
    if args.active_class_ids != list(range(100)) or (args.idx_from, args.idx_to) != (5, -1):
        raise ValueError("Unexpected full-class R10 protocol")

    seed_everything(int(args.seed))
    trainset, _ = load_resized_data(args)
    loader = ClassMemDataLoader(trainset, batch_size=args.batch_real, device=device)
    mix_synset = Synthesizer(args, args.nclass, args.nch, args.size, args.size, device=device)
    mix_synset.init(loader, init_type="mix")
    states = {
        "mix_initialization": mix_synset.data.detach().cpu(),
        "pcbn_w960_best_iter5000": load_best_images(),
    }
    augmentation, _ = diffaug(args, device=device)

    regularizer_args = copy.copy(args)
    regularizer_args.pcbn_enabled = True
    regularizer_args.pcbn_weight = 1.0
    regularizer_args.pcbn_layers = "all"
    regularizer_args.pcbn_normalize_layers = True

    samples = []
    layer_names = None
    for guide_index in GUIDES:
        model = define_model(args, 100).to(device).eval()
        guide_path = GUIDE_ROOT / f"cifar100_model_{guide_index}.pth"
        model.load_state_dict(torch.load(guide_path, map_location=device, weights_only=True))
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        regularizer = PCBNRegularizer(regularizer_args)
        hooks = regularizer.attach(model)
        if hooks != 12:
            raise ValueError(f"Expected 12 R10 BN hooks, found {hooks}")
        if layer_names is None:
            layer_names = list(regularizer.layer_names)
        elif layer_names != regularizer.layer_names:
            raise ValueError("BN hook order changed between guides")

        for state_name, state_images in states.items():
            for class_id in CLASSES:
                sample_seed = 1_000_000 * guide_index + 10_000 * list(states).index(state_name) + class_id
                seed_everything(sample_seed)
                real_images, _ = loader.class_sample(class_id)
                start = class_id * args.ipc
                raw_images = state_images[start : start + args.ipc].to(device)
                raw_targets = torch.full((args.ipc,), class_id, dtype=torch.long, device=device)
                syn_images, _ = mix_synset.decode(raw_images, raw_targets)
                augmented = augmentation(torch.cat([real_images, syn_images]))
                real_aug = augmented[: real_images.shape[0]].detach()
                syn_aug = augmented[real_images.shape[0] :].detach().requires_grad_(True)

                base_loss = matchloss(args, real_aug, syn_aug, model)
                real_stats = regularizer._collect_stats(model, real_aug, detach=True)
                syn_stats = regularizer._collect_stats(model, syn_aug, detach=False)
                layer_losses = [
                    F.mse_loss(syn_mean, real_mean) + F.mse_loss(syn_var, real_var)
                    for (real_mean, real_var), (syn_mean, syn_var) in zip(real_stats, syn_stats)
                ]
                raw_pcbn_loss = sum(layer_losses) / len(layer_losses)
                base_gradient = torch.autograd.grad(base_loss, syn_aug, retain_graph=True)[0]
                pcbn_gradient = torch.autograd.grad(raw_pcbn_loss, syn_aug)[0]
                base_norm = float(base_gradient.norm().item())
                raw_pcbn_norm = float(pcbn_gradient.norm().item())
                cosine = float(
                    F.cosine_similarity(base_gradient.flatten(), pcbn_gradient.flatten(), dim=0).item()
                )
                layer_values = [float(value.detach().item()) for value in layer_losses]
                layer_sum = sum(layer_values)
                max_layer_index = max(range(len(layer_values)), key=layer_values.__getitem__)
                weight_metrics = {}
                for weight in WEIGHTS:
                    weighted_ratio = weight * raw_pcbn_norm / base_norm
                    combined = base_gradient + weight * pcbn_gradient
                    weight_metrics[f"{weight:g}"] = {
                        "pcbn_to_dsdm_gradient_norm": weighted_ratio,
                        "combined_to_dsdm_gradient_norm": float(combined.norm().item()) / base_norm,
                    }
                samples.append(
                    {
                        "state": state_name,
                        "guide_index": guide_index,
                        "class_id": class_id,
                        "sample_seed": sample_seed,
                        "base_loss": float(base_loss.detach().item()),
                        "raw_normalized_pcbn_loss": float(raw_pcbn_loss.detach().item()),
                        "base_gradient_norm": base_norm,
                        "raw_pcbn_gradient_norm": raw_pcbn_norm,
                        "raw_gradient_cosine": cosine,
                        "max_layer": layer_names[max_layer_index],
                        "max_layer_fraction": layer_values[max_layer_index] / layer_sum,
                        "layer_losses": dict(zip(layer_names, layer_values)),
                        "weights": weight_metrics,
                    }
                )
                del augmented, real_aug, syn_aug, base_loss, raw_pcbn_loss
                del base_gradient, pcbn_gradient, layer_losses, real_stats, syn_stats

        regularizer.close()
        del regularizer, model
        torch.cuda.empty_cache()

    aggregate = {}
    for state_name in states:
        state_samples = [sample for sample in samples if sample["state"] == state_name]
        weight_summary = {}
        for weight in WEIGHTS:
            key = f"{weight:g}"
            ratios = [sample["weights"][key]["pcbn_to_dsdm_gradient_norm"] for sample in state_samples]
            combined = [sample["weights"][key]["combined_to_dsdm_gradient_norm"] for sample in state_samples]
            weight_summary[key] = {
                "pcbn_to_dsdm_gradient_norm": summarize(ratios),
                "combined_to_dsdm_gradient_norm": summarize(combined),
            }
        aggregate[state_name] = {
            "sample_count": len(state_samples),
            "raw_gradient_cosine": summarize([sample["raw_gradient_cosine"] for sample in state_samples]),
            "max_layer_fraction": summarize([sample["max_layer_fraction"] for sample in state_samples]),
            "weights": weight_summary,
        }

    payload = {
        "status": "passed",
        "purpose": "diagnose gradient scale before expanding the static PCBN weight",
        "config": str(CONFIG.relative_to(ROOT)),
        "pcbn_best_source": str(PCBN_BEST.relative_to(ROOT)),
        "protocol": {
            "guides": list(GUIDES),
            "classes": list(CLASSES),
            "weights": list(WEIGHTS),
            "bn_layers": layer_names,
            "feature_index": int(args.idx_from),
            "base_loss_note": "semantic DSDM without the post-smooth h_p term",
        },
        "aggregate": aggregate,
        "samples": samples,
    }
    if not all(math.isfinite(value) for sample in samples for value in (
        sample["base_loss"],
        sample["raw_normalized_pcbn_loss"],
        sample["base_gradient_norm"],
        sample["raw_pcbn_gradient_norm"],
        sample["raw_gradient_cosine"],
    )):
        raise FloatingPointError("Non-finite PCBN gradient diagnostic")
    atomic_write_json(payload, OUTPUT)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    print(f"saved: {OUTPUT}")


if __name__ == "__main__":
    main()
