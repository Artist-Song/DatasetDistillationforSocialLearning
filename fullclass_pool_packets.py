"""Materialize sender packets by slicing validated backbone-specific full-class pools."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import sys

import torch
import yaml

from agent_data import get_agent_class_split, get_agent_dir, get_agent_model_split, get_num_classes
from output_manager import _validate_packet_payload, atomic_torch_save, atomic_write_json
from packet_integrity import (
    POOL_PROTOCOL as STRICT_POOL_PROTOCOL,
    decoded_digests,
    file_sha256,
    strict_packet_validation_enabled,
    validate_strict_partition,
)


ROOT = Path(__file__).resolve().parent
POOL_PROTOCOL = "backbone_specific_fullclass_pool_slice_v1"


def _resolve_path(value, base=ROOT):
    path = Path(value)
    return path if path.is_absolute() else Path(base) / path


def _display_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_pool_catalog(path):
    """Load and structurally validate one reusable full-class packet catalog."""
    catalog_path = _resolve_path(path)
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Pool catalog does not exist: {catalog_path}")
    with catalog_path.open("r", encoding="utf-8") as handle:
        catalog = yaml.safe_load(handle) or {}
    required = {"dataset", "num_classes", "ipc", "factor", "decode_type", "sources"}
    missing = sorted(required.difference(catalog))
    if missing:
        raise ValueError(f"Pool catalog is missing fields: {missing}")
    if not isinstance(catalog["sources"], dict) or not catalog["sources"]:
        raise ValueError("Pool catalog sources must be a non-empty mapping")
    if bool(catalog.get("strict_validation", False)):
        if int(catalog.get("schema_version", -1)) < 2:
            raise ValueError("Strict pool catalog requires schema_version >= 2")
        if str(catalog.get("protocol")) != STRICT_POOL_PROTOCOL:
            raise ValueError(
                f"Strict pool catalog protocol must be {STRICT_POOL_PROTOCOL}, got {catalog.get('protocol')}"
            )
        required_source_fields = {
            "packet_path",
            "best_manifest",
            "source_model_id",
            "expected_packet_sha256",
            "expected_best_iteration",
            "expected_best_snapshot",
            "expected_best_snapshot_sha256",
            "expected_completed_iterations",
        }
        for model_id, source_spec in catalog["sources"].items():
            missing_source = sorted(required_source_fields.difference(source_spec))
            if missing_source:
                raise ValueError(f"Strict pool source {model_id} is missing fields: {missing_source}")
    catalog["catalog_path"] = _display_path(catalog_path)
    return catalog


def validate_pool_source(source_spec, catalog, expected_model_id=None):
    """Validate a source packet and prove that it equals its persisted best snapshot."""
    packet_path = _resolve_path(source_spec["packet_path"])
    if not packet_path.is_file():
        raise FileNotFoundError(f"Pool packet does not exist: {packet_path}")
    strict = bool(catalog.get("strict_validation", False))
    packet_sha256 = file_sha256(packet_path)
    if strict:
        if str(source_spec.get("source_model_id")) != str(expected_model_id):
            raise ValueError(
                f"Pool source model id mismatch: catalog={source_spec.get('source_model_id')} "
                f"sender={expected_model_id}"
            )
        if packet_sha256 != str(source_spec.get("expected_packet_sha256")):
            raise ValueError(f"Pool source packet SHA-256 mismatch: {packet_path}")
    packet = torch.load(packet_path, map_location="cpu", weights_only=False)
    images = packet.get("images")
    labels = packet.get("labels")
    if not torch.is_tensor(images) or not torch.is_tensor(labels):
        raise ValueError(f"Pool packet lacks image/label tensors: {packet_path}")
    if images.ndim != 4 or labels.ndim != 1 or images.shape[0] != labels.shape[0]:
        raise ValueError(f"Invalid pool tensor shapes: images={tuple(images.shape)} labels={tuple(labels.shape)}")
    if not bool(torch.isfinite(images).all().item()):
        raise FloatingPointError(f"Pool images contain NaN/Inf: {packet_path}")
    if str(packet.get("source")) != "dsdm":
        raise ValueError(f"Pool source must be dsdm: {packet_path}")
    if str(packet.get("dataset")) != str(catalog["dataset"]):
        raise ValueError(f"Pool dataset mismatch: {packet_path}")
    if int(packet.get("ipc", -1)) != int(catalog["ipc"]):
        raise ValueError(f"Pool IPC mismatch: {packet_path}")
    if int(packet.get("factor", -1)) != int(catalog["factor"]):
        raise ValueError(f"Pool factor mismatch: {packet_path}")
    if str(packet.get("decode_type")) != str(catalog["decode_type"]):
        raise ValueError(f"Pool decode type mismatch: {packet_path}")
    expected_classes = list(range(int(catalog["num_classes"])))
    if [int(value) for value in packet.get("class_ids", [])] != expected_classes:
        raise ValueError(f"Pool class_ids must be global 0..{len(expected_classes) - 1}: {packet_path}")
    counts = Counter(int(value) for value in labels.tolist())
    if sorted(counts) != expected_classes or set(counts.values()) != {int(catalog["ipc"])}:
        raise ValueError(f"Pool label distribution is not IPC-balanced across all classes: {packet_path}")
    meta = packet.get("meta", {})
    if strict and str(meta.get("sender_model")) != str(expected_model_id):
        raise ValueError(
            f"Pool source model metadata mismatch: packet={meta.get('sender_model')} sender={expected_model_id}"
        )
    if meta.get("condense_complete") is not True or int(meta.get("completed_iterations", -1)) <= 0:
        raise ValueError(f"Pool condensation is not marked complete: {packet_path}")
    if strict and int(meta.get("completed_iterations", -1)) != int(source_spec["expected_completed_iterations"]):
        raise ValueError(f"Pool completed iteration provenance mismatch: {packet_path}")
    if packet.get("has_sender_logits", False) or "sender_logits" in packet:
        raise ValueError(f"Full-class source pool must be image-only: {packet_path}")

    manifest_path = _resolve_path(source_spec["best_manifest"])
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Pool best manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    best_path = _resolve_path(manifest["latest_best"])
    history_path = _resolve_path(manifest["history_snapshot"])
    if not best_path.is_file() or not history_path.is_file():
        raise FileNotFoundError(f"Pool best snapshot/history is incomplete: {manifest_path}")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    if not torch.equal(images, best.get("images")) or not torch.equal(labels, best.get("labels")):
        raise ValueError(f"Pool packet does not equal its persisted best snapshot: {packet_path}")
    history = torch.load(history_path, map_location="cpu", weights_only=False)
    if not torch.equal(images, history.get("images")) or not torch.equal(labels, history.get("labels")):
        raise ValueError(f"Pool packet does not equal its persisted best history snapshot: {packet_path}")
    if strict:
        if int(manifest.get("iteration", -1)) != int(source_spec["expected_best_iteration"]):
            raise ValueError(f"Pool best iteration provenance mismatch: {manifest_path}")
        expected_snapshot = _resolve_path(source_spec["expected_best_snapshot"]).resolve()
        if history_path.resolve() != expected_snapshot:
            raise ValueError(
                f"Pool best snapshot path mismatch: manifest={history_path.resolve()} expected={expected_snapshot}"
            )
        if file_sha256(history_path) != str(source_spec["expected_best_snapshot_sha256"]):
            raise ValueError(f"Pool best snapshot SHA-256 mismatch: {history_path}")
    return {
        "packet": packet,
        "packet_path": packet_path,
        "packet_sha256": packet_sha256,
        "best_manifest": manifest_path,
        "best_acc": float(manifest["best_acc"]),
        "best_iteration": int(manifest["iteration"]),
        "completed_iterations": int(meta["completed_iterations"]),
        "pool_protocol": str(catalog.get("protocol", POOL_PROTOCOL)),
        "source_model_id": str(meta.get("sender_model", "")),
        "history_snapshot": history_path,
        "history_snapshot_sha256": file_sha256(history_path),
    }


def _slice_packet(source, class_ids, ipc):
    packet = source["packet"]
    images = packet["images"]
    labels = packet["labels"].long()
    image_chunks = []
    label_chunks = []
    for class_id in class_ids:
        indices = torch.nonzero(labels == int(class_id), as_tuple=False).flatten()
        if int(indices.numel()) != int(ipc):
            raise ValueError(f"Class {class_id} has {indices.numel()} raw images, expected IPC={ipc}")
        image_chunks.append(images[indices].detach().cpu().clone())
        label_chunks.append(labels[indices].detach().cpu().clone())
    return torch.cat(image_chunks), torch.cat(label_chunks)


def _existing_packet_matches(path, source_sha256, agent_id, model_id, class_ids, pool_protocol):
    packet = torch.load(path, map_location="cpu", weights_only=False)
    meta = packet.get("meta", {})
    return (
        meta.get("pool_protocol") == pool_protocol
        and meta.get("pool_source_sha256") == source_sha256
        and int(meta.get("sender_agent", -1)) == int(agent_id)
        and str(meta.get("sender_model")) == str(model_id)
        and [int(value) for value in packet.get("class_ids", [])] == list(class_ids)
    )


def materialize_agent_packet(args, agent_id, model_id, class_ids, source_spec, catalog, *, resume=False, overwrite=False, dry_run=False):
    """Create one sender-local DSDM packet from its backbone pool and expert classes."""
    class_ids = [int(value) for value in class_ids]
    if not class_ids or len(class_ids) != len(set(class_ids)):
        raise ValueError(f"Agent {agent_id} class_ids must be non-empty and unique")
    if any(value < 0 or value >= int(catalog["num_classes"]) for value in class_ids):
        raise ValueError(f"Agent {agent_id} has out-of-range global labels: {class_ids}")
    strict = bool(catalog.get("strict_validation", False)) or strict_packet_validation_enabled(args)
    if strict and overwrite:
        raise ValueError("Strict DKP materialization forbids overwrite; use a new run_name")
    source = validate_pool_source(source_spec, catalog, expected_model_id=model_id)
    packet_path = get_agent_dir(args, agent_id) / "packets" / "dsdm_packet.pt"
    provenance_path = get_agent_dir(args, agent_id) / "metrics" / "fullclass_pool_slice_dsdm.json"
    if packet_path.exists() and not overwrite:
        if resume and _existing_packet_matches(
            packet_path,
            source["packet_sha256"],
            agent_id,
            model_id,
            class_ids,
            source["pool_protocol"],
        ):
            if strict:
                dsdm_root = ROOT / "DSDM"
                if str(dsdm_root) not in sys.path:
                    sys.path.append(str(dsdm_root))
                from packet_consumer import _decode_dsdm_images
                from packet_integrity import validate_strict_dkp_packet

                existing = torch.load(packet_path, map_location="cpu", weights_only=False)
                decoded_images, decoded_labels = _decode_dsdm_images(args, existing)
                validate_strict_dkp_packet(
                    args,
                    existing,
                    decoded_images,
                    decoded_labels,
                    sender_agent=agent_id,
                    sender_model=model_id,
                    require_sender_logits=bool(existing.get("has_sender_logits", False)),
                )
            return {
                "agent_id": int(agent_id),
                "model": str(model_id),
                "class_ids": class_ids,
                "packet_path": _display_path(packet_path),
                "status": "reused",
                "source_sha256": source["packet_sha256"],
            }
        raise FileExistsError(f"Target packet already exists; use --resume or --overwrite: {packet_path}")

    images, labels = _slice_packet(source, class_ids, int(catalog["ipc"]))
    expected_labels = [class_id for class_id in class_ids for _ in range(int(catalog["ipc"]))]
    if labels.tolist() != expected_labels:
        raise ValueError(f"Agent {agent_id} slice does not preserve requested global-label order")
    payload = {
        "images": images,
        "labels": labels,
        "class_ids": class_ids,
        "source": "dsdm",
        "dataset": str(catalog["dataset"]),
        "ipc": int(catalog["ipc"]),
        "factor": int(catalog["factor"]),
        "decode_type": str(catalog["decode_type"]),
        "packet_format": "compact_multi_formation",
        "meta": {
            "run_name": str(args.run_name),
            "method": "DSDM",
            "condense_complete": True,
            "completed_iterations": source["completed_iterations"],
            "pool_protocol": source["pool_protocol"],
            "pool_reuse": True,
            "source_condensation_scope": "full_class_pool",
            "simulated_sender_local_distillation": True,
            "sender_agent": int(agent_id),
            "sender_model": str(model_id),
            "sender_class_ids": class_ids,
            "pool_source_packet": _display_path(source["packet_path"]),
            "pool_source_sha256": source["packet_sha256"],
            "pool_source_best_manifest": _display_path(source["best_manifest"]),
            "pool_source_best_snapshot": _display_path(source["history_snapshot"]),
            "pool_source_best_snapshot_sha256": source["history_snapshot_sha256"],
            "pool_source_best_acc": source["best_acc"],
            "pool_source_best_iteration": source["best_iteration"],
            "pool_catalog": str(catalog["catalog_path"]),
            "guide_weights_communicated": False,
        },
    }
    if strict:
        dsdm_root = ROOT / "DSDM"
        if str(dsdm_root) not in sys.path:
            sys.path.append(str(dsdm_root))
        from packet_consumer import _decode_dsdm_images

        decoded_images, decoded_labels = _decode_dsdm_images(args, payload)
        payload["decoded_integrity"] = decoded_digests(decoded_images, decoded_labels)
    _validate_packet_payload(payload)
    result = {
        "agent_id": int(agent_id),
        "model": str(model_id),
        "class_ids": class_ids,
        "raw_images": int(images.shape[0]),
        "decoded_images": int(images.shape[0]) * int(catalog["factor"]) ** 2,
        "packet_path": _display_path(packet_path),
        "source_packet": _display_path(source["packet_path"]),
        "source_sha256": source["packet_sha256"],
        "source_best_acc": source["best_acc"],
        "source_best_iteration": source["best_iteration"],
        "source_best_snapshot_sha256": source["history_snapshot_sha256"],
        "pool_protocol": source["pool_protocol"],
        "status": "validated" if dry_run else "materialized",
    }
    if not dry_run:
        atomic_torch_save(payload, packet_path)
        atomic_write_json(result, provenance_path)
    return result


def materialize_pool_packets(config, args, catalog, *, only_agent=None, resume=False, overwrite=False, dry_run=False):
    """Materialize all configured senders using an exact model-to-pool mapping."""
    if str(catalog["dataset"]) != str(args.dataset):
        raise ValueError(f"Catalog dataset={catalog['dataset']} does not match config dataset={args.dataset}")
    if int(catalog["num_classes"]) != int(get_num_classes(args)):
        raise ValueError("Catalog and config class counts differ")
    if int(catalog["ipc"]) != int(args.ipc) or int(catalog["factor"]) != int(args.factor):
        raise ValueError("Catalog and config IPC/factor differ")
    if bool(catalog.get("strict_validation", False)) or strict_packet_validation_enabled(args):
        validate_strict_partition(args)
        if str(catalog.get("protocol")) != STRICT_POOL_PROTOCOL:
            raise ValueError(f"Strict DKP requires pool protocol {STRICT_POOL_PROTOCOL}")
    class_split = get_agent_class_split(config)
    model_split = get_agent_model_split(config)
    agent_ids = [int(only_agent)] if only_agent is not None else sorted(class_split)
    results = []
    for agent_id in agent_ids:
        if agent_id not in class_split or agent_id not in model_split:
            raise ValueError(f"Unknown agent id: {agent_id}")
        model_id = model_split[agent_id]
        source_spec = catalog["sources"].get(model_id)
        if source_spec is None:
            raise ValueError(
                f"No full-class pool for model '{model_id}'. Exact backbone matching is required; "
                f"available={sorted(catalog['sources'])}"
            )
        results.append(
            materialize_agent_packet(
                args,
                agent_id,
                model_id,
                class_split[agent_id],
                source_spec,
                catalog,
                resume=resume,
                overwrite=overwrite,
                dry_run=dry_run,
            )
        )
    return results
