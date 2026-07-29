"""Strict integrity checks for opt-in DKP communication protocols."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch

from agent_data import get_agent_class_split, get_agent_model_split, get_num_classes


ROOT = Path(__file__).resolve().parent
STRICT_DKP_PROTOCOL = "dkp_sl_iclr2027_cifar100_5x20_ipc10_v1"
DYNAMIC_DKP_PROTOCOL_VERSION = "v2"
SUPPORTED_DKP_AGENT_COUNTS = (5, 10, 20)
POOL_PROTOCOL = "backbone_specific_fullclass_pool_slice_v2_dkp"
CIFAR100_PAT5_SEED0_CLASS_SPLIT = {
    0: [26, 86, 2, 55, 75, 93, 16, 73, 54, 95, 53, 92, 78, 13, 7, 30, 22, 24, 33, 8],
    1: [43, 62, 3, 71, 45, 48, 6, 99, 82, 76, 60, 80, 90, 68, 51, 27, 18, 56, 63, 74],
    2: [1, 61, 42, 41, 4, 15, 17, 40, 38, 5, 91, 59, 0, 34, 28, 50, 11, 35, 23, 52],
    3: [10, 31, 66, 57, 79, 85, 32, 84, 14, 89, 19, 29, 49, 97, 98, 69, 20, 94, 72, 77],
    4: [25, 37, 81, 46, 39, 65, 58, 12, 88, 70, 87, 36, 21, 83, 9, 96, 67, 64, 47, 44],
}


@dataclass(frozen=True)
class StrictDkpContract:
    """Resolved dimensions and identities for one strict DKP protocol."""

    protocol: str
    version: str
    num_classes: int
    agent_count: int
    classes_per_agent: int
    ipc: int
    factor: int
    decode_type: str
    class_split: dict
    model_split: dict

    @property
    def raw_per_sender(self):
        return self.classes_per_agent * self.ipc

    @property
    def decoded_per_class(self):
        return self.ipc * self.factor ** 2

    @property
    def decoded_per_sender(self):
        return self.classes_per_agent * self.decoded_per_class


def build_strict_dkp_protocol(agent_count, classes_per_agent, ipc):
    """Build the only dynamic protocol id accepted by the strict v2 gate."""
    values = {
        "agent_count": agent_count,
        "classes_per_agent": classes_per_agent,
        "ipc": ipc,
    }
    normalized = {}
    for name, value in values.items():
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a positive integer")
        try:
            normalized[name] = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a positive integer") from error
        if normalized[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    return (
        "dkp_sl_iclr2027_cifar100_"
        f"{normalized['agent_count']}x{normalized['classes_per_agent']}_"
        f"ipc{normalized['ipc']}_{DYNAMIC_DKP_PROTOCOL_VERSION}"
    )


def strict_packet_validation_enabled(args):
    """Return whether the new fail-closed packet protocol is explicitly enabled."""
    return bool(getattr(args, "strict_packet_validation", False))


def resolve_project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_tensor_digest(digest, tensor):
    tensor = tensor.detach().cpu().contiguous()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))


def tensor_sha256(tensor):
    digest = hashlib.sha256()
    _update_tensor_digest(digest, tensor)
    return digest.hexdigest()


def ordered_alignment_sha256(images, labels, logits):
    """Hash decoded image, global label and sender-logit rows in exact order."""
    if images.shape[0] != labels.shape[0] or images.shape[0] != logits.shape[0]:
        raise ValueError("decoded image/label/logit row counts differ")
    digest = hashlib.sha256(b"dkp-decoded-alignment-v1\0")
    for index in range(int(images.shape[0])):
        digest.update(int(index).to_bytes(8, byteorder="little", signed=False))
        _update_tensor_digest(digest, images[index])
        _update_tensor_digest(digest, labels[index : index + 1].long())
        _update_tensor_digest(digest, logits[index])
    return digest.hexdigest()


def decoded_digests(images, labels, logits=None):
    result = {
        "decoded_images_sha256": tensor_sha256(images),
        "decoded_labels_sha256": tensor_sha256(labels.long()),
    }
    if logits is not None:
        result.update(
            {
                "sender_logits_sha256": tensor_sha256(logits),
                "decoded_alignment_sha256": ordered_alignment_sha256(images, labels, logits),
            }
        )
    return result


def _require_explicit_mapping(args, attribute, config_key):
    if isinstance(args, dict):
        value = args.get("agents", {}).get(config_key)
    else:
        value = getattr(args, attribute, None)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Strict DKP protocol requires explicit agents.{config_key}")


def _require_positive_int(args, attribute, section=None):
    value = _configured_value(args, attribute, section=section)
    if isinstance(value, bool):
        raise ValueError(f"Strict DKP protocol requires positive integer {attribute}")
    try:
        value = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Strict DKP protocol requires positive integer {attribute}") from error
    if value <= 0:
        raise ValueError(f"Strict DKP protocol requires positive integer {attribute}")
    return value


def _configured_value(args, attribute, section=None):
    if not isinstance(args, dict):
        return getattr(args, attribute, None)
    if section is None:
        return args.get(attribute)
    return args.get(section, {}).get(attribute)


def resolve_strict_dkp_contract(args):
    """Resolve a complete v1/v2 contract, rejecting guessed or unsupported protocols."""
    dataset = _configured_value(args, "dataset")
    if isinstance(dataset, dict):
        dataset = dataset.get("name")
    if str(dataset) != "cifar100":
        raise ValueError("Strict DKP protocol requires dataset=cifar100")

    num_classes = int(get_num_classes(args))
    if num_classes != 100:
        raise ValueError("Strict DKP protocol requires 100 global classes")
    if not isinstance(args, dict):
        for attribute in ("num_classes", "nclass"):
            if hasattr(args, attribute) and int(getattr(args, attribute)) != num_classes:
                raise ValueError(f"Strict DKP {attribute} conflicts with the global class count")

    config_section = "distillation" if isinstance(args, dict) else None
    ipc = _require_positive_int(args, "ipc", section=config_section)
    factor = _require_positive_int(args, "factor", section=config_section)
    decode_type = str(_configured_value(args, "decode_type", section=config_section) or "")
    if decode_type != "single":
        raise ValueError("Strict DKP protocol requires deterministic decode_type=single")

    _require_explicit_mapping(args, "agent_class_split", "class_split")
    _require_explicit_mapping(args, "agent_model_split", "model_split")
    split = get_agent_class_split(args)
    model_split = get_agent_model_split(args)
    agent_count = len(split)
    if agent_count not in SUPPORTED_DKP_AGENT_COUNTS:
        raise ValueError(
            f"Strict DKP protocol supports agent counts {SUPPORTED_DKP_AGENT_COUNTS}, got {agent_count}"
        )
    if num_classes % agent_count != 0:
        raise ValueError(
            f"Strict DKP global class count {num_classes} is not divisible by {agent_count} agents"
        )
    if isinstance(args, dict):
        configured_agent_count = args.get("agents", {}).get("num_agents")
        if configured_agent_count is None or int(configured_agent_count) != agent_count:
            raise ValueError(
                "Strict DKP agents.num_agents must equal the explicit class/model split size"
            )
    expected_sender_ids = list(range(agent_count))
    if sorted(split) != expected_sender_ids:
        raise ValueError(
            f"Strict DKP protocol requires contiguous sender ids 0..{agent_count - 1}, got {sorted(split)}"
        )
    if sorted(model_split) != expected_sender_ids:
        raise ValueError(
            "Strict DKP agents.model_split must cover exactly the configured sender ids; "
            f"got {sorted(model_split)}"
        )

    classes_per_agent = num_classes // agent_count
    flat = []
    normalized_split = {}
    normalized_models = {}
    for agent_id in expected_sender_ids:
        classes = [int(value) for value in split[agent_id]]
        if len(classes) != classes_per_agent or len(set(classes)) != classes_per_agent:
            raise ValueError(
                f"Agent {agent_id} must have exactly {classes_per_agent} unique global classes"
            )
        normalized_split[agent_id] = classes
        flat.extend(classes)
        model_id = str(model_split[agent_id])
        if not model_id:
            raise ValueError(f"Agent {agent_id} model id must be non-empty")
        normalized_models[agent_id] = model_id
    counts = Counter(flat)
    invalid_counts = sorted(class_id for class_id, count in counts.items() if count != 1)
    if invalid_counts or sorted(flat) != list(range(num_classes)):
        raise ValueError(
            "Strict DKP agent classes must be disjoint with union 0..99; "
            f"invalid_counts={invalid_counts} union_size={len(counts)}"
        )

    if isinstance(args, dict):
        protocol = str(_configured_value(args, "protocol", section="communication") or "")
    else:
        protocol = str(_configured_value(args, "communication_protocol") or "")
    if not protocol:
        raise ValueError("Strict DKP protocol requires an explicit communication.protocol")
    dynamic_protocol = build_strict_dkp_protocol(agent_count, classes_per_agent, ipc)
    if protocol == STRICT_DKP_PROTOCOL:
        if agent_count != 5 or classes_per_agent != 20 or ipc != 10 or factor != 2:
            raise ValueError("Legacy strict DKP v1 is fixed to CIFAR-100 5x20, IPC10, factor2")
        for agent_id, expected_classes in CIFAR100_PAT5_SEED0_CLASS_SPLIT.items():
            if normalized_split[agent_id] != expected_classes:
                raise ValueError(
                    f"Agent {agent_id} classes do not match the legacy CIFAR-100 seed0 v1 order"
                )
        version = "v1"
    elif protocol == dynamic_protocol:
        if factor != 2:
            raise ValueError("Dynamic strict DKP v2 currently requires factor=2")
        version = DYNAMIC_DKP_PROTOCOL_VERSION
    else:
        raise ValueError(
            "Unknown or config-mismatched strict DKP protocol: "
            f"got={protocol!r} expected={dynamic_protocol!r} "
            f"(or legacy {STRICT_DKP_PROTOCOL!r})"
        )

    return StrictDkpContract(
        protocol=protocol,
        version=version,
        num_classes=num_classes,
        agent_count=agent_count,
        classes_per_agent=classes_per_agent,
        ipc=ipc,
        factor=factor,
        decode_type=decode_type,
        class_split=normalized_split,
        model_split=normalized_models,
    )


def validate_strict_partition(args):
    """Validate and return the config-defined global class partition."""
    return resolve_strict_dkp_contract(args).class_split


def validate_strict_external_sender_ids(args, receiver_agent, sender_ids):
    """Require the external route to contain every other sender and never self."""
    contract = resolve_strict_dkp_contract(args)
    receiver_agent = int(receiver_agent)
    if receiver_agent not in contract.class_split:
        raise ValueError(f"Unknown receiver agent id: {receiver_agent}")
    normalized = [int(sender_id) for sender_id in sender_ids]
    if receiver_agent in normalized:
        raise ValueError(f"Receiver {receiver_agent} external route includes its own DKP")
    expected = sorted(set(contract.class_split) - {receiver_agent})
    if len(normalized) != len(set(normalized)) or sorted(normalized) != expected:
        raise ValueError(
            f"Receiver {receiver_agent} external sender set mismatch: "
            f"got={normalized} expected={expected}"
        )
    return expected


def _expected_ordered_labels(class_ids, per_class):
    return [int(class_id) for class_id in class_ids for _ in range(int(per_class))]


def _require_equal(actual, expected, message):
    if actual != expected:
        raise ValueError(f"{message}: got={actual!r} expected={expected!r}")


def _require_integer_labels(labels, name):
    if not torch.is_tensor(labels) or labels.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise ValueError(f"Strict DKP {name} must be an integer tensor")


def _validate_teacher_provenance(
    packet,
    num_classes,
    *,
    sender_agent,
    sender_model,
    expected_classes,
):
    teacher = packet.get("sender_logit_teacher")
    meta_teacher = packet.get("meta", {}).get("sender_logit_teacher")
    quality_teacher = packet.get("sender_logit_quality", {}).get("teacher")
    if not isinstance(teacher, dict) or not isinstance(meta_teacher, dict) or not isinstance(quality_teacher, dict):
        raise ValueError("Strict DKP packet lacks complete sender expert provenance")
    checkpoint_sha = teacher.get("checkpoint_sha256")
    if not checkpoint_sha or len(str(checkpoint_sha)) != 64:
        raise ValueError("Strict DKP sender expert provenance lacks checkpoint SHA-256")
    _require_equal(meta_teacher.get("checkpoint_sha256"), checkpoint_sha, "meta teacher SHA mismatch")
    _require_equal(quality_teacher.get("checkpoint_sha256"), checkpoint_sha, "quality teacher SHA mismatch")
    _require_equal(int(teacher.get("output_dim", -1)), int(num_classes), "sender expert output dimension mismatch")
    _require_equal(int(teacher.get("agent_id", -1)), int(sender_agent), "sender expert agent mismatch")
    _require_equal(str(teacher.get("model_id")), str(sender_model), "sender expert model mismatch")
    _require_equal(
        [int(value) for value in teacher.get("class_ids", [])],
        expected_classes,
        "sender expert class order mismatch",
    )
    checkpoint_path = teacher.get("checkpoint")
    if not checkpoint_path:
        raise ValueError("Strict DKP sender expert provenance lacks checkpoint path")
    _require_equal(meta_teacher.get("checkpoint"), checkpoint_path, "meta teacher checkpoint path mismatch")
    _require_equal(quality_teacher.get("checkpoint"), checkpoint_path, "quality teacher checkpoint path mismatch")
    resolved = resolve_project_path(checkpoint_path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Sender expert checkpoint does not exist: {resolved}")
    _require_equal(file_sha256(resolved), checkpoint_sha, "sender expert checkpoint SHA mismatch")
    return str(checkpoint_sha)


def validate_strict_dkp_packet(
    args,
    packet,
    decoded_images,
    decoded_labels,
    *,
    sender_agent,
    sender_model,
    require_sender_logits,
):
    """Fail closed on all tensor, order and provenance invariants of one DKP."""
    contract = resolve_strict_dkp_contract(args)
    split = contract.class_split
    model_split = contract.model_split
    sender_agent = int(sender_agent)
    if sender_agent not in split:
        raise ValueError(f"Unknown sender agent id: {sender_agent}")
    expected_classes = [int(value) for value in split[sender_agent]]
    _require_equal(str(sender_model), str(model_split[sender_agent]), "manifest sender model mismatch")
    _require_equal(packet.get("class_ids"), expected_classes, "packet class_ids order mismatch")
    _require_equal(str(packet.get("source")), "dsdm", "packet source mismatch")
    _require_equal(str(packet.get("dataset")), "cifar100", "packet dataset mismatch")
    _require_equal(int(packet.get("ipc", -1)), contract.ipc, "packet IPC mismatch")
    _require_equal(int(packet.get("factor", -1)), contract.factor, "packet factor mismatch")
    _require_equal(str(packet.get("decode_type")), contract.decode_type, "packet decode type mismatch")

    images = packet.get("images")
    labels = packet.get("labels")
    if not torch.is_tensor(images) or not torch.is_tensor(labels):
        raise ValueError("Strict DKP packet lacks image/label tensors")
    _require_integer_labels(labels, "raw labels")
    if images.ndim != 4 or labels.ndim != 1 or images.shape[0] != labels.shape[0]:
        raise ValueError("Strict DKP raw image/label tensor shapes are invalid")
    if int(images.shape[0]) != contract.raw_per_sender:
        raise ValueError(
            f"Strict DKP sender must contain {contract.raw_per_sender} raw images, got {images.shape[0]}"
        )
    if not bool(torch.isfinite(images).all().item()):
        raise FloatingPointError("Strict DKP raw images contain NaN/Inf")
    _require_equal(
        labels.long().tolist(),
        _expected_ordered_labels(expected_classes, contract.ipc),
        "raw global-label order mismatch",
    )

    meta = packet.get("meta", {})
    _require_equal(meta.get("pool_protocol"), POOL_PROTOCOL, "pool protocol mismatch")
    _require_equal(int(meta.get("sender_agent", -1)), sender_agent, "packet sender id mismatch")
    _require_equal(str(meta.get("sender_model")), str(sender_model), "packet sender model mismatch")
    _require_equal(meta.get("sender_class_ids"), expected_classes, "packet provenance class order mismatch")
    source_path = resolve_project_path(meta.get("pool_source_packet", ""))
    if not source_path.is_file():
        raise FileNotFoundError(f"Pool source packet does not exist: {source_path}")
    _require_equal(file_sha256(source_path), meta.get("pool_source_sha256"), "pool source packet SHA mismatch")
    best_snapshot_path = resolve_project_path(meta.get("pool_source_best_snapshot", ""))
    if not best_snapshot_path.is_file():
        raise FileNotFoundError(f"Pool source best snapshot does not exist: {best_snapshot_path}")
    _require_equal(
        file_sha256(best_snapshot_path),
        meta.get("pool_source_best_snapshot_sha256"),
        "pool source best-snapshot SHA mismatch",
    )

    expected_decoded_per_class = contract.decoded_per_class
    expected_decoded = contract.decoded_per_sender
    if not torch.is_tensor(decoded_images) or not torch.is_tensor(decoded_labels):
        raise ValueError("Strict DKP decoded images/labels must be tensors")
    _require_integer_labels(decoded_labels, "decoded labels")
    if decoded_images.ndim != 4 or decoded_labels.ndim != 1:
        raise ValueError("Strict DKP decoded image/label tensor shapes are invalid")
    if int(decoded_images.shape[0]) != expected_decoded or int(decoded_labels.shape[0]) != expected_decoded:
        raise ValueError(
            f"Strict DKP decoded row count must be {expected_decoded}, "
            f"got images={decoded_images.shape[0]} labels={decoded_labels.shape[0]}"
        )
    if not bool(torch.isfinite(decoded_images).all().item()):
        raise FloatingPointError("Strict DKP decoded images contain NaN/Inf")
    _require_equal(
        decoded_labels.long().tolist(),
        _expected_ordered_labels(expected_classes, expected_decoded_per_class),
        "decoded global-label order mismatch",
    )
    expected_base_digests = decoded_digests(decoded_images, decoded_labels)
    stored_digests = packet.get("decoded_integrity", {})
    for key, value in expected_base_digests.items():
        _require_equal(stored_digests.get(key), value, f"{key} mismatch")

    has_logits = packet.get("has_sender_logits", False)
    if not isinstance(has_logits, bool):
        raise ValueError("Strict DKP has_sender_logits must be boolean when present")
    if require_sender_logits and not has_logits:
        raise ValueError("Strict DKP communication requires sender logits for every packet")
    if not has_logits:
        unexpected_logit_fields = sorted(
            key
            for key in (
                "sender_logits",
                "sender_logit_class_ids",
                "sender_logit_dim",
                "sender_logit_num_images",
                "sender_logit_teacher",
            )
            if key in packet
        )
        if unexpected_logit_fields:
            raise ValueError(
                "Strict DKP packet marks sender logits absent but retains fields: "
                f"{unexpected_logit_fields}"
            )
        return {"checkpoint_sha256": None, **expected_base_digests}

    logits = packet.get("sender_logits")
    logit_class_ids = packet.get("sender_logit_class_ids")
    expected_logit_shape = (expected_decoded, contract.classes_per_agent)
    if not torch.is_tensor(logits) or logits.ndim != 2 or tuple(logits.shape) != expected_logit_shape:
        raise ValueError(
            f"Strict DKP sender logits must have shape {expected_logit_shape}, "
            f"got {None if not torch.is_tensor(logits) else tuple(logits.shape)}"
        )
    if logits.dtype != torch.float16:
        raise ValueError(f"Strict DKP sender logits must use float16, got {logits.dtype}")
    if not bool(torch.isfinite(logits).all().item()):
        raise FloatingPointError("Strict DKP sender logits contain NaN/Inf")
    if not torch.is_tensor(logit_class_ids):
        raise ValueError("Strict DKP sender_logit_class_ids must be a tensor")
    _require_integer_labels(logit_class_ids, "sender_logit_class_ids")
    _require_equal(logit_class_ids.long().tolist(), expected_classes, "sender logit class order mismatch")
    _require_equal(
        int(packet.get("sender_logit_dim", -1)),
        contract.classes_per_agent,
        "sender logit dimension mismatch",
    )
    _require_equal(
        int(packet.get("sender_logit_num_images", -1)), expected_decoded, "sender logit row count metadata mismatch"
    )
    _require_equal(str(packet.get("sender_logit_dtype")), "float16", "sender logit dtype metadata mismatch")
    expected_digests = decoded_digests(decoded_images, decoded_labels, logits)
    for key, value in expected_digests.items():
        _require_equal(stored_digests.get(key), value, f"{key} mismatch")
    checkpoint_sha = _validate_teacher_provenance(
        packet,
        contract.num_classes,
        sender_agent=sender_agent,
        sender_model=sender_model,
        expected_classes=expected_classes,
    )
    return {"checkpoint_sha256": checkpoint_sha, **expected_digests}


def validate_strict_manifest_rows(args, rows, packet_method):
    """Require one complete immutable packet row per config-defined sender."""
    if str(packet_method) != "dsdm":
        raise ValueError("Strict DKP protocol requires packet_method=dsdm")
    contract = resolve_strict_dkp_contract(args)
    split = contract.class_split
    model_split = contract.model_split
    if len(rows) != contract.agent_count:
        raise ValueError(
            f"Strict DKP manifest requires exactly {contract.agent_count} rows, got {len(rows)}"
        )
    sender_ids = [int(row.get("sender_agent", -1)) for row in rows]
    expected_sender_ids = list(range(contract.agent_count))
    if sorted(sender_ids) != expected_sender_ids or len(set(sender_ids)) != contract.agent_count:
        raise ValueError(
            f"Strict DKP manifest requires one row per sender 0..{contract.agent_count - 1}, "
            f"got {sender_ids}"
        )
    for receiver_agent in expected_sender_ids:
        validate_strict_external_sender_ids(
            args,
            receiver_agent,
            [sender_agent for sender_agent in sender_ids if sender_agent != receiver_agent],
        )
    require_logits = bool(getattr(args, "use_sender_logits", False))
    for row in rows:
        sender_agent = int(row["sender_agent"])
        expected_classes = [int(value) for value in split[sender_agent]]
        expected_model = str(model_split[sender_agent])
        _require_equal(str(row.get("sender_model")), expected_model, "manifest sender model mismatch")
        _require_equal(
            [int(value) for value in str(row.get("classes", "")).split(",") if value != ""],
            expected_classes,
            "manifest class order mismatch",
        )
        _require_equal(str(row.get("method")).lower(), "dsdm", "manifest method mismatch")
        _require_equal(int(row.get("ipc", -1)), contract.ipc, "manifest IPC mismatch")
        _require_equal(str(row.get("protocol")), contract.protocol, "manifest protocol mismatch")
        _require_equal(str(row.get("complete", "")).lower(), "true", "manifest completeness gate failed")
        packet_path = resolve_project_path(row.get("packet_path", ""))
        if not packet_path.is_file():
            raise FileNotFoundError(f"Manifest packet does not exist: {packet_path}")
        _require_equal(file_sha256(packet_path), row.get("packet_sha256"), "manifest packet SHA mismatch")
        packet = torch.load(packet_path, map_location="cpu", weights_only=False)
        dsdm_root = ROOT / "DSDM"
        if str(dsdm_root) not in sys.path:
            sys.path.append(str(dsdm_root))
        from packet_consumer import _decode_dsdm_images

        decoded_images, decoded_labels = _decode_dsdm_images(args, packet)
        proof = validate_strict_dkp_packet(
            args,
            packet,
            decoded_images,
            decoded_labels,
            sender_agent=sender_agent,
            sender_model=expected_model,
            require_sender_logits=require_logits,
        )
        meta = packet.get("meta", {})
        _require_equal(row.get("pool_protocol"), meta.get("pool_protocol"), "manifest pool protocol mismatch")
        _require_equal(
            row.get("pool_source_sha256"), meta.get("pool_source_sha256"), "manifest pool source SHA mismatch"
        )
        _require_equal(
            row.get("pool_source_best_snapshot_sha256"),
            meta.get("pool_source_best_snapshot_sha256"),
            "manifest pool best-snapshot SHA mismatch",
        )
        _require_equal(
            row.get("decoded_images_sha256"),
            proof.get("decoded_images_sha256"),
            "manifest decoded-image digest mismatch",
        )
        _require_equal(
            row.get("decoded_labels_sha256"),
            proof.get("decoded_labels_sha256"),
            "manifest decoded-label digest mismatch",
        )
        _require_equal(
            row.get("sender_logits_sha256", ""),
            proof.get("sender_logits_sha256", ""),
            "manifest sender-logit digest mismatch",
        )
        _require_equal(row.get("decoded_alignment_sha256", ""), proof.get("decoded_alignment_sha256", ""), "manifest alignment digest mismatch")
        if require_logits:
            _require_equal(row.get("expert_checkpoint_sha256"), proof["checkpoint_sha256"], "manifest expert SHA mismatch")
        else:
            _require_equal(row.get("expert_checkpoint_sha256", ""), "", "image-only manifest expert SHA must be empty")
    return True


def validate_receiver_expert_provenance(args, receiver_agent, manifest_rows, checkpoint_path):
    """Prove that receiver initialization, FR teacher and sender teacher use one checkpoint."""
    if not strict_packet_validation_enabled(args):
        return None
    receiver_agent = int(receiver_agent)
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Receiver expert checkpoint does not exist: {checkpoint_path}")
    checkpoint_sha = file_sha256(checkpoint_path)
    expected_classes = [int(value) for value in validate_strict_partition(args)[receiver_agent]]

    expert_manifest_path = checkpoint_path.parent / "expert_manifest.json"
    if not expert_manifest_path.is_file():
        raise FileNotFoundError(f"Receiver expert manifest does not exist: {expert_manifest_path}")
    expert_manifest = json.loads(expert_manifest_path.read_text(encoding="utf-8"))
    _require_equal(expert_manifest.get("expert_sha256"), checkpoint_sha, "receiver expert manifest SHA mismatch")
    _require_equal(int(expert_manifest.get("agent_id", -1)), receiver_agent, "receiver expert manifest agent mismatch")
    _require_equal(
        int(expert_manifest.get("global_output_dim", -1)),
        get_num_classes(args),
        "receiver expert manifest output dimension mismatch",
    )
    _require_equal(
        [int(value) for value in expert_manifest.get("active_class_ids", [])],
        expected_classes,
        "receiver expert manifest class order mismatch",
    )

    reuse_manifest_path = (
        Path(getattr(args, "output_root"))
        / str(getattr(args, "run_name"))
        / "provenance"
        / "expert_reuse_manifest.json"
    )
    if not reuse_manifest_path.is_file():
        raise FileNotFoundError(f"Strict receiver run lacks expert reuse provenance: {reuse_manifest_path}")
    reuse_manifest = json.loads(reuse_manifest_path.read_text(encoding="utf-8"))
    records = [
        record
        for record in reuse_manifest.get("agents", [])
        if int(record.get("agent_id", -1)) == receiver_agent
    ]
    if len(records) != 1:
        raise ValueError(f"Expert reuse manifest must contain receiver {receiver_agent} exactly once")
    reused_sha = records[0].get("artifacts", {}).get("expert_model.pt", {}).get("sha256")
    _require_equal(reused_sha, checkpoint_sha, "receiver reused checkpoint SHA mismatch")

    if bool(getattr(args, "use_logits", getattr(args, "use_sender_logits", False))):
        sender_rows = [row for row in manifest_rows if int(row.get("sender_agent", -1)) == receiver_agent]
        if len(sender_rows) != 1:
            raise ValueError(f"Communication manifest must contain receiver/sender {receiver_agent} exactly once")
        _require_equal(
            sender_rows[0].get("expert_checkpoint_sha256"),
            checkpoint_sha,
            "receiver initialization/FR/sender-logit teacher SHA mismatch",
        )
    return checkpoint_sha
