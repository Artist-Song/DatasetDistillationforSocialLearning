import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from packet_integrity import (
    CIFAR100_PAT5_SEED0_CLASS_SPLIT,
    POOL_PROTOCOL,
    STRICT_DKP_PROTOCOL,
    build_strict_dkp_protocol,
    decoded_digests,
    file_sha256,
    resolve_strict_dkp_contract,
    validate_strict_dkp_packet,
    validate_strict_external_sender_ids,
    validate_strict_manifest_rows,
    validate_strict_partition,
    validate_receiver_expert_provenance,
)
from validate_packets import _expected_full_real_sender_raw
from packet_logits import _ensure_dsdm_path
from social_output_manager import read_packet_manifest, register_agent_packet, write_packet_manifest


MODELS = {
    0: "convnet3w1",
    1: "convnet4w15",
    2: "alexnet",
    3: "resnet10_standard",
    4: "resnet18_standard",
}


class StrictDkpPacketIntegrityTests(unittest.TestCase):
    @staticmethod
    def _dynamic_config(agent_count):
        per_agent = 100 // int(agent_count)
        return {
            "dataset": {"name": "cifar100", "num_classes": 100},
            "agents": {
                "num_agents": int(agent_count),
                "class_split": {
                    f"agent_{agent_id}": list(
                        range(agent_id * per_agent, (agent_id + 1) * per_agent)
                    )
                    for agent_id in range(agent_count)
                },
                "model_split": {
                    f"agent_{agent_id}": f"model_{agent_id}"
                    for agent_id in range(agent_count)
                },
            },
            "distillation": {"ipc": 10, "factor": 2, "decode_type": "single"},
            "communication": {
                "protocol": build_strict_dkp_protocol(agent_count, per_agent, 10)
            },
        }

    def _args(self, root, *, use_sender_logits=True):
        return SimpleNamespace(
            dataset="cifar100",
            num_classes=100,
            nclass=100,
            ipc=10,
            factor=2,
            decode_type="single",
            batch_syn_max=128,
            agent_class_split={key: list(value) for key, value in CIFAR100_PAT5_SEED0_CLASS_SPLIT.items()},
            agent_model_split=dict(MODELS),
            strict_packet_validation=True,
            use_sender_logits=use_sender_logits,
            communication_protocol=STRICT_DKP_PROTOCOL,
            output_root=str(root / "outputs"),
            run_name="strict_fixture",
        )

    def _packet(self, root, args, agent_id, *, with_logits=True):
        class_ids = args.agent_class_split[agent_id]
        raw_labels = torch.tensor(
            [class_id for class_id in class_ids for _ in range(10)], dtype=torch.long
        )
        raw_images = torch.arange(200 * 3 * 4 * 4, dtype=torch.float32).reshape(200, 3, 4, 4)
        raw_images = raw_images / raw_images.max()
        source_path = root / f"source_agent_{agent_id}.pt"
        torch.save({"agent": agent_id}, source_path)
        best_snapshot_path = root / f"best_snapshot_agent_{agent_id}.pt"
        torch.save({"agent": agent_id, "kind": "best"}, best_snapshot_path)
        checkpoint_path = root / f"expert_agent_{agent_id}.pt"
        checkpoint_path.write_bytes(f"expert-{agent_id}".encode("ascii"))
        teacher = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "output_dim": 100,
            "agent_id": agent_id,
            "model_id": MODELS[agent_id],
            "class_ids": list(class_ids),
        }
        packet = {
            "images": raw_images,
            "labels": raw_labels,
            "class_ids": list(class_ids),
            "source": "dsdm",
            "dataset": "cifar100",
            "ipc": 10,
            "factor": 2,
            "decode_type": "single",
            "meta": {
                "pool_protocol": POOL_PROTOCOL,
                "pool_source_packet": str(source_path),
                "pool_source_sha256": file_sha256(source_path),
                "pool_source_best_snapshot": str(best_snapshot_path),
                "pool_source_best_snapshot_sha256": file_sha256(best_snapshot_path),
                "sender_agent": agent_id,
                "sender_model": MODELS[agent_id],
                "sender_class_ids": list(class_ids),
            },
        }
        _ensure_dsdm_path()
        from packet_consumer import _decode_dsdm_images

        decoded_images, decoded_labels = _decode_dsdm_images(args, packet)
        if with_logits:
            logits = torch.arange(decoded_images.shape[0] * 20, dtype=torch.float16).reshape(-1, 20)
            logits = logits / 1000
            packet.update(
                {
                    "has_sender_logits": True,
                    "sender_logits": logits,
                    "sender_logit_class_ids": torch.tensor(class_ids, dtype=torch.long),
                    "sender_logit_dim": 20,
                    "sender_logit_num_images": int(decoded_images.shape[0]),
                    "sender_logit_dtype": "float16",
                    "sender_logit_teacher": dict(teacher),
                    "sender_logit_quality": {"teacher": dict(teacher)},
                }
            )
            packet["meta"]["sender_logit_teacher"] = dict(teacher)
            packet["decoded_integrity"] = decoded_digests(decoded_images, decoded_labels, logits)
        else:
            packet["decoded_integrity"] = decoded_digests(decoded_images, decoded_labels)
        return packet, decoded_images, decoded_labels

    def test_accepts_exact_5x20_packet_and_alignment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = self._args(root)
            packet, images, labels = self._packet(root, args, 0)
            proof = validate_strict_dkp_packet(
                args,
                packet,
                images,
                labels,
                sender_agent=0,
                sender_model=MODELS[0],
                require_sender_logits=True,
            )
            self.assertEqual(proof["checkpoint_sha256"], file_sha256(root / "expert_agent_0.pt"))
            self.assertEqual(len(proof["decoded_alignment_sha256"]), 64)

    def test_rejects_logit_class_order_or_alignment_tampering(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = self._args(root)
            packet, images, labels = self._packet(root, args, 0)
            packet["sender_logit_class_ids"] = packet["sender_logit_class_ids"].flip(0)
            with self.assertRaisesRegex(ValueError, "sender logit class order"):
                validate_strict_dkp_packet(
                    args,
                    packet,
                    images,
                    labels,
                    sender_agent=0,
                    sender_model=MODELS[0],
                    require_sender_logits=True,
                )

            packet, images, labels = self._packet(root, args, 0)
            packet["sender_logits"][0, 0] += 1
            with self.assertRaisesRegex(ValueError, "sender_logits_sha256"):
                validate_strict_dkp_packet(
                    args,
                    packet,
                    images,
                    labels,
                    sender_agent=0,
                    sender_model=MODELS[0],
                    require_sender_logits=True,
                )

    def test_legacy_v1_rejects_non_seed0_partition_even_when_disjoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = self._args(Path(temp_dir))
            args.agent_class_split[0], args.agent_class_split[1] = (
                args.agent_class_split[1],
                args.agent_class_split[0],
            )
            with self.assertRaisesRegex(ValueError, "legacy CIFAR-100 seed0 v1"):
                validate_strict_partition(args)

    def test_dynamic_v2_accepts_balanced_5_10_20_agent_contracts(self):
        for agent_count in (5, 10, 20):
            with self.subTest(agent_count=agent_count):
                contract = resolve_strict_dkp_contract(
                    self._dynamic_config(agent_count)
                )
                self.assertEqual(contract.version, "v2")
                self.assertEqual(contract.agent_count, agent_count)
                self.assertEqual(contract.classes_per_agent, 100 // agent_count)
                self.assertEqual(
                    contract.decoded_per_sender,
                    (100 // agent_count) * 40,
                )

    def test_full_real_sender_count_scales_with_owned_classes(self):
        for agent_count in (5, 10, 20):
            with self.subTest(agent_count=agent_count):
                per_agent = 100 // agent_count
                args = SimpleNamespace(
                    dataset="cifar100",
                    agent_class_split={
                        agent_id: list(range(agent_id * per_agent, (agent_id + 1) * per_agent))
                        for agent_id in range(agent_count)
                    },
                )
                self.assertEqual(
                    _expected_full_real_sender_raw(args, agent_count - 1),
                    per_agent * 500,
                )

    def test_dynamic_v2_rejects_agent_count_or_factor_drift(self):
        config = self._dynamic_config(10)
        config["agents"]["num_agents"] = 9
        with self.assertRaisesRegex(ValueError, "num_agents"):
            resolve_strict_dkp_contract(config)

        config = self._dynamic_config(20)
        config["distillation"]["factor"] = 1
        with self.assertRaisesRegex(ValueError, "factor=2"):
            resolve_strict_dkp_contract(config)

    def test_manifest_requires_five_complete_unique_senders_and_current_hashes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = self._args(root)
            rows = []
            for agent_id in range(5):
                packet, _, _ = self._packet(root, args, agent_id)
                packet_path = root / f"packet_{agent_id}.pt"
                torch.save(packet, packet_path)
                integrity = packet["decoded_integrity"]
                rows.append(
                    {
                        "sender_agent": str(agent_id),
                        "sender_model": MODELS[agent_id],
                        "classes": ",".join(str(value) for value in args.agent_class_split[agent_id]),
                        "method": "DSDM",
                        "ipc": "10",
                        "protocol": STRICT_DKP_PROTOCOL,
                        "complete": "true",
                        "packet_path": str(packet_path),
                        "packet_sha256": file_sha256(packet_path),
                        "pool_protocol": packet["meta"]["pool_protocol"],
                        "pool_source_sha256": packet["meta"]["pool_source_sha256"],
                        "pool_source_best_snapshot_sha256": packet["meta"][
                            "pool_source_best_snapshot_sha256"
                        ],
                        "decoded_images_sha256": integrity["decoded_images_sha256"],
                        "decoded_labels_sha256": integrity["decoded_labels_sha256"],
                        "sender_logits_sha256": integrity["sender_logits_sha256"],
                        "decoded_alignment_sha256": integrity["decoded_alignment_sha256"],
                        "expert_checkpoint_sha256": packet["sender_logit_teacher"]["checkpoint_sha256"],
                    }
                )
            self.assertTrue(validate_strict_manifest_rows(args, rows, "dsdm"))
            with self.assertRaisesRegex(ValueError, r"exactly (?:five|5)"):
                validate_strict_manifest_rows(args, rows[:-1], "dsdm")
            rows[0]["decoded_images_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "decoded-image digest"):
                validate_strict_manifest_rows(args, rows, "dsdm")
            rows[0]["decoded_images_sha256"] = torch.load(
                rows[0]["packet_path"], map_location="cpu", weights_only=False
            )["decoded_integrity"]["decoded_images_sha256"]
            rows[0]["packet_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "manifest packet SHA"):
                validate_strict_manifest_rows(args, rows, "dsdm")

    def test_ce_only_packet_may_omit_logits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = self._args(root, use_sender_logits=False)
            packet, images, labels = self._packet(root, args, 0, with_logits=False)
            proof = validate_strict_dkp_packet(
                args,
                packet,
                images,
                labels,
                sender_agent=0,
                sender_model=MODELS[0],
                require_sender_logits=False,
            )
            self.assertIsNone(proof["checkpoint_sha256"])

    def test_strict_manifest_builder_emits_rich_complete_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = self._args(root)
            rows = []
            for agent_id in range(5):
                packet, _, _ = self._packet(root, args, agent_id)
                source_path = root / f"derived_packet_{agent_id}.pt"
                torch.save(packet, source_path)
                rows.append(register_agent_packet(args, agent_id, source_path, "dsdm"))
            manifest_path = write_packet_manifest(args, rows, "dsdm")
            self.assertTrue(manifest_path.is_file())
            loaded = read_packet_manifest(args, "dsdm")
            self.assertEqual(len(loaded), 5)
            self.assertTrue(all(row["complete"] == "true" for row in loaded))
            self.assertTrue(all(len(row["packet_sha256"]) == 64 for row in loaded))
            self.assertTrue(all(len(row["decoded_alignment_sha256"]) == 64 for row in loaded))

    def test_receiver_init_fr_and_sender_teacher_share_checkpoint_sha(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = self._args(root)
            args.use_logits = True
            checkpoint_dir = (
                Path(args.output_root)
                / args.run_name
                / "agents"
                / "agent_0"
                / "checkpoints"
            )
            checkpoint_dir.mkdir(parents=True)
            checkpoint_path = checkpoint_dir / "expert_model.pt"
            checkpoint_path.write_bytes(b"one-receiver-expert")
            checkpoint_sha = file_sha256(checkpoint_path)
            (checkpoint_dir / "expert_manifest.json").write_text(
                json.dumps(
                    {
                        "expert_sha256": checkpoint_sha,
                        "agent_id": 0,
                        "global_output_dim": 100,
                        "active_class_ids": args.agent_class_split[0],
                    }
                ),
                encoding="utf-8",
            )
            provenance_dir = Path(args.output_root) / args.run_name / "provenance"
            provenance_dir.mkdir(parents=True)
            (provenance_dir / "expert_reuse_manifest.json").write_text(
                json.dumps(
                    {
                        "agents": [
                            {
                                "agent_id": 0,
                                "artifacts": {"expert_model.pt": {"sha256": checkpoint_sha}},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            rows = [{"sender_agent": "0", "expert_checkpoint_sha256": checkpoint_sha}]
            self.assertEqual(
                validate_receiver_expert_provenance(args, 0, rows, checkpoint_path),
                checkpoint_sha,
            )
            rows[0]["expert_checkpoint_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "initialization/FR/sender-logit"):
                validate_receiver_expert_provenance(args, 0, rows, checkpoint_path)


if __name__ == "__main__":
    unittest.main()
