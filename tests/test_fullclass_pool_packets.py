import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from fullclass_pool_packets import materialize_agent_packet, validate_pool_source
from packet_integrity import POOL_PROTOCOL, file_sha256


class FullClassPoolPacketTests(unittest.TestCase):
    def _fixture(self, root, *, complete=True):
        source_root = root / "source"
        packet_path = source_root / "agents/agent_0/packets/dsdm_packet.pt"
        best_path = source_root / "agents/agent_0/synthetic/data_best.pt"
        history_path = source_root / "agents/agent_0/synthetic/history/best_iter_00100.pt"
        manifest_path = source_root / "agents/agent_0/synthetic/best_manifest.json"
        packet_path.parent.mkdir(parents=True)
        history_path.parent.mkdir(parents=True)
        images = torch.arange(4 * 2 * 3 * 4 * 4, dtype=torch.float32).reshape(8, 3, 4, 4)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
        payload = {
            "images": images,
            "labels": labels,
            "class_ids": [0, 1, 2, 3],
            "source": "dsdm",
            "dataset": "fixture",
            "ipc": 2,
            "factor": 2,
            "decode_type": "single",
            "meta": {"condense_complete": complete, "completed_iterations": 100},
        }
        torch.save(payload, packet_path)
        best = {"images": images.clone(), "labels": labels.clone()}
        torch.save(best, best_path)
        torch.save(best, history_path)
        manifest_path.write_text(
            json.dumps(
                {
                    "best_acc": 55.5,
                    "iteration": 100,
                    "latest_best": str(best_path),
                    "history_snapshot": str(history_path),
                }
            ),
            encoding="utf-8",
        )
        return packet_path, manifest_path, images

    def test_materializes_requested_global_classes_in_config_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            packet_path, manifest_path, source_images = self._fixture(root)
            args = SimpleNamespace(output_root=str(root / "outputs"), run_name="target")
            catalog = {
                "dataset": "fixture",
                "num_classes": 4,
                "ipc": 2,
                "factor": 2,
                "decode_type": "single",
                "catalog_path": "fixture_catalog.yaml",
            }
            result = materialize_agent_packet(
                args,
                2,
                "model_b",
                [3, 1],
                {"packet_path": str(packet_path), "best_manifest": str(manifest_path)},
                catalog,
            )
            target = torch.load(root / "outputs/target/agents/agent_2/packets/dsdm_packet.pt", weights_only=False)
            self.assertEqual(target["class_ids"], [3, 1])
            self.assertEqual(target["labels"].tolist(), [3, 3, 1, 1])
            self.assertTrue(torch.equal(target["images"], torch.cat([source_images[6:8], source_images[2:4]])))
            self.assertTrue(target["meta"]["pool_reuse"])
            self.assertEqual(target["meta"]["sender_model"], "model_b")
            self.assertFalse(target["meta"]["guide_weights_communicated"])
            self.assertEqual(result["raw_images"], 4)

    def test_rejects_incomplete_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            packet_path, manifest_path, _ = self._fixture(root, complete=False)
            args = SimpleNamespace(output_root=str(root / "outputs"), run_name="target")
            catalog = {
                "dataset": "fixture",
                "num_classes": 4,
                "ipc": 2,
                "factor": 2,
                "decode_type": "single",
                "catalog_path": "fixture_catalog.yaml",
            }
            with self.assertRaisesRegex(ValueError, "not marked complete"):
                materialize_agent_packet(
                    args,
                    0,
                    "model_a",
                    [0, 1],
                    {"packet_path": str(packet_path), "best_manifest": str(manifest_path)},
                    catalog,
                )

    def test_strict_source_requires_exact_model_sha_and_best_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            packet_path, manifest_path, _ = self._fixture(root)
            packet = torch.load(packet_path, map_location="cpu", weights_only=False)
            packet["meta"]["sender_model"] = "resnet10_standard"
            torch.save(packet, packet_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            history_path = Path(manifest["history_snapshot"])
            source_spec = {
                "packet_path": str(packet_path),
                "best_manifest": str(manifest_path),
                "source_model_id": "resnet10_standard",
                "expected_packet_sha256": file_sha256(packet_path),
                "expected_best_iteration": 100,
                "expected_best_snapshot": str(history_path),
                "expected_best_snapshot_sha256": file_sha256(history_path),
                "expected_completed_iterations": 100,
            }
            catalog = {
                "schema_version": 2,
                "protocol": POOL_PROTOCOL,
                "strict_validation": True,
                "dataset": "fixture",
                "num_classes": 4,
                "ipc": 2,
                "factor": 2,
                "decode_type": "single",
            }
            validated = validate_pool_source(source_spec, catalog, "resnet10_standard")
            self.assertEqual(validated["source_model_id"], "resnet10_standard")
            with self.assertRaisesRegex(ValueError, "source model id mismatch"):
                validate_pool_source(source_spec, catalog, "resnet10")
            source_spec["expected_packet_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "packet SHA-256"):
                validate_pool_source(source_spec, catalog, "resnet10_standard")


if __name__ == "__main__":
    unittest.main()
