import json
import os
import tempfile
import unittest
from pathlib import Path

import torch

from output_manager import atomic_copyfile, atomic_torch_save, finalize_dsdm_packet, save_best_synthetic


class AtomicArtifactWriteTest(unittest.TestCase):
    def test_torch_save_breaks_target_link_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.pt"
            target = Path(tmpdir) / "target.pt"
            torch.save({"value": 1}, source)
            os.link(source, target)

            atomic_torch_save({"value": 2}, target)

            self.assertEqual(torch.load(source, map_location="cpu")["value"], 1)
            self.assertEqual(torch.load(target, map_location="cpu")["value"], 2)
            self.assertFalse(os.path.samefile(source, target))

    def test_copy_breaks_target_link_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            original = Path(tmpdir) / "original.bin"
            target = Path(tmpdir) / "target.bin"
            replacement = Path(tmpdir) / "replacement.bin"
            original.write_bytes(b"old")
            replacement.write_bytes(b"new")
            os.link(original, target)

            atomic_copyfile(replacement, target)

            self.assertEqual(original.read_bytes(), b"old")
            self.assertEqual(target.read_bytes(), b"new")
            self.assertFalse(os.path.samefile(original, target))

    def test_best_synthetic_keeps_immutable_iteration_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = type(
                "Args",
                (),
                {
                    "output_root": tmpdir,
                    "run_name": "run",
                    "dataset": "tinyimagenet",
                    "ipc": 10,
                    "pcbn_enabled": True,
                    "pcbn_weight": 1.5,
                    "pcbn_layers": "all",
                    "pcbn_normalize_layers": True,
                    "pcbn_hook_count": 20,
                },
            )()
            synset = type(
                "Synset",
                (),
                {
                    "data": torch.ones(2, 3, 4, 4),
                    "targets": torch.tensor([0, 1]),
                },
            )()

            save_best_synthetic(args, synset, best_acc=12.5, iteration=100)
            synset.data.zero_()
            save_best_synthetic(args, synset, best_acc=13.0, iteration=500)

            history_100 = Path(tmpdir) / "run" / "synthetic" / "history" / "best_iter_00100.pt"
            history_500 = Path(tmpdir) / "run" / "synthetic" / "history" / "best_iter_00500.pt"
            manifest = json.loads(
                (Path(tmpdir) / "run" / "synthetic" / "best_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(torch.equal(torch.load(history_100)["images"], torch.ones(2, 3, 4, 4)))
            self.assertTrue(torch.equal(torch.load(history_500)["images"], torch.zeros(2, 3, 4, 4)))
            self.assertEqual(manifest["iteration"], 500)
            self.assertEqual(manifest["best_acc"], 13.0)
            self.assertEqual(manifest["pcbn_hook_count"], 20)
            self.assertTrue(manifest["pcbn_normalize_layers"])

    def test_finalize_dsdm_packet_records_completion_and_numerics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            packet_path = Path(tmpdir) / "packet.pt"
            torch.save(
                {
                    "images": torch.ones(2, 3, 4, 4),
                    "labels": torch.tensor([0, 1]),
                    "meta": {"best_acc": 64.9},
                },
                packet_path,
            )
            finalize_dsdm_packet(
                packet_path,
                {
                    "completed_iterations": 10000,
                    "grad_clip_norm": 100.0,
                    "grad_clip_count": 1,
                    "max_grad_norm": 1234.0,
                },
            )
            packet = torch.load(packet_path, map_location="cpu")
            self.assertTrue(packet["meta"]["condense_complete"])
            self.assertEqual(packet["meta"]["completed_iterations"], 10000)
            self.assertEqual(packet["meta"]["grad_clip_count"], 1)


if __name__ == "__main__":
    unittest.main()
