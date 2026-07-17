import os
import tempfile
import unittest
from pathlib import Path

import torch

from output_manager import atomic_copyfile, atomic_torch_save


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


if __name__ == "__main__":
    unittest.main()
