import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import Dataset

from selection_methods import build_fast_packet


class TinyLabeledDataset(Dataset):
    def __init__(self):
        self.images = torch.arange(10 * 3 * 4 * 4, dtype=torch.float32).reshape(10, 3, 4, 4)
        self.targets = [3] * 5 + [7] * 5
        self.indices = list(range(100, 110))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.images[index], self.targets[index]


class FastSelectionTest(unittest.TestCase):
    def test_exact_ipc_and_cache_reuse(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                dataset="cifar100",
                ipc=2,
                fast_cache_root=tmpdir,
                fast_repo_path="unused-in-mocked-test",
                fast_commit="test-commit",
                fast_seed=0,
            )

            def fake_selector(**kwargs):
                self.assertEqual(kwargs["X"].shape, (5, 48))
                return np.array([1, 3]), kwargs["X"][[1, 3]], {}

            provenance = {
                "repo_path": "unused-in-mocked-test",
                "official_commit": "test-commit",
                "tracked_patch_present": True,
                "tracked_patch_sha256": "a" * 64,
            }
            with patch("selection_methods._fast_repo_provenance", return_value=provenance), patch(
                "selection_methods._load_fast_official_selector", return_value=fake_selector
            ) as loader:
                images, labels, classes, meta = build_fast_packet(args, TinyLabeledDataset())
                self.assertEqual(loader.call_count, 2)
                self.assertEqual(tuple(images.shape), (4, 3, 4, 4))
                self.assertEqual(labels.tolist(), [3, 3, 7, 7])
                self.assertEqual(classes, [3, 7])
                self.assertEqual(meta["selected_dataset_indices"], [101, 103, 106, 108])
                self.assertEqual(meta["cache_hits"], 0)
                self.assertEqual(meta["tracked_patch_sha256"], "a" * 64)

            with patch("selection_methods._fast_repo_provenance", return_value=provenance), patch(
                "selection_methods._load_fast_official_selector"
            ) as loader:
                _, labels, _, meta = build_fast_packet(args, TinyLabeledDataset())
                loader.assert_not_called()
                self.assertEqual(labels.tolist(), [3, 3, 7, 7])
                self.assertEqual(meta["cache_hits"], 2)
                self.assertEqual(len(list(Path(tmpdir).rglob("*.npz"))), 2)


if __name__ == "__main__":
    unittest.main()
