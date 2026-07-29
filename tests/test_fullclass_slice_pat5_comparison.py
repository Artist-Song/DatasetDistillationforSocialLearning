import tempfile
import unittest
from pathlib import Path

import torch

from scripts.compare_fullclass_slice_pat5_conv3 import load_packet_slice


class FullClassSlicePAT5ComparisonTest(unittest.TestCase):
    def test_slice_preserves_global_labels_and_requested_class_order(self):
        with tempfile.TemporaryDirectory(dir=".") as temp_dir:
            path = Path(temp_dir) / "packet.pt"
            images = torch.arange(12 * 3 * 2 * 2, dtype=torch.float32).reshape(12, 3, 2, 2)
            labels = torch.tensor([0] * 3 + [1] * 3 + [2] * 3 + [3] * 3)
            torch.save(
                {
                    "images": images,
                    "labels": labels,
                    "class_ids": [0, 1, 2, 3],
                    "factor": 2,
                    "decode_type": "single",
                },
                path,
            )
            sliced_images, sliced_labels, metadata = load_packet_slice(path, [3, 1], ipc=3)
            self.assertEqual(tuple(sliced_images.shape), (6, 3, 2, 2))
            self.assertEqual(sliced_labels.tolist(), [3, 3, 3, 1, 1, 1])
            self.assertEqual(metadata["raw_images_before_slice"], 12)
            self.assertEqual(metadata["raw_images_after_slice"], 6)
            self.assertTrue(torch.equal(sliced_images[:3], images[9:12]))


if __name__ == "__main__":
    unittest.main()
