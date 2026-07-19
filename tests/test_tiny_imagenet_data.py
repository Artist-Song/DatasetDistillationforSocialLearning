import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tiny_imagenet_data import TinyImageNetDataset, load_tiny_imagenet_leaked_validation_paths


class TinyImageNetDataTest(unittest.TestCase):
    def _build_fixture(self, root: Path) -> None:
        wnids = [f"n{index:08d}" for index in range(200)]
        root.mkdir(parents=True)
        (root / "wnids.txt").write_text("\n".join(wnids) + "\n", encoding="utf-8")
        (root / "val" / "images").mkdir(parents=True)
        annotations = []
        for class_id, wnid in enumerate(wnids):
            train_dir = root / "train" / wnid / "images"
            train_dir.mkdir(parents=True)
            train_name = f"{wnid}_0.JPEG"
            Image.new("RGB", (64, 64), (class_id % 255, 0, 0)).save(train_dir / train_name)
            val_name = f"val_{class_id}.JPEG"
            Image.new("RGB", (64, 64), (0, class_id % 255, 0)).save(root / "val" / "images" / val_name)
            annotations.append(f"{val_name}\t{wnid}\t0\t0\t63\t63")
        (root / "val" / "val_annotations.txt").write_text(
            "\n".join(reversed(annotations)) + "\n", encoding="utf-8"
        )

    def test_global_mapping_is_preserved_for_active_classes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "tiny-imagenet-200"
            self._build_fixture(root)
            train = TinyImageNetDataset(root, "train", active_class_ids=[3, 17])
            val = TinyImageNetDataset(root, "val", active_class_ids=[3, 17])
            self.assertEqual(train.targets, [3, 17])
            self.assertEqual(sorted(val.targets), [3, 17])
            self.assertEqual(train.nclass, 200)
            self.assertEqual(val.class_to_idx["n00000017"], 17)

    def test_integrity_report_exclusions_are_applied_without_remapping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "tiny-imagenet-200"
            self._build_fixture(root)
            excluded = root / "val" / "images" / "val_17.JPEG"
            report_path = Path(tmpdir) / "integrity.json"
            report_path.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "exact_duplicates": {"leaked_validation_paths": [str(excluded)]},
                    }
                ),
                encoding="utf-8",
            )

            excluded_paths = load_tiny_imagenet_leaked_validation_paths(report_path)
            val = TinyImageNetDataset(root, "val", excluded_paths=excluded_paths)

            self.assertEqual(len(val), 199)
            self.assertNotIn(17, val.targets)
            self.assertEqual(val.class_to_idx["n00000018"], 18)


if __name__ == "__main__":
    unittest.main()
