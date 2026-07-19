from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


TINY_IMAGENET_MEAN = (0.485, 0.456, 0.406)
TINY_IMAGENET_STD = (0.229, 0.224, 0.225)


def load_tiny_imagenet_wnids(root: str | Path) -> list[str]:
    root = Path(root)
    path = root / "wnids.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Tiny-ImageNet class list is missing: {path}")
    wnids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(wnids) != len(set(wnids)):
        raise ValueError(f"Tiny-ImageNet class list contains duplicates: {path}")
    return wnids


def load_tiny_imagenet_val_annotations(root: str | Path) -> dict[str, str]:
    path = Path(root) / "val" / "val_annotations.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Tiny-ImageNet validation annotations are missing: {path}")
    annotations: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split("\t")
        if len(fields) < 2:
            raise ValueError(f"Malformed validation annotation at {path}:{line_number}")
        filename, wnid = fields[:2]
        if filename in annotations:
            raise ValueError(f"Duplicate validation filename at {path}:{line_number}: {filename}")
        annotations[filename] = wnid
    return annotations


def load_tiny_imagenet_leaked_validation_paths(report_path: str | Path) -> set[Path]:
    """Load validation paths excluded by the repository's integrity gate."""
    report_path = Path(report_path)
    if not report_path.is_file():
        raise FileNotFoundError(f"Tiny-ImageNet integrity report is missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("passed"):
        raise ValueError(f"Tiny-ImageNet integrity report did not pass: {report_path}")
    paths = report.get("exact_duplicates", {}).get("leaked_validation_paths", [])
    return {Path(path).resolve() for path in paths}


class TinyImageNetDataset(Dataset):
    """Tiny-ImageNet loader that preserves the global wnids.txt label mapping."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform=None,
        active_class_ids: list[int] | None = None,
        excluded_paths: set[str | Path] | None = None,
    ):
        self.root = Path(root)
        self.split = str(split)
        self.transform = transform
        self.classes = load_tiny_imagenet_wnids(self.root)
        self.class_to_idx = {wnid: index for index, wnid in enumerate(self.classes)}
        if len(self.classes) != 200:
            raise ValueError(f"Tiny-ImageNet must contain 200 classes, found {len(self.classes)}")

        if active_class_ids is None:
            active = set(range(len(self.classes)))
        else:
            active = {int(class_id) for class_id in active_class_ids}
            invalid = sorted(active.difference(range(len(self.classes))))
            if invalid:
                raise ValueError(f"Invalid Tiny-ImageNet global class ids: {invalid}")

        samples: list[tuple[Path, int]] = []
        if self.split == "train":
            for class_id, wnid in enumerate(self.classes):
                if class_id not in active:
                    continue
                image_dir = self.root / "train" / wnid / "images"
                samples.extend((path, class_id) for path in sorted(image_dir.glob("*.JPEG")))
        elif self.split == "val":
            annotations = load_tiny_imagenet_val_annotations(self.root)
            image_dir = self.root / "val" / "images"
            for filename, wnid in sorted(annotations.items()):
                if wnid not in self.class_to_idx:
                    raise ValueError(f"Validation annotation references an unknown class: {wnid}")
                class_id = self.class_to_idx[wnid]
                if class_id in active:
                    samples.append((image_dir / filename, class_id))
        else:
            raise ValueError(f"Unsupported Tiny-ImageNet split: {self.split}")

        excluded = {Path(path).resolve() for path in (excluded_paths or set())}
        if excluded:
            samples = [(path, target) for path, target in samples if path.resolve() not in excluded]

        missing = [str(path) for path, _ in samples if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Tiny-ImageNet sample files are missing: {missing[:5]}")
        if not samples:
            raise ValueError(f"Tiny-ImageNet split is empty: split={self.split}")

        self.samples = samples
        self.targets = [target for _, target in samples]
        self.nclass = len(self.classes)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def build_tiny_imagenet_transforms(train: bool):
    normalize = transforms.Normalize(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD)
    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(64, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose([transforms.ToTensor(), normalize])
