#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tiny_imagenet_data import (
    TinyImageNetDataset,
    load_tiny_imagenet_val_annotations,
    load_tiny_imagenet_wnids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Tiny-ImageNet archive and extracted data")
    parser.add_argument("--data-root", default="data/tiny-imagenet-200")
    parser.add_argument("--archive", default="data/tiny-imagenet-200.zip")
    parser.add_argument(
        "--output",
        default="outputs/tinyimagenet_data_validation_20260718/data_integrity.json",
    )
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(item: tuple[str, int, Path]) -> dict:
    split, target, path = item
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    with Image.open(BytesIO(data)) as image:
        image.load()
        size = tuple(image.size)
        mode = image.mode
        image_format = image.format
    return {
        "split": split,
        "target": int(target),
        "path": str(path),
        "sha256": digest,
        "size": size,
        "mode": mode,
        "format": image_format,
    }


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, mode="w", encoding="utf-8", delete=False
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def validate_boxes(path: Path, expected_files: set[str], errors: list[str]) -> None:
    if not path.is_file():
        errors.append(f"missing box annotations: {path}")
        return
    annotated: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split("\t")
        if len(fields) != 5:
            errors.append(f"malformed box annotation: {path}:{line_number}")
            continue
        filename = fields[0]
        annotated.add(filename)
        try:
            x0, y0, x1, y1 = (int(value) for value in fields[1:])
        except ValueError:
            errors.append(f"non-integer box annotation: {path}:{line_number}")
            continue
        if not (0 <= x0 <= x1 <= 63 and 0 <= y0 <= y1 <= 63):
            errors.append(f"out-of-range box annotation: {path}:{line_number}")
    if annotated != expected_files:
        errors.append(
            f"box/image filename mismatch: {path} annotated={len(annotated)} images={len(expected_files)}"
        )


def main() -> None:
    args = parse_args()
    data_root = (ROOT / args.data_root).resolve()
    archive = (ROOT / args.archive).resolve()
    output = (ROOT / args.output).resolve()
    errors: list[str] = []
    warnings: list[str] = []

    wnids = load_tiny_imagenet_wnids(data_root)
    if len(wnids) != 200:
        errors.append(f"expected 200 wnids, found {len(wnids)}")

    train_dataset = TinyImageNetDataset(data_root, "train")
    val_dataset = TinyImageNetDataset(data_root, "val")
    train_counts = Counter(train_dataset.targets)
    val_counts = Counter(val_dataset.targets)
    if set(train_counts) != set(range(200)) or any(value != 500 for value in train_counts.values()):
        errors.append("training split is not exactly 200 classes x 500 images")
    if set(val_counts) != set(range(200)) or any(value != 50 for value in val_counts.values()):
        errors.append("validation split is not exactly 200 classes x 50 images")

    train_dirs = {path.name for path in (data_root / "train").iterdir() if path.is_dir()}
    if train_dirs != set(wnids):
        errors.append("train directory classes do not match wnids.txt")

    for wnid in wnids:
        image_dir = data_root / "train" / wnid / "images"
        filenames = {path.name for path in image_dir.glob("*.JPEG")}
        validate_boxes(data_root / "train" / wnid / f"{wnid}_boxes.txt", filenames, errors)

    val_annotations = load_tiny_imagenet_val_annotations(data_root)
    if len(val_annotations) != 10000:
        errors.append(f"expected 10000 validation annotations, found {len(val_annotations)}")
    if set(val_annotations.values()) != set(wnids):
        errors.append("validation annotations do not cover exactly the wnids.txt classes")
    val_files = {path.name for path in (data_root / "val" / "images").glob("*.JPEG")}
    if val_files != set(val_annotations):
        errors.append("validation image filenames do not match val_annotations.txt")

    items = [
        *(('train', target, path) for path, target in train_dataset.samples),
        *(('val', target, path) for path, target in val_dataset.samples),
    ]
    size_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    format_counts: Counter[str] = Counter()
    hashes: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    decode_errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        for index, item in enumerate(executor.map(inspect_image, items), start=1):
            size_counts[f"{item['size'][0]}x{item['size'][1]}"] += 1
            mode_counts[item["mode"]] += 1
            format_counts[str(item["format"])] += 1
            hashes[item["sha256"]].append((item["split"], item["target"], item["path"]))
            if index % 10000 == 0:
                print(f"[tiny-data] decoded {index}/{len(items)}", flush=True)

    if size_counts != Counter({"64x64": 110000}):
        errors.append(f"unexpected image sizes: {dict(size_counts)}")
    if format_counts != Counter({"JPEG": 110000}):
        errors.append(f"unexpected image formats: {dict(format_counts)}")
    if mode_counts != Counter({"RGB": 110000}):
        warnings.append(f"non-RGB source images will be converted by the loader: {dict(mode_counts)}")

    duplicate_groups = [locations for locations in hashes.values() if len(locations) > 1]
    cross_split_duplicates = [
        locations for locations in duplicate_groups if len({location[0] for location in locations}) > 1
    ]
    within_split_duplicates = len(duplicate_groups) - len(cross_split_duplicates)
    if within_split_duplicates:
        warnings.append(f"within-split exact duplicate groups: {within_split_duplicates}")
    if cross_split_duplicates:
        warnings.append(f"train/val exact-image leakage groups: {len(cross_split_duplicates)}")
    same_label_duplicates = sum(
        len({location[1] for location in locations}) == 1 for locations in duplicate_groups
    )
    conflicting_label_duplicates = len(duplicate_groups) - same_label_duplicates
    leaked_validation_paths = sorted(
        location[2]
        for locations in cross_split_duplicates
        for location in locations
        if location[0] == "val"
    )

    test_images = list((data_root / "test" / "images").glob("*.JPEG"))
    if len(test_images) != 10000:
        errors.append(f"expected 10000 unlabeled test images, found {len(test_images)}")

    payload = {
        "passed": not errors,
        "data_root": str(data_root),
        "archive": {
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "md5": file_digest(archive, "md5"),
            "sha256": file_digest(archive, "sha256"),
            "zip_crc_test": "passed",
        },
        "class_mapping": {
            "source": "wnids.txt order",
            "num_classes": len(wnids),
            "first_wnid": wnids[0],
            "last_wnid": wnids[-1],
        },
        "splits": {
            "train": {"images": len(train_dataset), "per_class_min": min(train_counts.values()), "per_class_max": max(train_counts.values())},
            "val": {"images": len(val_dataset), "per_class_min": min(val_counts.values()), "per_class_max": max(val_counts.values())},
            "test_unlabeled": {"images": len(test_images)},
        },
        "decoded_images": len(items) - len(decode_errors),
        "image_sizes": dict(size_counts),
        "image_modes": dict(mode_counts),
        "image_formats": dict(format_counts),
        "exact_duplicates": {
            "within_split_groups": within_split_duplicates,
            "cross_split_groups": len(cross_split_duplicates),
            "same_label_groups": same_label_duplicates,
            "conflicting_label_groups": conflicting_label_duplicates,
            "strict_leakage_free": not cross_split_duplicates,
            "leaked_validation_paths": leaked_validation_paths,
            "cross_split_examples": cross_split_duplicates,
            "examples": duplicate_groups[:10],
        },
        "errors": errors + decode_errors,
        "warnings": warnings,
    }
    atomic_write_json(output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    print(f"[tiny-data] wrote {output}", flush=True)
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
