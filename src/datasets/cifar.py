"""
CIFAR dataset helpers for local expert pretraining.

This module keeps the first data path intentionally small:
- load CIFAR10 / CIFAR100 train split
- build direct class splits such as CIFAR10-5-2
- return class-filtered subsets for each agent
"""

from typing import List, Sequence

from torch.utils.data import Subset
from torchvision import datasets, transforms


def build_cifar_dataset(
    name: str,
    root: str,
    train: bool,
    image_size=(32, 32),
    download: bool = True,
):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
    )

    name = name.lower()
    if name == "cifar10":
        return datasets.CIFAR10(root=root, train=train, transform=transform, download=download)
    if name == "cifar100":
        return datasets.CIFAR100(root=root, train=train, transform=transform, download=download)

    raise ValueError(f"Unsupported CIFAR dataset: {name}")


def build_cifar_train_dataset(name: str, root: str, image_size=(32, 32), download: bool = True):
    return build_cifar_dataset(
        name=name,
        root=root,
        train=True,
        image_size=image_size,
        download=download,
    )


def build_cifar_test_dataset(name: str, root: str, image_size=(32, 32), download: bool = True):
    return build_cifar_dataset(
        name=name,
        root=root,
        train=False,
        image_size=image_size,
        download=download,
    )


def make_direct_class_splits(num_classes: int, num_agents: int, classes_per_agent: int) -> List[List[int]]:
    expected_classes = num_agents * classes_per_agent
    if expected_classes != num_classes:
        raise ValueError(
            "direct split requires num_agents * classes_per_agent == num_classes, "
            f"got {num_agents} * {classes_per_agent} != {num_classes}"
        )

    class_ids = list(range(num_classes))
    return [
        class_ids[start : start + classes_per_agent]
        for start in range(0, num_classes, classes_per_agent)
    ]


def subset_by_classes(dataset, class_ids: Sequence[int]) -> Subset:
    class_set = set(class_ids)
    indices = [idx for idx, target in enumerate(dataset.targets) if target in class_set]
    return Subset(dataset, indices)
