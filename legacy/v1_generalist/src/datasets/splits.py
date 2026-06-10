"""
Dataset split helpers for partial-known social packet experiments.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence

from torch.utils.data import Subset


@dataclass(frozen=True)
class AgentClassSplit:
    agent_id: int
    known: List[int]
    missing: List[int]


def make_cifar10_partial6_splits() -> Dict[int, AgentClassSplit]:
    fixed_known = {
        0: [0, 1, 2, 3, 4, 5],
        1: [2, 3, 4, 5, 6, 7],
        2: [4, 5, 6, 7, 8, 9],
        3: [0, 1, 6, 7, 8, 9],
        4: [0, 1, 2, 3, 8, 9],
    }
    all_classes = set(range(10))
    return {
        agent_id: AgentClassSplit(
            agent_id=agent_id,
            known=known,
            missing=sorted(all_classes.difference(known)),
        )
        for agent_id, known in fixed_known.items()
    }


def get_partial_split(split_name: str) -> Dict[int, AgentClassSplit]:
    if split_name == "cifar10_partial6":
        return make_cifar10_partial6_splits()
    raise ValueError(f"unknown partial split: {split_name}")


def subset_by_classes(dataset, class_ids: Sequence[int]) -> Subset:
    class_set = set(class_ids)
    indices = [idx for idx, target in enumerate(dataset.targets) if target in class_set]
    return Subset(dataset, indices)


def class_indices(dataset, class_ids: Sequence[int]) -> Dict[int, List[int]]:
    class_set = set(class_ids)
    indices = {class_id: [] for class_id in class_ids}
    for idx, target in enumerate(dataset.targets):
        if target in class_set:
            indices[target].append(idx)
    return indices
