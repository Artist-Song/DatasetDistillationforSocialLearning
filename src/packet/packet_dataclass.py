"""v2 social packet dataclass.

Packets in the v2 mainline carry image tensors, hard labels, and metadata only.
They intentionally do not carry soft targets, teacher logits, teacher
probabilities, gradients, or model parameters.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

import torch


@dataclass
class SocialPacket:
    """Hard-label image packet exchanged between v2 agents."""

    sender_id: int
    class_ids: Union[List[int], torch.Tensor]
    images: torch.Tensor
    hard_labels: torch.Tensor
    meta: Dict[str, Any] = field(default_factory=dict)

    def byte_stats(self) -> Dict[str, int]:
        bytes_images = self.images.nelement() * self.images.element_size()
        bytes_labels = self.hard_labels.nelement() * self.hard_labels.element_size()
        return {
            "bytes_images": bytes_images,
            "bytes_labels": bytes_labels,
            "bytes_total": bytes_images + bytes_labels,
        }
