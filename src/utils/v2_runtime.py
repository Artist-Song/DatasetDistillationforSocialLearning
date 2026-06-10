"""Runtime helpers shared by v2 entry points."""

import torch


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device_name)
        print(f"WARNING: requested {device_name}, but CUDA is unavailable; falling back to cpu")
        return torch.device("cpu")

    return torch.device(device_name)
