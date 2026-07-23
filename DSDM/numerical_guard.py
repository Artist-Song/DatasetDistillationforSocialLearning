import math

import torch


def ensure_finite_tensor(tensor, name, *, iteration=None, class_id=None, guide_idx=None):
    """Raise with DSDM context before a non-finite tensor can contaminate artifacts."""
    finite_mask = torch.isfinite(tensor)
    if bool(finite_mask.all().item()):
        return
    context = []
    if iteration is not None:
        context.append(f"iteration={int(iteration)}")
    if class_id is not None:
        context.append(f"class_id={int(class_id)}")
    if guide_idx is not None:
        context.append(f"guide_idx={int(guide_idx)}")
    nonfinite = int((~finite_mask).sum().item())
    suffix = f" ({', '.join(context)})" if context else ""
    raise FloatingPointError(f"non-finite {name}: count={nonfinite}{suffix}")


def clip_and_validate_gradients(parameters, max_norm, *, iteration, class_id, guide_idx):
    """Validate gradients and optionally clip their global norm."""
    parameters = [parameter for parameter in parameters if parameter.grad is not None]
    if not parameters:
        raise RuntimeError(
            "DSDM synthetic parameters have no gradients "
            f"(iteration={int(iteration)}, class_id={int(class_id)}, guide_idx={int(guide_idx)})"
        )
    for parameter in parameters:
        ensure_finite_tensor(
            parameter.grad,
            "synthetic gradient",
            iteration=iteration,
            class_id=class_id,
            guide_idx=guide_idx,
        )

    clip_limit = float(max_norm or 0.0)
    effective_limit = clip_limit if clip_limit > 0 else float("inf")
    total_norm = torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm=effective_limit,
        error_if_nonfinite=True,
    )
    total_norm_value = float(total_norm.detach().item())
    if not math.isfinite(total_norm_value):
        raise FloatingPointError(
            "non-finite synthetic gradient norm "
            f"(iteration={int(iteration)}, class_id={int(class_id)}, guide_idx={int(guide_idx)})"
        )
    return total_norm_value, bool(clip_limit > 0 and total_norm_value > clip_limit)
