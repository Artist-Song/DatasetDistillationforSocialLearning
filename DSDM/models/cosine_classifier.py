import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineClassifier(nn.Module):
    """Bias-free cosine classifier with one learnable positive global scale."""

    is_cosine_classifier = True

    def __init__(self, in_features, out_features, scale_init=10.0):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        initial_scale = float(scale_init)
        if not math.isfinite(initial_scale) or initial_scale <= 0.0:
            raise ValueError(f"scale_init must be finite and positive: {scale_init}")

        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.log_scale = nn.Parameter(
            torch.tensor(math.log(math.expm1(initial_scale)), dtype=torch.float32)
        )
        self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    @property
    def scale(self):
        return F.softplus(self.log_scale)

    @property
    def scale_parameter(self):
        """Expose the unconstrained scale parameter for optimizer grouping."""
        return self.log_scale

    def forward(self, features):
        normalized_features = F.normalize(features, p=2, dim=1)
        normalized_weight = F.normalize(self.weight, p=2, dim=1)
        return self.scale * F.linear(normalized_features, normalized_weight)


def build_classifier(in_features, out_features, classifier_type="linear", scale_init=10.0):
    """Build a legacy linear head or the opt-in cosine head."""
    resolved = str(classifier_type).strip().lower()
    if resolved in {"linear", ""}:
        return nn.Linear(int(in_features), int(out_features))
    if resolved == "cosine":
        return CosineClassifier(in_features, out_features, scale_init=scale_init)
    raise ValueError(f"unsupported classifier_type: {classifier_type}")


def get_cosine_classifier(model):
    """Return the model's unique cosine head, independent of backbone layout."""
    matches = [
        module
        for module in model.modules()
        if bool(getattr(module, "is_cosine_classifier", False))
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one cosine classifier, found {len(matches)}")
    return matches[0]


def get_output_classifier(model):
    """Return the terminal linear or cosine classifier used by a task model."""
    supported = (nn.Linear, CosineClassifier)
    for attribute in ("fc", "classifier"):
        candidate = getattr(model, attribute, None)
        if isinstance(candidate, supported):
            return candidate
        if isinstance(candidate, nn.Sequential):
            matches = [module for module in candidate.modules() if isinstance(module, supported)]
            if matches:
                return matches[-1]

    matches = [module for module in model.modules() if isinstance(module, supported)]
    if not matches:
        raise ValueError("model does not contain a supported output classifier")
    return matches[-1]


def get_output_classifier_type(model):
    """Return the runtime task-head type without trusting configuration metadata."""
    classifier = get_output_classifier(model)
    return "cosine" if isinstance(classifier, CosineClassifier) else "linear"


def get_classifier_weight(model):
    """Return the stored classifier weight parameter."""
    return get_cosine_classifier(model).weight


def get_normalized_classifier_weight(model):
    """Return normalized classifier rows without mutating the stored parameter."""
    return F.normalize(get_classifier_weight(model), p=2, dim=1)


def set_classifier_weight_rows(model, class_ids, prototypes):
    """Normalize and copy prototype rows into a cosine classifier."""
    head = get_cosine_classifier(model)
    ids = [int(class_id) for class_id in class_ids]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"class_ids must be non-empty and unique: {ids}")
    if min(ids) < 0 or max(ids) >= head.out_features:
        raise ValueError(
            f"class_ids outside classifier range [0, {head.out_features}): {ids}"
        )

    values = torch.as_tensor(prototypes, device=head.weight.device, dtype=head.weight.dtype)
    expected_shape = (len(ids), head.in_features)
    if tuple(values.shape) != expected_shape:
        raise ValueError(
            f"prototype shape mismatch: actual={tuple(values.shape)} expected={expected_shape}"
        )
    if not torch.isfinite(values).all():
        raise ValueError("prototype rows contain non-finite values")
    norms = values.norm(p=2, dim=1)
    if torch.any(norms <= 0):
        raise ValueError("prototype rows must have non-zero norm")

    normalized = F.normalize(values, p=2, dim=1)
    row_index = torch.tensor(ids, device=head.weight.device, dtype=torch.long)
    with torch.no_grad():
        head.weight.index_copy_(0, row_index, normalized)


def sgd_parameter_groups(model, weight_decay):
    """Exclude cosine scale parameters from weight decay."""
    scale_parameter_ids = {
        id(module.log_scale)
        for module in model.modules()
        if bool(getattr(module, "is_cosine_classifier", False))
    }
    decay_parameters = []
    scale_parameters = []
    for parameter in model.parameters():
        if id(parameter) in scale_parameter_ids:
            scale_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)

    groups = [{"params": decay_parameters, "weight_decay": float(weight_decay)}]
    if scale_parameters:
        groups.append({"params": scale_parameters, "weight_decay": 0.0})
    return groups
