"""Communication accounting shared by external baseline adapters."""

from __future__ import annotations

from collections.abc import Mapping


def desa_communication_accounting(
    anchor_counts: Mapping[int, int],
    class_counts: Mapping[int, int],
    rounds: int,
    *,
    logit_element_bytes: int = 4,
) -> dict:
    """Count one-time DeSA anchors and round-wise owner-logit transmissions."""
    owners = sorted(int(owner) for owner in anchor_counts)
    if owners != sorted(int(owner) for owner in class_counts):
        raise ValueError("DeSA anchor owners and class owners must match")
    if len(owners) < 2:
        raise ValueError("DeSA communication requires at least two agents")
    if int(rounds) <= 0:
        raise ValueError("DeSA rounds must be positive")
    if int(logit_element_bytes) <= 0:
        raise ValueError("logit_element_bytes must be positive")

    anchors = {owner: int(anchor_counts[owner]) for owner in owners}
    classes = {owner: int(class_counts[owner]) for owner in owners}
    if any(count <= 0 for count in anchors.values()):
        raise ValueError("every DeSA sender must provide at least one anchor")
    if any(count <= 0 for count in classes.values()):
        raise ValueError("every DeSA sender must own at least one class")

    images_per_receiver = {}
    logit_bytes_per_receiver = {}
    for receiver in owners:
        external_owners = [owner for owner in owners if owner != receiver]
        images_per_receiver[receiver] = sum(anchors[owner] for owner in external_owners)
        logit_bytes_per_receiver[receiver] = int(rounds) * sum(
            anchors[owner] * classes[owner] * int(logit_element_bytes)
            for owner in external_owners
        )

    return {
        "unique_sender_images": sum(anchors.values()),
        "external_images_per_receiver": images_per_receiver,
        "iterative_owner_logit_bytes_per_receiver": logit_bytes_per_receiver,
        "receiver_incidence_images": sum(images_per_receiver.values()),
        "iterative_owner_logit_bytes_all_agents": sum(logit_bytes_per_receiver.values()),
        "rounds": int(rounds),
        "logit_element_bytes": int(logit_element_bytes),
    }
