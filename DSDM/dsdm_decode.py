"""Lightweight deterministic decode utilities shared by DSDM and packet consumers."""

from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F


def decode_zoom(img, target, factor, size=-1):
    if size == -1:
        size = img.shape[-1]
    resize = nn.Upsample(size=size, mode="bilinear")

    height = img.shape[-1]
    remained = height % factor
    if remained > 0:
        img = F.pad(img, pad=(0, factor - remained, 0, factor - remained), value=0.5)
    crop_size = ceil(height / factor)
    crop_count = factor**2

    cropped = []
    for row in range(factor):
        for column in range(factor):
            height_offset = row * crop_size
            width_offset = column * crop_size
            cropped.append(
                img[
                    :,
                    :,
                    height_offset : height_offset + crop_size,
                    width_offset : width_offset + crop_size,
                ]
            )
    decoded_data = resize(torch.cat(cropped))
    decoded_target = torch.cat([target for _ in range(crop_count)])
    return decoded_data, decoded_target


def decode_zoom_multi(img, target, factor_max):
    decoded_data = []
    decoded_target = []
    for factor in range(1, factor_max + 1):
        data_for_factor, target_for_factor = decode_zoom(img, target, factor)
        decoded_data.append(data_for_factor)
        decoded_target.append(target_for_factor)
    return torch.cat(decoded_data), torch.cat(decoded_target)


def decode_fn(data, target, factor, decode_type, bound=128):
    # ``bound`` is retained for exact compatibility with the original DSDM API.
    del bound
    if factor > 1:
        if decode_type == "multi":
            data, target = decode_zoom_multi(data, target, factor)
        else:
            data, target = decode_zoom(data, target, factor)
    return data, target
